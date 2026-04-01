from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .decoder_pillow import OptTransDecoderPillow
from .encoder_pillow import OptTransEncoderPillow
from .video_camera_fallback import decode_camera_video_mode


@dataclass
class DecodedVideoFrame:
    frame_num: int
    total_frames: int
    data_len: int
    data: bytes
    source_index: int


class OptTransVideoTransport:
    def __init__(self):
        self.encoder = OptTransEncoderPillow()
        self.decoder = OptTransDecoderPillow()
        self.frame_size = (self.encoder.image_size, self.encoder.image_size)
        self._fast_decode_params = [
            (2, None, False, 0, 0),
            (3, None, False, 0, 0),
            (2, None, False, 0, 2),
        ]
        self._full_decode_params = [
            (3, None, False, 0, 0),
            (2, None, False, 0, 2),
            (4, None, False, 0, 0),
            (3, 128, False, 0, 0),
            (3, 128, True, 0, 0),
        ]
        self._candidate_quad_limit = 5
        self._tracked_retry_limit = 2
        self._module_inner_margins = (2, 1)
        self._video_mask_patterns = (1,)

    def _to_bgr_frame(self, image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _to_pil_image(self, frame_bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def _clone_decoded_frame(self, decoded_frame: DecodedVideoFrame, source_index: int) -> DecodedVideoFrame:
        return DecodedVideoFrame(
            frame_num=decoded_frame.frame_num,
            total_frames=decoded_frame.total_frames,
            data_len=decoded_frame.data_len,
            data=decoded_frame.data,
            source_index=source_index,
        )

    def _make_marker_frame(self, kind: str) -> np.ndarray:
        width, height = self.frame_size
        frame = np.full((height, width, 3), 255, dtype=np.uint8)

        if kind == "start":
            primary = (50, 180, 40)
            secondary = (255, 255, 255)
            accent = (20, 120, 20)
            label = "OPTTRANS START"
            icon = "play"
        elif kind == "end":
            primary = (40, 40, 210)
            secondary = (255, 255, 255)
            accent = (20, 20, 120)
            label = "OPTTRANS END"
            icon = "stop"
        else:
            raise ValueError(f"Unknown marker kind: {kind}")

        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), primary, thickness=-1)
        border = max(24, width // 24)
        cv2.rectangle(
            frame,
            (border, border),
            (width - border, height - border),
            secondary,
            thickness=border,
        )

        inner_margin = border * 3
        cv2.rectangle(
            frame,
            (inner_margin, inner_margin),
            (width - inner_margin, height - inner_margin),
            accent,
            thickness=max(2, border // 2),
        )

        stripe_step = max(18, width // 32)
        for x in range(0, width, stripe_step):
            cv2.line(frame, (x, 0), (0, x), secondary, thickness=2)
            cv2.line(frame, (width - 1, x), (x, height - 1), secondary, thickness=2)

        center_x = width // 2
        center_y = height // 2
        icon_size = width // 8
        if icon == "play":
            triangle = np.array(
                [
                    (center_x - icon_size // 2, center_y - icon_size),
                    (center_x - icon_size // 2, center_y + icon_size),
                    (center_x + icon_size, center_y),
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(frame, triangle, secondary)
        else:
            cv2.rectangle(
                frame,
                (center_x - icon_size, center_y - icon_size),
                (center_x + icon_size, center_y + icon_size),
                secondary,
                thickness=-1,
            )

        text_scale = width / 900.0
        cv2.putText(
            frame,
            label,
            (width // 7, height - height // 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            secondary,
            thickness=4,
            lineType=cv2.LINE_AA,
        )
        return frame

    def _marker_mean_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        thumb = cv2.resize(frame_bgr, (48, 48), interpolation=cv2.INTER_AREA)
        return thumb.mean(axis=(0, 1))

    def _is_start_marker(self, frame_bgr: np.ndarray) -> bool:
        blue, green, red = self._marker_mean_bgr(frame_bgr)
        return green > red + 35 and green > blue + 35

    def _is_end_marker(self, frame_bgr: np.ndarray) -> bool:
        blue, green, red = self._marker_mean_bgr(frame_bgr)
        return red > green + 35 and red > blue + 35

    def _select_video_codec(self, output_video: str) -> int:
        suffix = Path(output_video).suffix.lower()
        if suffix == ".mp4":
            return cv2.VideoWriter_fourcc(*"mp4v")
        if suffix == ".avi":
            return cv2.VideoWriter_fourcc(*"MJPG")
        return cv2.VideoWriter_fourcc(*"XVID")

    def _build_data_frames(self, data: bytes) -> list[np.ndarray]:
        total_frames = (len(data) + self.encoder.data_per_frame - 1) // self.encoder.data_per_frame
        frames: list[np.ndarray] = []
        for frame_num in range(total_frames):
            start = frame_num * self.encoder.data_per_frame
            end = min((frame_num + 1) * self.encoder.data_per_frame, len(data))
            frame_data = data[start:end]
            image = self.encoder.build_image(
                frame_data,
                frame_num=frame_num,
                total_frames=total_frames,
                mask_patterns=self._video_mask_patterns,
            )
            frames.append(self._to_bgr_frame(image))
        return frames

    def _frame_signature(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        return thumb >= int(thumb.mean())

    def _signature_distance(self, left: np.ndarray, right: np.ndarray) -> int:
        return int(np.count_nonzero(left != right))

    def _lookup_cached_frame(
        self,
        signature: np.ndarray,
        decode_cache: list[tuple[np.ndarray, DecodedVideoFrame]],
        *,
        max_distance: int = 4,
    ) -> DecodedVideoFrame | None:
        for cached_signature, decoded_frame in reversed(decode_cache):
            if self._signature_distance(signature, cached_signature) <= max_distance:
                return self._clone_decoded_frame(decoded_frame, decoded_frame.source_index)
        return None

    def _aligned_module_means(self, frame_bgr: np.ndarray, inner_margin: int) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        expected_height = self.frame_size[1]
        expected_width = self.frame_size[0]
        if gray.shape[0] != expected_height or gray.shape[1] != expected_width:
            gray = cv2.resize(gray, (expected_width, expected_height), interpolation=cv2.INTER_AREA)
        gray = np.ascontiguousarray(gray)

        total_modules = self.encoder.total_size
        module_size = self.encoder.module_size
        blocks = gray.reshape(total_modules, module_size, total_modules, module_size)
        margin = max(0, min(inner_margin, module_size // 2 - 1))
        if margin > 0:
            blocks = blocks[:, margin:module_size - margin, :, margin:module_size - margin]
        return blocks.mean(axis=(1, 3))

    def _decode_fast_aligned_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        thresholds: tuple[int, ...] | None = None,
        inner_margins: tuple[int, ...] | None = None,
    ) -> tuple[DecodedVideoFrame | None, str | None]:
        row_slice = slice(self.encoder.margin, self.encoder.margin + self.encoder.matrix_size)
        col_slice = slice(self.encoder.margin, self.encoder.margin + self.encoder.matrix_size)

        if inner_margins is None:
            inner_margins = self._module_inner_margins

        for inner_margin in inner_margins:
            module_means = self._aligned_module_means(frame_bgr, inner_margin)
            matrix_means = module_means[row_slice, col_slice]
            otsu_input = np.ascontiguousarray(matrix_means.astype(np.uint8))
            otsu_threshold, _ = cv2.threshold(
                otsu_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            candidate_thresholds = [int(otsu_threshold), 128, int(matrix_means.mean())]
            if thresholds is not None:
                candidate_thresholds.extend(int(value) for value in thresholds)
            seen_thresholds: set[int] = set()

            for threshold in candidate_thresholds:
                if threshold in seen_thresholds:
                    continue
                seen_thresholds.add(threshold)
                matrix = (matrix_means < threshold).astype(np.uint8).tolist()
                control_info = self.decoder._select_control_info(matrix)
                if control_info is None:
                    continue
                payload = self.decoder._decode_payload(matrix, control_info)
                if payload is None:
                    continue
                return (
                    DecodedVideoFrame(
                        frame_num=int(control_info.frame_num),
                        total_frames=int(control_info.total_frames),
                        data_len=int(control_info.data_len),
                        data=payload,
                        source_index=-1,
                    ),
                    f"fast-grid margin={inner_margin} threshold={threshold}",
                )

        return None, None

    def _candidate_quads(self, pil_image: Image.Image, limit: int | None = None):
        if limit is None:
            limit = self._candidate_quad_limit
        img_gray = np.array(pil_image.convert("L"))
        candidates = self.decoder._find_finder_candidates(img_gray)[:6]
        if len(candidates) < 4:
            return []

        scored_quads = []
        for combo in combinations(candidates, 4):
            ordered = self.decoder._order_candidate_quad(list(combo))
            if ordered is None:
                continue
            score = self.decoder._quad_geometry_score(ordered)
            if score is None:
                continue
            scored_quads.append((score, ordered))

        scored_quads.sort(key=lambda item: -item[0])
        return [
            [tuple((point[0], point[1])) for point in ordered]
            for _, ordered in scored_quads[:limit]
        ]

    def _try_decode_matrix(
        self,
        image: Image.Image,
        *,
        output_size: int | None = None,
        scale: int = 1,
        tracked: bool = False,
    ) -> DecodedVideoFrame | None:
        params = self._fast_decode_params if tracked else self._full_decode_params
        for sample_radius_factor, threshold_override, invert, offset_x, offset_y in params:
            working_image = image
            if invert:
                working_image = Image.fromarray(255 - np.array(image))

            matrix = self.decoder._sample_modules(
                working_image,
                output_size=output_size,
                scale=scale,
                sample_radius_factor=sample_radius_factor,
                threshold_override=threshold_override,
                sample_offset_x=offset_x,
                sample_offset_y=offset_y,
            )
            control_info = self.decoder._select_control_info(matrix)
            if control_info is None:
                continue

            payload = self.decoder._decode_payload(matrix, control_info)
            if payload is None:
                continue

            return DecodedVideoFrame(
                frame_num=int(control_info.frame_num),
                total_frames=int(control_info.total_frames),
                data_len=int(control_info.data_len),
                data=payload,
                source_index=-1,
            )
        return None

    def _decode_pil_with_info(
        self,
        pil_image: Image.Image,
        tracked_quad=None,
        allow_full_search: bool = True,
    ) -> tuple[DecodedVideoFrame | None, str | None, list[tuple[int, int]] | None]:
        image_array = np.array(pil_image.convert("L"))

        if tracked_quad is not None:
            tracked_warp = self.decoder._warp_from_corners(image_array, tracked_quad)
            if tracked_warp is not None:
                processed, output_size, scale = tracked_warp
                if isinstance(processed, np.ndarray):
                    processed = Image.fromarray(processed)
                tracked_result = self._try_decode_matrix(
                    processed,
                    output_size=output_size,
                    scale=scale,
                    tracked=True,
                )
                if tracked_result is not None:
                    return tracked_result, "tracked-quad", tracked_quad

        direct_result = self._try_decode_matrix(pil_image, tracked=True)
        if direct_result is not None:
            return direct_result, "direct", tracked_quad

        if not allow_full_search:
            return None, None, tracked_quad

        warp_normal, warp_flipped = self.decoder._detect_and_warp(pil_image)
        for label, warp_result in (("warp", warp_normal), ("warp-flipped", warp_flipped)):
            if warp_result is None:
                continue
            processed, output_size, scale = warp_result
            if isinstance(processed, np.ndarray):
                processed = Image.fromarray(processed)
            decoded = self._try_decode_matrix(processed, output_size=output_size, scale=scale)
            if decoded is not None:
                return decoded, label, tracked_quad

        for quad in self._candidate_quads(pil_image):
            candidate_warp = self.decoder._warp_from_corners(image_array, quad)
            if candidate_warp is None:
                continue
            processed, output_size, scale = candidate_warp
            if isinstance(processed, np.ndarray):
                processed = Image.fromarray(processed)
            decoded = self._try_decode_matrix(
                processed,
                output_size=output_size,
                scale=scale,
                tracked=True,
            )
            if decoded is not None:
                return decoded, "candidate-quad", quad

        return None, None, tracked_quad

    def _print_decode_progress(self, received_frames: dict[int, DecodedVideoFrame], total_frames: int | None) -> None:
        if total_frames is None:
            print(f"Collected {len(received_frames)} frame(s), waiting for frame metadata...")
            return

        completed = len(received_frames)
        width = min(30, max(10, total_frames))
        filled = int(round(width * completed / total_frames)) if total_frames else width
        bar = "[" + "#" * filled + "-" * (width - filled) + "]"

        missing = [str(index) for index in range(total_frames) if index not in received_frames]
        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += ", ..."
            print(f"{bar} {completed}/{total_frames} frames, missing: {preview}")
        else:
            print(f"{bar} {completed}/{total_frames} frames, all received")

    def _missing_frame_numbers(
        self,
        received_frames: dict[int, DecodedVideoFrame],
        expected_total_frames: int | None,
    ) -> list[int]:
        if expected_total_frames is None:
            return []
        return [index for index in range(expected_total_frames) if index not in received_frames]

    def _format_missing_frames(self, missing: list[int]) -> str:
        preview = ", ".join(str(index) for index in missing[:8])
        if len(missing) > 8:
            preview += ", ..."
        return preview

    def _merge_decoded_results(
        self,
        primary_frames: dict[int, DecodedVideoFrame],
        primary_total_frames: int | None,
        secondary_frames: dict[int, DecodedVideoFrame],
        secondary_total_frames: int | None,
    ) -> tuple[dict[int, DecodedVideoFrame], int | None]:
        expected_totals = {
            total_frames
            for total_frames in (primary_total_frames, secondary_total_frames)
            if total_frames is not None
        }
        if len(expected_totals) > 1:
            raise ValueError(
                "Inconsistent total frame count between decode passes: "
                + ", ".join(str(value) for value in sorted(expected_totals))
            )

        merged: dict[int, DecodedVideoFrame] = {
            frame_num: self._clone_decoded_frame(frame, frame.source_index)
            for frame_num, frame in primary_frames.items()
        }
        for frame_num, frame in secondary_frames.items():
            merged.setdefault(frame_num, self._clone_decoded_frame(frame, frame.source_index))

        expected_total_frames = next(iter(expected_totals)) if expected_totals else None
        return merged, expected_total_frames

    def encode_file_to_video(
        self,
        input_file: str,
        output_video: str,
        fps: int = 7,
        marker_frames: int = 8,
        data_frames: int = 2,
        max_seconds: float | None = None,
        save_cut: bool = False,
    ) -> tuple[int, int, float]:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_file}")

        input_bytes = input_path.read_bytes()
        
        if max_seconds is not None and max_seconds > 0:
            effective_fps = fps / data_frames
            max_bytes = int(effective_fps * self.encoder.data_per_frame * max_seconds)
            if len(input_bytes) > max_bytes:
                input_bytes = input_bytes[:max_bytes]
        
        if save_cut and (max_seconds is not None and max_seconds > 0):
            cut_path = input_path.with_name(f"{input_path.stem}_cut{input_path.suffix}")
            cut_path.write_bytes(input_bytes)
        
        data_frames_bgr = self._build_data_frames(input_bytes)

        writer = cv2.VideoWriter(
            output_video,
            self._select_video_codec(output_video),
            float(fps),
            self.frame_size,
        )
        if not writer.isOpened():
            raise ValueError(f"Failed to open video writer for {output_video}")

        try:
            if marker_frames > 0:
                start_marker = self._make_marker_frame("start")
                for _ in range(marker_frames):
                    writer.write(start_marker)
            for frame in data_frames_bgr:
                for _ in range(data_frames):
                    writer.write(frame)
            if marker_frames > 0:
                end_marker = self._make_marker_frame("end")
                for _ in range(marker_frames):
                    writer.write(end_marker)
        finally:
            writer.release()
        
        effective_fps = fps / data_frames
        kbps = (len(input_bytes) * 8) / (len(data_frames_bgr) / effective_fps) / 1000 if len(data_frames_bgr) > 0 else 0.0
        
        return len(data_frames_bgr), len(input_bytes), kbps

    def _finalize_decoded_output(
        self,
        received_frames: dict[int, DecodedVideoFrame],
        expected_total_frames: int | None,
        output_file: str,
    ) -> int:
        if not received_frames:
            raise ValueError("No decodable data frames were found in the video")
        if expected_total_frames is None:
            raise ValueError("Unable to determine total frame count from the video")

        missing = self._missing_frame_numbers(received_frames, expected_total_frames)
        if missing:
            missing_text = self._format_missing_frames(missing)
            raise ValueError(f"Video decode incomplete, missing frame(s): {missing_text}")

        ordered_payload = bytearray()
        for frame_num in range(expected_total_frames):
            frame = received_frames[frame_num]
            ordered_payload.extend(frame.data[: frame.data_len])

        Path(output_file).write_bytes(bytes(ordered_payload))
        return len(ordered_payload)

    def _decode_video_native_mode(self, input_video: str):
        capture = cv2.VideoCapture(input_video)
        if not capture.isOpened():
            raise ValueError(f"Failed to open video: {input_video}")

        started = False
        received_frames: dict[int, DecodedVideoFrame] = {}
        expected_total_frames: int | None = None
        decode_cache: list[tuple[np.ndarray, DecodedVideoFrame]] = []
        tracked_quad = None
        tracked_failures = 0
        frame_index = 0
        last_signature: np.ndarray | None = None
        last_decoded: DecodedVideoFrame | None = None

        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_index += 1

                if not started:
                    if self._is_start_marker(frame_bgr):
                        started = True
                        continue
                    else:
                        started = True

                if self._is_end_marker(frame_bgr):
                    if expected_total_frames is not None and len(received_frames) >= expected_total_frames:
                        break
                    continue

                signature = self._frame_signature(frame_bgr)
                if last_signature is not None and self._signature_distance(signature, last_signature) <= 1:
                    if last_decoded is None:
                        continue
                    decoded_frame = self._clone_decoded_frame(last_decoded, frame_index)
                    method = "repeat-skip"
                else:
                    cached = self._lookup_cached_frame(signature, decode_cache)
                    if cached is not None:
                        decoded_frame = self._clone_decoded_frame(cached, frame_index)
                        method = "cache-hit"
                    else:
                        decoded_frame, method = self._decode_fast_aligned_frame(frame_bgr)
                        if decoded_frame is None:
                            pil_image = self._to_pil_image(frame_bgr)
                            allow_full_search = tracked_quad is None or tracked_failures >= self._tracked_retry_limit
                            decoded_frame, method, new_tracked_quad = self._decode_pil_with_info(
                                pil_image,
                                tracked_quad=tracked_quad,
                                allow_full_search=allow_full_search,
                            )
                            if decoded_frame is None:
                                last_signature = signature
                                last_decoded = None
                                if tracked_quad is not None:
                                    tracked_failures += 1
                                    if tracked_failures >= self._tracked_retry_limit:
                                        tracked_quad = None
                                continue
                            if new_tracked_quad is not None:
                                tracked_quad = new_tracked_quad
                        tracked_failures = 0
                        decode_cache.append((signature, self._clone_decoded_frame(decoded_frame, frame_index)))
                        if len(decode_cache) > 24:
                            decode_cache.pop(0)

                last_signature = signature
                decoded_frame.source_index = frame_index
                last_decoded = self._clone_decoded_frame(decoded_frame, frame_index)

                if expected_total_frames is None:
                    expected_total_frames = decoded_frame.total_frames
                elif decoded_frame.total_frames != expected_total_frames:
                    raise ValueError(
                        f"Inconsistent total frame count: expected {expected_total_frames}, got {decoded_frame.total_frames}"
                    )

                if decoded_frame.frame_num in received_frames:
                    continue

                received_frames[decoded_frame.frame_num] = decoded_frame
                print(
                    f"Received frame {decoded_frame.frame_num + 1}/{decoded_frame.total_frames} "
                    f"from video frame #{frame_index} ({method})"
                )
                self._print_decode_progress(received_frames, decoded_frame.total_frames)

                if len(received_frames) == decoded_frame.total_frames:
                    break
        finally:
            capture.release()

        return received_frames, expected_total_frames

    def decode_video_to_file(self, input_video: str, output_file: str) -> int:
        native_received_frames: dict[int, DecodedVideoFrame] = {}
        native_total_frames: int | None = None

        try:
            native_received_frames, native_total_frames = self._decode_video_native_mode(input_video)
            missing = self._missing_frame_numbers(native_received_frames, native_total_frames)
            if not missing:
                return self._finalize_decoded_output(native_received_frames, native_total_frames, output_file)
            missing_text = self._format_missing_frames(missing)
            native_error = ValueError(f"Native video decode incomplete, missing frame(s): {missing_text}")
        except Exception as native_error:
            pass

        print("Native video decode check finished. Trying camera-style fallback...")
        fallback_received_frames, fallback_total_frames = decode_camera_video_mode(self, input_video)
        received_frames, expected_total_frames = self._merge_decoded_results(
            native_received_frames,
            native_total_frames,
            fallback_received_frames,
            fallback_total_frames,
        )

        return self._finalize_decoded_output(received_frames, expected_total_frames, output_file)
