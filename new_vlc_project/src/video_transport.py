from __future__ import annotations

import io
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .decoder_pillow import OptTransDecoderPillow
from .encoder_pillow import OptTransEncoderPillow


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

    def _to_bgr_frame(self, image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

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
            thickness=border // 2,
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

    def _is_start_marker(self, frame_bgr: np.ndarray) -> bool:
        mean_bgr = frame_bgr.mean(axis=(0, 1))
        blue, green, red = mean_bgr
        return green > red + 35 and green > blue + 35

    def _is_end_marker(self, frame_bgr: np.ndarray) -> bool:
        mean_bgr = frame_bgr.mean(axis=(0, 1))
        blue, green, red = mean_bgr
        return red > green + 35 and red > blue + 35

    def _build_data_images(self, data: bytes) -> list[Image.Image]:
        total_frames = (len(data) + self.encoder.data_per_frame - 1) // self.encoder.data_per_frame
        images = []
        for frame_num in range(total_frames):
            start = frame_num * self.encoder.data_per_frame
            end = min((frame_num + 1) * self.encoder.data_per_frame, len(data))
            frame_data = data[start:end]
            buffer = io.BytesIO()
            image = self.encoder.encode_data(
                frame_data,
                buffer,
                frame_num=frame_num,
                total_frames=total_frames,
            )
            images.append(image)
        return images

    def _select_four_candidate_points(self, pil_image: Image.Image):
        img_gray = np.array(pil_image.convert("L"))
        candidates = self.decoder._find_finder_candidates(img_gray)[:8]
        if len(candidates) < 4:
            return None

        scored_quads = []
        for combo in combinations(candidates, 4):
            ordered = self.decoder._order_candidate_quad(list(combo))
            if ordered is None:
                continue
            score = self.decoder._quad_geometry_score(ordered)
            if score is None:
                continue
            scored_quads.append((score, ordered))

        if not scored_quads:
            return None

        scored_quads.sort(key=lambda item: -item[0])
        return [tuple((point[0], point[1])) for point in scored_quads[0][1]]

    def _decode_pil_image(self, pil_image: Image.Image):
        result, method = self.decoder._try_decode_with_thresholds(pil_image)
        if result is not None:
            return result, method

        warp_normal, warp_flipped = self.decoder._detect_and_warp(pil_image)
        if warp_normal is not None:
            processed, output_size, scale = warp_normal
            result, method = self.decoder._try_decode_with_thresholds(processed, output_size, scale)
            if result is not None:
                return result, method

        if warp_flipped is not None:
            processed, output_size, scale = warp_flipped
            result, method = self.decoder._try_decode_with_thresholds(processed, output_size, scale)
            if result is not None:
                return result, method

        candidate_result = self.decoder._try_decode_from_candidate_quads(pil_image)
        if candidate_result is not None:
            return candidate_result
        return None, None

    def _decode_pil_with_info(self, pil_image: Image.Image):
        params_list = [
            (3, None, False, 0, 0),
            (3, None, False, 0, 2),
            (2, None, False, 0, 2),
            (2, None, False, -2, 2),
            (2, None, False, 2, 2),
            (3, None, True, 0, 0),
            (2, None, False, 0, 0),
            (4, None, False, 0, 0),
            (4, None, False, 0, 2),
            (3, 128, False, 0, 0),
            (3, 128, False, 0, 2),
            (3, 128, True, 0, 0),
            (2, None, True, 0, 0),
            (4, None, True, 0, 0),
        ]

        def try_with_matrix(image, output_size=None, scale=1):
            best = None
            for sample_radius_factor, threshold_override, invert, offset_x, offset_y in params_list:
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
                desc = f"radius={sample_radius_factor}"
                if threshold_override is not None:
                    desc += f" threshold={threshold_override}"
                if offset_x or offset_y:
                    desc += f" offset=({offset_x},{offset_y})"
                if invert:
                    desc += " inverted"
                score = self.decoder._timing_quality(matrix)
                if best is None or score > best[3]:
                    best = (payload, control_info, desc, score)
            return best

        direct = try_with_matrix(pil_image)
        if direct is not None:
            return direct[:3]

        warp_normal, warp_flipped = self.decoder._detect_and_warp(pil_image)
        for item in (warp_normal, warp_flipped):
            if item is None:
                continue
            processed, output_size, scale = item
            decoded = try_with_matrix(processed, output_size, scale)
            if decoded is not None:
                return decoded[:3]

        img_gray = np.array(pil_image.convert("L"))
        candidates = self.decoder._find_finder_candidates(img_gray)[:8]
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

        for _, quad in scored_quads:
            warped = self.decoder._warp_from_corners(
                img_gray, [(point[0], point[1]) for point in quad]
            )
            processed, output_size, scale = warped
            decoded = try_with_matrix(processed, output_size, scale)
            if decoded is not None:
                return decoded[:3]

        return None, None, None

    def _video_fourcc(self, output_path: Path) -> int:
        if output_path.suffix.lower() == ".avi":
            return cv2.VideoWriter_fourcc(*"MJPG")
        return cv2.VideoWriter_fourcc(*"mp4v")

    def _print_decode_progress(self, decoded_frames, expected_total_frames):
        if expected_total_frames is None:
            print(f"Collected {len(decoded_frames)} frame(s), waiting for frame metadata...")
            return

        received = len(decoded_frames)
        width = min(30, max(10, expected_total_frames))
        filled = int(width * received / expected_total_frames)
        bar = "#" * filled + "-" * (width - filled)
        missing = [str(index) for index in range(expected_total_frames) if index not in decoded_frames]
        missing_preview = ", ".join(missing[:8])
        if len(missing) > 8:
            missing_preview += ", ..."
        if missing_preview:
            print(f"[{bar}] {received}/{expected_total_frames} frames, missing: {missing_preview}")
        else:
            print(f"[{bar}] {received}/{expected_total_frames} frames, all received")

    def encode_file_to_video(
        self,
        input_file: str,
        output_video: str,
        fps: int = 6,
        marker_frames: int = 12,
        data_frames: int = 3,
    ) -> int:
        input_bytes = Path(input_file).read_bytes()
        data_images = self._build_data_images(input_bytes)
        start_marker = self._make_marker_frame("start")
        end_marker = self._make_marker_frame("end")

        output_path = Path(output_video)
        writer = cv2.VideoWriter(
            str(output_path),
            self._video_fourcc(output_path),
            fps,
            self.frame_size,
        )
        if not writer.isOpened():
            raise ValueError(f"Failed to open video writer for {output_video}")

        try:
            for _ in range(marker_frames):
                writer.write(start_marker)
            for image in data_images:
                frame = self._to_bgr_frame(image)
                for _ in range(data_frames):
                    writer.write(frame)
            for _ in range(marker_frames):
                writer.write(end_marker)
        finally:
            writer.release()

        return len(data_images)

    def decode_video_to_file(self, input_video: str, output_file: str) -> int:
        capture = cv2.VideoCapture(input_video)
        if not capture.isOpened():
            raise ValueError(f"Failed to open video: {input_video}")

        seen_start = False
        seen_end = False
        decoded_frames: dict[int, DecodedVideoFrame] = {}
        expected_total_frames: int | None = None
        frame_index = -1

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1

                if self._is_start_marker(frame):
                    seen_start = True
                    continue

                if not seen_start:
                    continue

                if self._is_end_marker(frame):
                    seen_end = True
                    if expected_total_frames is not None and len(decoded_frames) >= expected_total_frames:
                        break
                    continue

                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                payload, control_info, _ = self._decode_pil_with_info(pil_image)
                if payload is None or control_info is None:
                    continue

                expected_total_frames = control_info.total_frames
                is_new_frame = control_info.frame_num not in decoded_frames
                if is_new_frame:
                    decoded_frames[control_info.frame_num] = DecodedVideoFrame(
                        frame_num=control_info.frame_num,
                        total_frames=control_info.total_frames,
                        data_len=control_info.data_len,
                        data=payload,
                        source_index=frame_index,
                    )
                    print(
                        f"Received frame {control_info.frame_num + 1}/{control_info.total_frames} "
                        f"from video frame #{frame_index}"
                    )
                    self._print_decode_progress(decoded_frames, expected_total_frames)

                if expected_total_frames is not None and len(decoded_frames) >= expected_total_frames and seen_end:
                    break
        finally:
            capture.release()

        if not seen_start:
            raise ValueError("Did not detect a START marker in the video")
        if expected_total_frames is None:
            raise ValueError("Did not decode any valid data frames from the video")
        if len(decoded_frames) != expected_total_frames:
            missing = [str(index) for index in range(expected_total_frames) if index not in decoded_frames]
            raise ValueError(f"Missing decoded frames: {', '.join(missing)}")

        ordered_data = b"".join(decoded_frames[index].data for index in range(expected_total_frames))
        Path(output_file).write_bytes(ordered_data)
        return len(ordered_data)
