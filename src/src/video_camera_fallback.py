from __future__ import annotations

import statistics

import cv2
import numpy as np

PRIMITIVE_GROUP_DISTANCE = 12
MERGED_GROUP_DISTANCE = 30
CAMERA_THRESHOLDS = (185, 170, 165, 160, 155, 150, 145, 142, 140, 139, 138, 136, 134, 132, 130, 128, 124)
CAMERA_SCALES = (0.995, 1.0, 1.005, 0.99, 1.01)
CAMERA_INNER_MARGINS = (2, 1, 0, 3, 4)


def _order_quad_points(points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmax(diffs)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
        ],
        dtype=np.float32,
    )


def _find_content_quad(frame_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    min_area = height * width * 0.05
    best_quad = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(-1, 2).astype(np.float32)
        _, _, box_w, box_h = cv2.boundingRect(approx)
        aspect = box_w / box_h if box_h else 0.0
        if not 0.7 <= aspect <= 1.3:
            continue

        quad_center = pts.mean(axis=0)
        dist = np.linalg.norm(quad_center - center) / max(height, width)
        score = area / (height * width) - dist * 0.4 - abs(1.0 - aspect) * 0.2
        if score > best_score:
            best_score = score
            best_quad = pts

    if best_quad is None:
        return None
    return _order_quad_points(best_quad)


def _adjust_quad(
    quad: np.ndarray,
    *,
    scale: float = 1.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> np.ndarray:
    center = quad.mean(axis=0)
    adjusted = center + (quad - center) * scale
    adjusted[:, 0] += shift_x
    adjusted[:, 1] += shift_y
    return adjusted.astype(np.float32)


def _warp_content_quad(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    frame_size: tuple[int, int],
    *,
    scale: float = 1.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted_quad = _adjust_quad(quad, scale=scale, shift_x=shift_x, shift_y=shift_y)
    dst = np.array(
        [
            [0, 0],
            [frame_size[0] - 1, 0],
            [0, frame_size[1] - 1],
            [frame_size[0] - 1, frame_size[1] - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(adjusted_quad, dst)
    warped = cv2.warpPerspective(
        frame_bgr,
        transform,
        frame_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, adjusted_quad


def _average_frames(frames: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([frame.astype(np.float32) for frame in frames], axis=0)
    return np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)


def _majority_signature(signatures: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([signature.astype(np.uint8) for signature in signatures], axis=0)
    return stack.mean(axis=0) >= 0.5


def _classify_group(mean_bgr: np.ndarray) -> str:
    blue, green, red = mean_bgr
    if green > red + 25 and green > blue + 25:
        return "start"
    if red > green + 50 and red > blue + 80:
        return "end"
    return "data"


def _make_group(frame_indices: list[int], signatures: list[np.ndarray], means: list[np.ndarray]) -> dict:
    mean_bgr = np.mean(np.stack(means, axis=0), axis=0)
    return {
        "frame_indices": list(frame_indices),
        "signature": _majority_signature(signatures),
        "mean": mean_bgr,
        "kind": _classify_group(mean_bgr),
        "count": len(frame_indices),
        "start": frame_indices[0],
        "end": frame_indices[-1],
    }


def _build_primitive_groups(transport, input_video: str) -> list[dict]:
    capture = cv2.VideoCapture(input_video)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {input_video}")

    content_quad = None
    content_quad_refresh = 0
    last_signature = None
    current_indices: list[int] = []
    current_signatures: list[np.ndarray] = []
    current_means: list[np.ndarray] = []
    groups: list[dict] = []
    frame_index = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index += 1

            if content_quad is None or content_quad_refresh <= 0:
                detected_quad = _find_content_quad(frame_bgr)
                if detected_quad is not None:
                    content_quad = detected_quad
                    content_quad_refresh = 12
                else:
                    content_quad_refresh = 0
                    continue
            else:
                content_quad_refresh -= 1

            warped, _ = _warp_content_quad(frame_bgr, content_quad, transport.frame_size)
            signature = transport._frame_signature(warped)
            mean_bgr = transport._marker_mean_bgr(warped)

            if not current_indices:
                current_indices = [frame_index]
                current_signatures = [signature]
                current_means = [mean_bgr]
            else:
                distance = transport._signature_distance(signature, last_signature)
                if distance <= PRIMITIVE_GROUP_DISTANCE:
                    current_indices.append(frame_index)
                    current_signatures.append(signature)
                    current_means.append(mean_bgr)
                else:
                    groups.append(_make_group(current_indices, current_signatures, current_means))
                    current_indices = [frame_index]
                    current_signatures = [signature]
                    current_means = [mean_bgr]

            last_signature = signature
    finally:
        capture.release()

    if current_indices:
        groups.append(_make_group(current_indices, current_signatures, current_means))
    return groups


def _merge_group(left: dict, right: dict) -> dict:
    frame_indices = left["frame_indices"] + right["frame_indices"]
    mean_bgr = (left["mean"] * left["count"] + right["mean"] * right["count"]) / (left["count"] + right["count"])
    return {
        "frame_indices": frame_indices,
        "signature": _majority_signature([left["signature"], right["signature"]]),
        "mean": mean_bgr,
        "kind": "data",
        "count": len(frame_indices),
        "start": frame_indices[0],
        "end": frame_indices[-1],
    }


def _slice_data_groups(groups: list[dict]) -> list[dict]:
    data_start = 0
    for index, group in enumerate(groups):
        if group["kind"] == "start":
            data_start = index + 1
            break

    data_end = len(groups)
    for index in range(len(groups) - 1, -1, -1):
        if groups[index]["kind"] == "end" and index >= data_start:
            data_end = index
            break

    sliced = groups[data_start:data_end]
    if sliced:
        return sliced
    return [group for group in groups if group["kind"] == "data"]


def _merge_data_groups(transport, groups: list[dict]) -> list[dict]:
    if not groups:
        return []

    merged: list[dict] = []
    for group in groups:
        data_group = {
            "frame_indices": list(group["frame_indices"]),
            "signature": group["signature"],
            "mean": group["mean"],
            "kind": "data",
            "count": group["count"],
            "start": group["start"],
            "end": group["end"],
        }
        if not merged:
            merged.append(data_group)
            continue

        distance = transport._signature_distance(merged[-1]["signature"], data_group["signature"])
        if distance <= MERGED_GROUP_DISTANCE:
            merged[-1] = _merge_group(merged[-1], data_group)
        else:
            merged.append(data_group)

    stable = False
    while not stable and len(merged) > 1:
        stable = True
        counts = [group["count"] for group in merged if group["count"] >= 4]
        target = round(statistics.median(counts)) if counts else 1
        small = max(4, target // 2)
        rebuilt: list[dict] = []
        index = 0
        while index < len(merged):
            current = merged[index]
            if index + 1 < len(merged):
                following = merged[index + 1]
                combined = current["count"] + following["count"]
                if (
                    (current["count"] <= small or following["count"] <= small)
                    and abs(combined - target) <= 2
                ):
                    rebuilt.append(_merge_group(current, following))
                    stable = False
                    index += 2
                    continue
            rebuilt.append(current)
            index += 1
        merged = rebuilt

    return merged


def _build_segment_candidates(frames: list[np.ndarray]) -> list[tuple[str, list[np.ndarray]]]:
    candidates: list[tuple[str, list[np.ndarray]]] = [("all", frames)]
    frame_count = len(frames)

    if frame_count >= 5:
        candidates.append(("trim", frames[1:-1]))
        center = frame_count // 2
        candidates.append(("center5", frames[max(0, center - 2):min(frame_count, center + 3)]))
    if frame_count >= 3:
        center = frame_count // 2
        candidates.append(("center3", frames[max(0, center - 1):min(frame_count, center + 2)]))

    single_order: list[int] = []
    center = frame_count // 2
    for delta in range(frame_count):
        left = center - delta
        right = center + delta
        if 0 <= left < frame_count:
            single_order.append(left)
        if delta and 0 <= right < frame_count:
            single_order.append(right)

    seen: set[int] = set()
    for position in single_order:
        if position in seen:
            continue
        seen.add(position)
        candidates.append((f"single{position}", [frames[position]]))

    return candidates


def _decode_camera_segment(transport, frames: list[np.ndarray]):
    for label, candidate_frames in _build_segment_candidates(frames):
        candidate_frame = candidate_frames[0] if len(candidate_frames) == 1 else _average_frames(candidate_frames)
        content_quad = _find_content_quad(candidate_frame)
        if content_quad is None:
            continue

        for scale in CAMERA_SCALES:
            warped, _ = _warp_content_quad(candidate_frame, content_quad, transport.frame_size, scale=scale)
            decoded_frame, detail = transport._decode_fast_aligned_frame(
                warped,
                thresholds=CAMERA_THRESHOLDS,
                inner_margins=CAMERA_INNER_MARGINS,
            )
            if decoded_frame is not None:
                return decoded_frame, f"camera-segment {label} scale={scale} {detail}"

        warped, _ = _warp_content_quad(candidate_frame, content_quad, transport.frame_size, scale=0.995)
        decoded_frame, method, _ = transport._decode_pil_with_info(
            transport._to_pil_image(warped),
            tracked_quad=None,
            allow_full_search=False,
        )
        if decoded_frame is not None:
            return decoded_frame, f"camera-segment {label} {method}"

    return None, None


def decode_camera_video_mode(transport, input_video: str):
    primitive_groups = _build_primitive_groups(transport, input_video)
    data_groups = _slice_data_groups(primitive_groups)
    segments = _merge_data_groups(transport, data_groups)
    if not segments:
        raise ValueError("No content segments were detected in the video")

    capture = cv2.VideoCapture(input_video)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {input_video}")

    received_frames = {}
    expected_total_frames = None
    segment_index = 0
    frame_index = 0
    segment_frames: list[np.ndarray] = []

    try:
        while segment_index < len(segments):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index += 1
            segment = segments[segment_index]

            if frame_index < segment["start"]:
                continue
            if frame_index <= segment["end"]:
                segment_frames.append(frame_bgr)
            if frame_index < segment["end"]:
                continue

            decoded_frame, method = _decode_camera_segment(transport, segment_frames)
            source_index = segment["end"]
            segment_frames = []
            segment_index += 1

            if decoded_frame is None:
                continue

            decoded_frame.source_index = source_index
            if expected_total_frames is None:
                expected_total_frames = decoded_frame.total_frames
            elif decoded_frame.total_frames != expected_total_frames:
                raise ValueError(
                    f"Inconsistent total frame count: expected {expected_total_frames}, got {decoded_frame.total_frames}"
                )

            if decoded_frame.frame_num in received_frames:
                continue

            received_frames[decoded_frame.frame_num] = transport._clone_decoded_frame(decoded_frame, source_index)
            print(
                f"Received frame {decoded_frame.frame_num + 1}/{decoded_frame.total_frames} "
                f"from video frame #{source_index} ({method})"
            )
            transport._print_decode_progress(received_frames, decoded_frame.total_frames)

            if expected_total_frames is not None and len(received_frames) >= expected_total_frames:
                break
    finally:
        capture.release()

    return received_frames, expected_total_frames
