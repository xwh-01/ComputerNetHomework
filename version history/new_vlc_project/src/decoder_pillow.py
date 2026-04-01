from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np
from PIL import Image
from reedsolo import RSCodec

from .layout_v2 import ControlInfo, OptTransV2Layout


class OptTransDecoderPillow:
    def __init__(self):
        self.layout = OptTransV2Layout()
        self.module_size = self.layout.module_size
        self.matrix_size = self.layout.matrix_size
        self.margin = self.layout.margin
        self.total_size = self.layout.total_size
        self.data_per_frame = self.layout.data_per_frame
        self.blocks = self.layout.blocks
        self.data_per_block = self.layout.data_per_block
        self.ecc_per_block = self.layout.ecc_per_block
        self.block_size = self.layout.block_size
        self.rs = RSCodec(self.ecc_per_block)

    def _sample_modules(
        self,
        image,
        output_size=None,
        scale=1,
        sample_radius_factor=3,
        threshold_override=None,
        sample_offset_x=0,
        sample_offset_y=0,
    ):
        img_width, img_height = image.size
        if output_size is None:
            module_pixel_size = min(img_width, img_height) // self.total_size
        else:
            module_pixel_size = self.module_size * scale

        matrix = [[0] * self.matrix_size for _ in range(self.matrix_size)]
        img_np = np.array(image.convert("L"))
        sample_radius = max(1, module_pixel_size // sample_radius_factor)

        if threshold_override is not None:
            threshold = threshold_override
        elif output_size is None:
            threshold = 128
        else:
            all_brightness = []
            for row in range(self.matrix_size):
                for col in range(self.matrix_size):
                    cx = int(
                        (self.margin + col) * module_pixel_size
                        + module_pixel_size // 2
                        + sample_offset_x
                    )
                    cy = int(
                        (self.margin + row) * module_pixel_size
                        + module_pixel_size // 2
                        + sample_offset_y
                    )
                    x1 = max(0, cx - sample_radius)
                    x2 = min(img_width - 1, cx + sample_radius)
                    y1 = max(0, cy - sample_radius)
                    y2 = min(img_height - 1, cy + sample_radius)
                    if x1 <= x2 and y1 <= y2:
                        all_brightness.append(np.mean(img_np[y1:y2 + 1, x1:x2 + 1]))

            if all_brightness:
                np_brightness = np.array(all_brightness, dtype=np.uint8)
                threshold, _ = cv2.threshold(
                    np_brightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            else:
                threshold = 128

        for row in range(self.matrix_size):
            for col in range(self.matrix_size):
                cx = int(
                    (self.margin + col) * module_pixel_size
                    + module_pixel_size // 2
                    + sample_offset_x
                )
                cy = int(
                    (self.margin + row) * module_pixel_size
                    + module_pixel_size // 2
                    + sample_offset_y
                )
                x1 = max(0, cx - sample_radius)
                x2 = min(img_width - 1, cx + sample_radius)
                y1 = max(0, cy - sample_radius)
                y2 = min(img_height - 1, cy + sample_radius)
                if x1 <= x2 and y1 <= y2:
                    region = img_np[y1:y2 + 1, x1:x2 + 1]
                    matrix[row][col] = 1 if np.mean(region) < threshold else 0

        return matrix

    def _read_control_block(self, matrix, block_index: int) -> bytes:
        bits = [matrix[row][col] for row, col in self.layout.iter_control_positions(block_index)]
        bytes_list = []
        for start in range(0, len(bits), 8):
            byte = 0
            for bit in bits[start:start + 8]:
                byte = (byte << 1) | bit
            bytes_list.append(byte)
        return bytes(bytes_list)

    def _select_control_info(self, matrix) -> ControlInfo | None:
        raw_top = self._read_control_block(matrix, 0)
        raw_bottom = self._read_control_block(matrix, 1)
        top_info = self.layout.parse_control_bytes(raw_top)
        bottom_info = self.layout.parse_control_bytes(raw_bottom)

        if top_info and not bottom_info:
            return top_info
        if bottom_info and not top_info:
            return bottom_info
        if not top_info and not bottom_info:
            return None
        if top_info.raw != bottom_info.raw:
            return None
        return top_info

    def _snake_read(self, matrix):
        return [matrix[row][col] for row, col in self.layout.data_positions]

    def _get_mask_func(self, mask_pattern: int):
        mask_funcs = [
            lambda row, col: (row + col) % 2 == 0,
            lambda row, col: row % 2 == 0,
            lambda row, col: col % 3 == 0,
            lambda row, col: (row + col) % 3 == 0,
            lambda row, col: (row // 2 + col // 3) % 2 == 0,
            lambda row, col: ((row * col) % 2) + ((row * col) % 3) == 0,
            lambda row, col: (((row * col) % 2) + ((row * col) % 3)) % 2 == 0,
            lambda row, col: (((row + col) % 2) + ((row * col) % 3)) % 2 == 0,
        ]
        return mask_funcs[mask_pattern % 8]

    def _apply_mask(self, matrix, mask_pattern: int):
        masked = [row[:] for row in matrix]
        mask_func = self._get_mask_func(mask_pattern)
        for row, col in self.layout.data_positions:
            if mask_func(row, col):
                masked[row][col] = 1 - masked[row][col]
        return masked

    def _timing_quality(self, matrix) -> float:
        checks = 0
        matches = 0

        for index, col in enumerate(range(12, 116)):
            checks += 1
            if matrix[self.layout.timing_row][col] == self.layout.timing_value(index):
                matches += 1

        for index, row in enumerate(range(12, 116)):
            checks += 1
            if matrix[row][self.layout.timing_col] == self.layout.timing_value(index):
                matches += 1

        return matches / checks if checks else 0.0

    def _decode_payload(self, matrix, control_info: ControlInfo) -> bytes | None:
        if self._timing_quality(matrix) < 0.8:
            return None

        unmasked = self._apply_mask(matrix, control_info.mask_pattern)
        data_bits = self._snake_read(unmasked)
        data_bytes = []
        for start in range(0, len(data_bits), 8):
            byte = 0
            for bit in data_bits[start:start + 8]:
                byte = (byte << 1) | bit
            data_bytes.append(byte)

        decoded = bytearray()
        for block_index in range(self.blocks):
            start = block_index * self.block_size
            end = start + self.block_size
            if end > len(data_bytes):
                return None
            block_data = bytes(data_bytes[start:end])
            try:
                decoded.extend(self.rs.decode(block_data)[0])
            except Exception:
                return None

        return bytes(decoded[:control_info.data_len])

    def _find_finder_candidates(self, img):
        height, width = img.shape
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        all_candidates = []

        for invert in [False, True]:
            working_img = 255 - img_blur if invert else img_blur
            _, img_thresh = cv2.threshold(
                working_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            contours, hierarchy = cv2.findContours(
                img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if hierarchy is None:
                continue

            for index, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 100 or area > (width * height) // 8:
                    continue

                has_child = hierarchy[0][index][2] != -1
                has_parent = hierarchy[0][index][3] != -1
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                rect_area = w_rect * h_rect
                if rect_area == 0:
                    continue

                solidity = area / rect_area
                aspect_ratio = float(w_rect) / h_rect
                if not (0.5 < solidity < 1.5 and 0.5 < aspect_ratio < 2.0):
                    continue
                if not (has_child or has_parent):
                    continue

                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    all_candidates.append((cx, cy, area))

                all_candidates.append((x + w_rect // 2, y + h_rect // 2, area))

        unique_candidates = []
        min_dist = min(width, height) // 40
        for candidate in sorted(all_candidates, key=lambda item: -item[2]):
            duplicate = False
            for existing in unique_candidates:
                dist = ((candidate[0] - existing[0]) ** 2 + (candidate[1] - existing[1]) ** 2) ** 0.5
                if dist < min_dist:
                    duplicate = True
                    break
            if not duplicate:
                unique_candidates.append(candidate)

        return unique_candidates[:12]

    def _order_candidate_quad(self, points):
        tl = min(points, key=lambda point: point[0] + point[1])
        tr = min(points, key=lambda point: -point[0] + point[1])
        bl = min(points, key=lambda point: point[0] - point[1])
        br = max(points, key=lambda point: point[0] + point[1])

        unique = {(point[0], point[1]) for point in (tl, tr, bl, br)}
        if len(unique) < 4:
            return None
        return (tl, tr, bl, br)

    def _quad_geometry_score(self, quad):
        tl, tr, bl, br = quad

        def dist(point_a, point_b):
            return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5

        top_len = dist(tl, tr)
        bottom_len = dist(bl, br)
        left_len = dist(tl, bl)
        right_len = dist(tr, br)

        side_ratio1 = min(top_len, bottom_len) / max(top_len, bottom_len) if max(top_len, bottom_len) else 0
        side_ratio2 = min(left_len, right_len) / max(left_len, right_len) if max(left_len, right_len) else 0
        aspect = max(top_len, left_len) / min(top_len, left_len) if min(top_len, left_len) else 100
        score = side_ratio1 + side_ratio2

        if not (0.7 < side_ratio1 < 1.3 and 0.7 < side_ratio2 < 1.3 and 0.5 < aspect < 2.0):
            return None
        return score

    def _find_finder_corners(self, img):
        candidates = self._find_finder_candidates(img)
        if len(candidates) < 4:
            return None

        best_four = None
        best_score = -1
        for four_points in combinations(candidates, 4):
            ordered = self._order_candidate_quad(list(four_points))
            if ordered is None:
                continue
            score = self._quad_geometry_score(ordered)
            if score is not None and score > best_score:
                best_score = score
                best_four = ordered

        if best_four is not None:
            return tuple((point[0], point[1]) for point in best_four)

        points = [item[:2] for item in candidates[:8]]
        tl = min(points, key=lambda point: point[0] + point[1])
        tr = min(points, key=lambda point: -point[0] + point[1])
        bl = min(points, key=lambda point: point[0] - point[1])
        br = max(points, key=lambda point: point[0] + point[1])
        return (tl, tr, bl, br)

    def _warp_from_corners(self, img_gray, corners):
        tl_finder, tr_finder, bl_finder, br_finder = corners
        src_pts = np.float32([tl_finder, tr_finder, bl_finder, br_finder])

        scale = 3
        module_pixel = self.module_size * scale
        dst_tl = (
            (self.margin + self.layout.large_finder_size / 2) * module_pixel,
            (self.margin + self.layout.large_finder_size / 2) * module_pixel,
        )
        dst_tr = (
            (self.margin + self.matrix_size - self.layout.large_finder_size / 2) * module_pixel,
            (self.margin + self.layout.large_finder_size / 2) * module_pixel,
        )
        dst_bl = (
            (self.margin + self.layout.large_finder_size / 2) * module_pixel,
            (self.margin + self.matrix_size - self.layout.large_finder_size / 2) * module_pixel,
        )
        dst_br = (
            (self.margin + self.matrix_size - self.layout.small_finder_size / 2) * module_pixel,
            (self.margin + self.matrix_size - self.layout.small_finder_size / 2) * module_pixel,
        )

        dst_pts = np.float32([dst_tl, dst_tr, dst_bl, dst_br])
        output_size = int(self.total_size * module_pixel)
        transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            img_gray, transform, (output_size, output_size), flags=cv2.INTER_LANCZOS4
        )
        warped = cv2.GaussianBlur(warped, (3, 3), 0.3)
        return (Image.fromarray(warped), output_size, scale)

    def _detect_and_warp(self, image):
        img_gray = np.array(image.convert("L"))
        corners = self._find_finder_corners(img_gray)
        if corners is None:
            return None, None

        warped = self._warp_from_corners(img_gray, corners)
        processed_image, output_size, scale = warped
        flipped = Image.fromarray(255 - np.array(processed_image))
        return warped, (flipped, output_size, scale)

    def _try_decode_from_image(
        self,
        image,
        output_size=None,
        scale=1,
        sample_radius_factor=3,
        threshold_override=None,
        sample_offset_x=0,
        sample_offset_y=0,
    ):
        matrix = self._sample_modules(
            image,
            output_size=output_size,
            scale=scale,
            sample_radius_factor=sample_radius_factor,
            threshold_override=threshold_override,
            sample_offset_x=sample_offset_x,
            sample_offset_y=sample_offset_y,
        )
        control_info = self._select_control_info(matrix)
        if control_info is None:
            return None, 0.0
        return self._decode_payload(matrix, control_info), self._timing_quality(matrix)

    def _try_decode_with_thresholds(self, image, output_size=None, scale=1):
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

        best_result = None
        best_score = -1.0
        best_desc = None

        for sample_radius_factor, threshold_override, invert, offset_x, offset_y in params_list:
            working_img = image
            if invert:
                working_img = Image.fromarray(255 - np.array(image))

            result, timing_score = self._try_decode_from_image(
                working_img,
                output_size=output_size,
                scale=scale,
                sample_radius_factor=sample_radius_factor,
                threshold_override=threshold_override,
                sample_offset_x=offset_x,
                sample_offset_y=offset_y,
            )
            if result is None:
                continue

            score = timing_score
            desc = f"radius={sample_radius_factor}"
            if threshold_override is not None:
                desc += f" threshold={threshold_override}"
            if offset_x or offset_y:
                desc += f" offset=({offset_x},{offset_y})"
            if invert:
                desc += " inverted"

            if score > best_score:
                best_score = score
                best_result = result
                best_desc = desc

        return best_result, best_desc or "no-valid-decode"

    def _try_decode_from_candidate_quads(self, image):
        img_gray = np.array(image.convert("L"))
        candidates = self._find_finder_candidates(img_gray)[:8]
        if len(candidates) < 4:
            return None

        scored_quads = []
        for combo in combinations(candidates, 4):
            ordered = self._order_candidate_quad(list(combo))
            if ordered is None:
                continue
            score = self._quad_geometry_score(ordered)
            if score is None:
                continue
            scored_quads.append((score, ordered))

        scored_quads.sort(key=lambda item: -item[0])
        for _, quad in scored_quads:
            warped = self._warp_from_corners(img_gray, [(point[0], point[1]) for point in quad])
            processed_image, output_size, scale = warped
            result, method = self._try_decode_with_thresholds(processed_image, output_size, scale)
            if result is not None:
                return result, method

        return None

    def decode_data(self, input_image):
        image = Image.open(input_image)

        print("Trying direct sampling decode...")
        result, method = self._try_decode_with_thresholds(image)
        if result is not None:
            print(f"Direct sampling succeeded via {method}")
            return result

        print("Trying perspective correction...")
        warp_normal, warp_flipped = self._detect_and_warp(image)

        if warp_normal is not None:
            processed_image, output_size, scale = warp_normal
            result, method = self._try_decode_with_thresholds(processed_image, output_size, scale)
            if result is not None:
                print(f"Warped decode succeeded via {method}")
                return result

        if warp_flipped is not None:
            processed_image, output_size, scale = warp_flipped
            result, method = self._try_decode_with_thresholds(processed_image, output_size, scale)
            if result is not None:
                print(f"Inverted warped decode succeeded via {method}")
                return result

        print("Trying candidate quad fallback...")
        candidate_result = self._try_decode_from_candidate_quads(image)
        if candidate_result is not None:
            result, method = candidate_result
            print(f"Candidate quad decode succeeded via {method}")
            return result

        raise ValueError("Failed to decode a valid version 2 payload")

    def decode_file(self, input_image, output_file):
        data = self.decode_data(input_image)
        with open(output_file, "wb") as file_obj:
            file_obj.write(data)
        return len(data)
