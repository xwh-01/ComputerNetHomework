from __future__ import annotations

from pathlib import Path

from reedsolo import RSCodec

from .layout_v2 import OptTransV2Layout


class OptTransEncoderPillow:
    def __init__(self, version: int = 2):
        self.layout = OptTransV2Layout()
        if version != self.layout.version:
            raise ValueError(f"Only version {self.layout.version} is supported")

        self.version = version
        self.module_size = self.layout.module_size
        self.matrix_size = self.layout.matrix_size
        self.margin = self.layout.margin
        self.total_size = self.layout.total_size
        self.image_size = self.layout.image_size
        self.data_per_frame = self.layout.data_per_frame
        self.blocks = self.layout.blocks
        self.data_per_block = self.layout.data_per_block
        self.ecc_per_block = self.layout.ecc_per_block
        self.block_size = self.layout.block_size
        self.total_data_area = self.layout.total_encoded_bytes
        self.rs = RSCodec(self.ecc_per_block)
        self._base_matrix = self.layout.build_base_matrix()

    def _encode_data(self, data: bytes) -> bytearray:
        if len(data) > self.data_per_frame:
            raise ValueError(
                f"Data length exceeds per-frame capacity of {self.data_per_frame} bytes"
            )

        padded_data = data + b"\x00" * (self.data_per_frame - len(data))
        encoded = bytearray()
        for block_index in range(self.blocks):
            start = block_index * self.data_per_block
            end = start + self.data_per_block
            encoded.extend(self.rs.encode(padded_data[start:end]))
        return encoded

    def _write_control_area(self, matrix: list[list[int]], control_bytes: bytes) -> None:
        control_bits = []
        for byte in control_bytes:
            control_bits.extend((byte >> (7 - bit)) & 1 for bit in range(8))

        for block_index in range(2):
            for bit_index, (row, col) in enumerate(self.layout.iter_control_positions(block_index)):
                matrix[row][col] = control_bits[bit_index]

    def _place_data_bits(self, matrix: list[list[int]], data_bits: list[int]) -> None:
        if len(data_bits) != self.layout.data_capacity_bits:
            raise ValueError(
                f"Encoded payload size mismatch: {len(data_bits)} != {self.layout.data_capacity_bits}"
            )

        for bit, (row, col) in zip(data_bits, self.layout.data_positions):
            matrix[row][col] = bit

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

    def _apply_mask(self, matrix: list[list[int]], mask_pattern: int) -> list[list[int]]:
        masked = [row[:] for row in matrix]
        mask_func = self._get_mask_func(mask_pattern)
        for row, col in self.layout.data_positions:
            if mask_func(row, col):
                masked[row][col] = 1 - masked[row][col]
        return masked

    def _finder_like_penalty(self, bits: list[int]) -> int:
        penalty = 0
        target = [1, 0, 1, 1, 1, 0, 1]
        for start in range(len(bits) - len(target) + 1):
            if bits[start:start + len(target)] != target:
                continue
            left_white = start >= 4 and bits[start - 4:start] == [0, 0, 0, 0]
            right_white = (
                start + len(target) + 4 <= len(bits)
                and bits[start + len(target):start + len(target) + 4] == [0, 0, 0, 0]
            )
            if left_white or right_white:
                penalty += 40
        return penalty

    def _collect_row_segments(self, matrix: list[list[int]]) -> list[list[int]]:
        segments: list[list[int]] = []
        for row in range(self.matrix_size):
            current: list[int] = []
            for col in range(self.matrix_size):
                if self.layout.is_data_module(row, col):
                    current.append(matrix[row][col])
                elif current:
                    segments.append(current)
                    current = []
            if current:
                segments.append(current)
        return segments

    def _collect_col_segments(self, matrix: list[list[int]]) -> list[list[int]]:
        segments: list[list[int]] = []
        for col in range(self.matrix_size):
            current: list[int] = []
            for row in range(self.matrix_size):
                if self.layout.is_data_module(row, col):
                    current.append(matrix[row][col])
                elif current:
                    segments.append(current)
                    current = []
            if current:
                segments.append(current)
        return segments

    def _calculate_mask_penalty(self, matrix: list[list[int]]) -> int:
        penalty = 0

        for segment in self._collect_row_segments(matrix) + self._collect_col_segments(matrix):
            run_length = 1
            for index in range(1, len(segment)):
                if segment[index] == segment[index - 1]:
                    run_length += 1
                else:
                    if run_length >= 5:
                        penalty += run_length - 2
                    run_length = 1
            if run_length >= 5:
                penalty += run_length - 2
            penalty += self._finder_like_penalty(segment)

        for row in range(self.matrix_size - 1):
            for col in range(self.matrix_size - 1):
                if not all(
                    self.layout.is_data_module(r, c)
                    for r, c in (
                        (row, col),
                        (row, col + 1),
                        (row + 1, col),
                        (row + 1, col + 1),
                    )
                ):
                    continue
                if (
                    matrix[row][col]
                    == matrix[row][col + 1]
                    == matrix[row + 1][col]
                    == matrix[row + 1][col + 1]
                ):
                    penalty += 3

        dark_modules = sum(matrix[row][col] for row, col in self.layout.data_positions)
        ratio = dark_modules / len(self.layout.data_positions)
        penalty += int(abs(ratio - 0.5) * 100)
        return penalty

    def build_image(
        self,
        data: bytes,
        frame_num: int = 0,
        total_frames: int = 1,
        mask_patterns: tuple[int, ...] | None = None,
    ):
        matrix = [row[:] for row in self._base_matrix]

        encoded_bytes = self._encode_data(data)
        data_bits = []
        for byte in encoded_bytes:
            data_bits.extend((byte >> (7 - bit)) & 1 for bit in range(8))
        self._place_data_bits(matrix, data_bits)

        if mask_patterns is None:
            mask_patterns = tuple(range(8))
        if not mask_patterns:
            raise ValueError("mask_patterns must contain at least one candidate")

        if len(mask_patterns) == 1:
            best_mask = mask_patterns[0]
            final_control = self.layout.build_control_bytes(
                data_len=len(data),
                mask_pattern=best_mask,
                frame_num=frame_num,
                total_frames=total_frames,
            )
            self._write_control_area(matrix, final_control)
            final_matrix = self._apply_mask(matrix, best_mask)
            return self.layout.render_matrix(final_matrix)

        best_mask = mask_patterns[0]
        best_penalty = float("inf")
        for mask_pattern in mask_patterns:
            control_bytes = self.layout.build_control_bytes(
                data_len=len(data),
                mask_pattern=mask_pattern,
                frame_num=frame_num,
                total_frames=total_frames,
            )
            trial = [row[:] for row in matrix]
            self._write_control_area(trial, control_bytes)
            masked = self._apply_mask(trial, mask_pattern)
            penalty = self._calculate_mask_penalty(masked)
            if penalty < best_penalty:
                best_penalty = penalty
                best_mask = mask_pattern

        final_control = self.layout.build_control_bytes(
            data_len=len(data),
            mask_pattern=best_mask,
            frame_num=frame_num,
            total_frames=total_frames,
        )
        self._write_control_area(matrix, final_control)
        final_matrix = self._apply_mask(matrix, best_mask)
        return self.layout.render_matrix(final_matrix)

    def encode_data(self, data: bytes, output_image=None, frame_num: int = 0, total_frames: int = 1):
        image = self.build_image(data, frame_num=frame_num, total_frames=total_frames)
        if output_image is not None:
            if hasattr(output_image, "write"):
                image.save(output_image, format="PNG")
            else:
                image.save(output_image)
        return image

    def encode_file(self, input_file, output_image):
        with open(input_file, "rb") as file_obj:
            data = file_obj.read()

        total_frames = (len(data) + self.data_per_frame - 1) // self.data_per_frame
        if total_frames == 1:
            return self.encode_data(data, output_image, frame_num=0, total_frames=1)

        output_path = Path(output_image)
        for frame_num in range(total_frames):
            start = frame_num * self.data_per_frame
            end = min((frame_num + 1) * self.data_per_frame, len(data))
            frame_data = data[start:end]
            frame_output = output_path.with_name(
                f"{output_path.stem}_frame{frame_num}{output_path.suffix}"
            )
            image = self.build_image(frame_data, frame_num=frame_num, total_frames=total_frames)
            image.save(frame_output)
        return total_frames
