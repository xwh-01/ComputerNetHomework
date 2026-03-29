from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


def crc16_modbus(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial & 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


@dataclass(frozen=True)
class ControlInfo:
    version: int
    frame_num: int
    total_frames: int
    data_len: int
    mask_pattern: int
    raw: bytes


class OptTransV2Layout:
    version = 2
    matrix_size = 128
    module_size = 8
    margin = 5
    large_finder_size = 11
    small_finder_size = 7
    control_height = 4
    control_width = 32
    timing_row = 11
    timing_col = 11
    blocks = 8
    data_per_block = 173
    ecc_per_block = 68

    def __init__(self) -> None:
        self.total_size = self.matrix_size + 2 * self.margin
        self.image_size = self.total_size * self.module_size
        self.block_size = self.data_per_block + self.ecc_per_block
        self.data_per_frame = self.blocks * self.data_per_block
        self.total_encoded_bytes = self.blocks * self.block_size

        self.large_finder_origins = (
            (0, 0),
            (0, self.matrix_size - self.large_finder_size),
            (self.matrix_size - self.large_finder_size, 0),
        )
        self.large_reserved_rects = (
            (0, 0, 12, 12),
            (0, self.matrix_size - 12, 12, 12),
            (self.matrix_size - 12, 0, 12, 12),
        )
        self.small_finder_origin = (
            self.matrix_size - self.small_finder_size,
            self.matrix_size - self.small_finder_size,
        )
        self.small_reserved_rect = (self.matrix_size - 8, self.matrix_size - 8, 8, 8)
        self.control_rects = (
            (13, 48, self.control_height, self.control_width),
            (111, 48, self.control_height, self.control_width),
        )
        self.data_positions = tuple(self._build_data_positions())
        self.data_capacity_bits = len(self.data_positions)
        expected_bits = self.total_encoded_bytes * 8
        if self.data_capacity_bits != expected_bits:
            raise ValueError(
                f"V2 layout capacity mismatch: {self.data_capacity_bits} != {expected_bits}"
            )

    def _in_rect(self, row: int, col: int, rect: tuple[int, int, int, int]) -> bool:
        top, left, height, width = rect
        return top <= row < top + height and left <= col < left + width

    def _finder_pattern(self) -> list[list[int]]:
        size = self.large_finder_size
        pattern = [[0] * size for _ in range(size)]
        border_width = 2
        center_size = 5
        center_start = (size - center_size) // 2
        center_end = center_start + center_size
        for row in range(size):
            for col in range(size):
                if (
                    row < border_width
                    or row >= size - border_width
                    or col < border_width
                    or col >= size - border_width
                ):
                    pattern[row][col] = 1
                elif center_start <= row < center_end and center_start <= col < center_end:
                    pattern[row][col] = 1
        return pattern

    def _small_finder_pattern(self) -> list[list[int]]:
        size = self.small_finder_size
        pattern = [[0] * size for _ in range(size)]
        border_width = 1
        center_size = 3
        center_start = (size - center_size) // 2
        center_end = center_start + center_size
        for row in range(size):
            for col in range(size):
                if (
                    row < border_width
                    or row >= size - border_width
                    or col < border_width
                    or col >= size - border_width
                ):
                    pattern[row][col] = 1
                elif center_start <= row < center_end and center_start <= col < center_end:
                    pattern[row][col] = 1
        return pattern

    def is_large_finder_module(self, row: int, col: int) -> bool:
        for top, left in self.large_finder_origins:
            if top <= row < top + self.large_finder_size and left <= col < left + self.large_finder_size:
                return True
        return False

    def is_large_finder_reserved(self, row: int, col: int) -> bool:
        return any(self._in_rect(row, col, rect) for rect in self.large_reserved_rects)

    def is_small_finder_module(self, row: int, col: int) -> bool:
        top, left = self.small_finder_origin
        return top <= row < top + self.small_finder_size and left <= col < left + self.small_finder_size

    def is_small_finder_reserved(self, row: int, col: int) -> bool:
        return self._in_rect(row, col, self.small_reserved_rect)

    def is_timing_module(self, row: int, col: int) -> bool:
        return (
            row == self.timing_row and 12 <= col <= 115
        ) or (
            col == self.timing_col and 12 <= row <= 115
        )

    def is_control_module(self, row: int, col: int) -> bool:
        return any(self._in_rect(row, col, rect) for rect in self.control_rects)

    def is_function_module(self, row: int, col: int) -> bool:
        return (
            self.is_large_finder_reserved(row, col)
            or self.is_small_finder_reserved(row, col)
            or self.is_timing_module(row, col)
            or self.is_control_module(row, col)
        )

    def is_data_module(self, row: int, col: int) -> bool:
        return not self.is_function_module(row, col)

    def timing_value(self, index: int) -> int:
        return 1 if index % 2 == 0 else 0

    def _build_data_positions(self) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        for row in range(self.matrix_size - 1, -1, -1):
            cols = range(self.matrix_size - 1, -1, -1) if row % 2 == 0 else range(self.matrix_size)
            for col in cols:
                if self.is_data_module(row, col):
                    positions.append((row, col))
        return positions

    def iter_control_positions(self, block_index: int) -> list[tuple[int, int]]:
        top, left, height, width = self.control_rects[block_index]
        return [
            (row, col)
            for row in range(top, top + height)
            for col in range(left, left + width)
        ]

    def build_base_matrix(self) -> list[list[int]]:
        matrix = [[0] * self.matrix_size for _ in range(self.matrix_size)]
        finder = self._finder_pattern()
        for top, left in self.large_finder_origins:
            for row in range(self.large_finder_size):
                for col in range(self.large_finder_size):
                    matrix[top + row][left + col] = finder[row][col]

        small_finder = self._small_finder_pattern()
        small_top, small_left = self.small_finder_origin
        for row in range(self.small_finder_size):
            for col in range(self.small_finder_size):
                matrix[small_top + row][small_left + col] = small_finder[row][col]

        for index, col in enumerate(range(12, 116)):
            matrix[self.timing_row][col] = self.timing_value(index)
        for index, row in enumerate(range(12, 116)):
            matrix[row][self.timing_col] = self.timing_value(index)

        return matrix

    def render_matrix(self, matrix: list[list[int]]) -> Image.Image:
        matrix_array = np.asarray(matrix, dtype=np.uint8)
        padded = np.zeros((self.total_size, self.total_size), dtype=np.uint8)
        padded[
            self.margin:self.margin + self.matrix_size,
            self.margin:self.margin + self.matrix_size,
        ] = matrix_array

        module_pixels = np.where(padded == 1, 0, 255).astype(np.uint8)
        image_array = np.repeat(
            np.repeat(module_pixels, self.module_size, axis=0),
            self.module_size,
            axis=1,
        )
        rgb = np.repeat(image_array[:, :, None], 3, axis=2)
        return Image.fromarray(rgb, mode="RGB")

    def build_control_bytes(
        self,
        data_len: int,
        mask_pattern: int,
        frame_num: int,
        total_frames: int,
    ) -> bytes:
        control = bytearray(16)
        control[0] = self.version
        control[1] = (frame_num >> 8) & 0xFF
        control[2] = frame_num & 0xFF
        control[3] = (total_frames >> 8) & 0xFF
        control[4] = total_frames & 0xFF
        control[5] = (data_len >> 16) & 0xFF
        control[6] = (data_len >> 8) & 0xFF
        control[7] = data_len & 0xFF
        control[8] = mask_pattern & 0xFF
        control[9:14] = b"\x00" * 5
        crc = crc16_modbus(bytes(control[:14]))
        control[14] = (crc >> 8) & 0xFF
        control[15] = crc & 0xFF
        return bytes(control)

    def parse_control_bytes(self, raw: bytes) -> ControlInfo | None:
        if len(raw) != 16:
            return None
        if raw[0] != self.version:
            return None
        crc = (raw[14] << 8) | raw[15]
        if crc != crc16_modbus(raw[:14]):
            return None

        frame_num = (raw[1] << 8) | raw[2]
        total_frames = (raw[3] << 8) | raw[4]
        data_len = (raw[5] << 16) | (raw[6] << 8) | raw[7]
        mask_pattern = raw[8]

        if total_frames <= 0:
            return None
        if frame_num >= total_frames:
            return None
        if not (0 < data_len <= self.data_per_frame):
            return None
        if not (0 <= mask_pattern < 8):
            return None

        return ControlInfo(
            version=raw[0],
            frame_num=frame_num,
            total_frames=total_frames,
            data_len=data_len,
            mask_pattern=mask_pattern,
            raw=bytes(raw),
        )
