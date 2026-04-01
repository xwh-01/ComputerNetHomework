#!/usr/bin/env python3
from pathlib import Path
import sys


def compare_bits(file1: str, file2: str):
    path1 = Path(file1)
    path2 = Path(file2)

    if not path1.exists():
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    if not path2.exists():
        print(f"Error: File not found: {file2}")
        sys.exit(1)

    data1 = path1.read_bytes()
    data2 = path2.read_bytes()

    min_len = min(len(data1), len(data2))
    max_len = max(len(data1), len(data2))

    val_bits = []
    correct_count = 0
    total_bits = max_len * 8

    for i in range(max_len):
        if i < min_len:
            byte1 = data1[i]
            byte2 = data2[i]
        else:
            byte1 = data1[i] if i < len(data1) else 0
            byte2 = data2[i] if i < len(data2) else 0

        for bit_pos in range(7, -1, -1):
            bit1 = (byte1 >> bit_pos) & 1
            bit2 = (byte2 >> bit_pos) & 1
            if bit1 == bit2:
                val_bits.append(b"1")
                correct_count += 1
            else:
                val_bits.append(b"0")

    val_path = path1.with_suffix(".val")
    val_path.write_bytes(b"".join(val_bits))

    accuracy = (correct_count / total_bits) * 100 if total_bits > 0 else 0

    print(f"✓ Comparison complete!")
    print(f"  File 1: {path1.name} ({len(data1)} bytes)")
    print(f"  File 2: {path2.name} ({len(data2)} bytes)")
    print(f"  Total bits: {total_bits}")
    print(f"  Correct bits: {correct_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Val file saved: {val_path.name}")

    return accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: py -3.12 compare_bits.py <file1> <file2>")
        print("Example: py -3.12 compare_bits.py examples/input_cut.bin examples/input_decoded.bin")
        sys.exit(1)

    compare_bits(sys.argv[1], sys.argv[2])
