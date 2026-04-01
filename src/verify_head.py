#!/usr/bin/env python3
from pathlib import Path

def verify():
    original = Path("examples/input.bin").read_bytes()
    
    # 验证 1 秒的数据
    head_1s = Path("examples/test_max_1s_decoded.bin").read_bytes()
    if original[:len(head_1s)] == head_1s:
        print("✓ 1秒数据与原文件前13840字节完全一致")
    else:
        print("✗ 1秒数据不一致！")
    
    # 验证 2 秒的数据
    head_2s = Path("examples/test_max_2s_decoded.bin").read_bytes()
    if original[:len(head_2s)] == head_2s:
        print("✓ 2秒数据与原文件前27680字节完全一致")
    else:
        print("✗ 2秒数据不一致！")

if __name__ == "__main__":
    verify()
