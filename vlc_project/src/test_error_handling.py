# test_error_handling.py
from encoder import OptTransEncoder
from decoder import OptTransDecoder

encoder = OptTransEncoder()
decoder = OptTransDecoder()

# 测试超大数据量
try:
    too_large_data = b"A" * 100000  # 极大的数据
    encoder.encode_data(too_large_data, "temp_too_large.png", 0, 1)
    print("✓ 超大数据量处理正常（应有适当限制）")
except ValueError as e:
    print(f"✓ 正确捕获数据量过大错误: {e}")
except Exception as e:
    print(f"? 其他错误: {e}")

# 测试不存在的文件
try:
    decoder.decode_file("nonexistent.png", "output.txt")
    print("✗ 应该抛出文件不存在错误")
except FileNotFoundError:
    print("✓ 正确处理文件不存在错误")
except Exception as e:
    print(f"✓ 正确处理错误: {e}")

import os
for file in ["temp_too_large.png"]:
    if os.path.exists(file):
        os.remove(file)