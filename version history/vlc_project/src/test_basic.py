# test_basic.py
from encoder import OptTransEncoder
from decoder import OptTransDecoder

# 创建测试数据
test_data = b"Hello, OptTrans! This is a test message."

# 创建编码器和解码器
encoder = OptTransEncoder()
decoder = OptTransDecoder()

# 编码测试数据
print("开始编码...")
encoder.encode_data(test_data, "test_output.png", frame_num=0, total_frames=1)
print("编码完成")

# 解码测试数据
print("开始解码...")
decoded_data = decoder.decode_data("test_output.png")
print("解码完成")

# 验证数据一致性
if decoded_data == test_data:
    print("✓ 编码解码测试通过！")
else:
    print("✗ 编码解码测试失败！")
    print(f"原始数据: {test_data}")
    print(f"解码数据: {decoded_data}")

# 清理测试文件
import os
os.remove("test_output.png")