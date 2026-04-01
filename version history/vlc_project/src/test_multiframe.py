# test_multiframe.py
from encoder import OptTransEncoder
from decoder import OptTransDecoder

# 创建较大的测试数据（超出单帧容量）
large_data = b"A" * 2000  # 假设这超过了单帧容量

encoder = OptTransEncoder()
decoder = OptTransDecoder()

print("开始多帧编码...")
frames = encoder.encode_file("large_test_input.bin", "large_output.png")
print(f"生成了 {len(frames) if isinstance(frames, list) else 1} 个帧")

# 如果生成了多个帧文件，需要合并解码
import os
if os.path.exists("large_output_frame0.png"):  # 多帧情况
    print("检测到多帧，进行多帧解码测试")
    decoded_size = decoder.decode_file("large_output_frame0.png", "large_decoded.bin")
else:  # 单帧情况
    decoded_size = decoder.decode_file("large_output.png", "large_decoded.bin")

# 验证数据一致性
with open("large_test_input.bin", "wb") as f:
    f.write(large_data)

with open("large_decoded.bin", "rb") as f:
    decoded_data = f.read()

if large_data == decoded_data:
    print("✓ 多帧编码解码测试通过！")
else:
    print("✗ 多帧编码解码测试失败！")
    print(f"原始数据长度: {len(large_data)}")
    print(f"解码数据长度: {len(decoded_data)}")

# 清理测试文件
for file in ["large_test_input.bin", "large_output.png", "large_decoded.bin"] + \
             [f"large_output_frame{i}.png" for i in range(10)]:  # 清理可能的多帧文件
    if os.path.exists(file):
        os.remove(file)