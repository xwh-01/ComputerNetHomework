# test_file.py
from encoder import OptTransEncoder
from decoder import OptTransDecoder

# 创建测试文件
with open("test_input.txt", "w") as f:
    f.write("This is a test file for OptTrans.\n" * 100)  # 创建较大文件测试分帧

# 创建编码器和解码器
encoder = OptTransEncoder()
decoder = OptTransDecoder()

# 编码文件
print("开始文件编码...")
encoder.encode_file("test_input.txt", "test_output.png")
print("文件编码完成")

# 解码文件
print("开始文件解码...")
decoded_size = decoder.decode_file("test_output.png", "test_decoded.txt")
print("文件解码完成")

# 验证文件内容一致性
with open("test_input.txt", "r") as f:
    original_content = f.read()

with open("test_decoded.txt", "r") as f:
    decoded_content = f.read()

if original_content == decoded_content:
    print("✓ 文件编码解码测试通过！")
else:
    print("✗ 文件编码解码测试失败！")

# 清理测试文件
import os
for file in ["test_input.txt", "test_output.png", "test_decoded.txt"]:
    if os.path.exists(file):
        os.remove(file)