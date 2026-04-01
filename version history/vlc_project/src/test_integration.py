# test_integration.py
import os

def run_all_tests():
    """运行所有测试"""
    print("="*50)
    print("开始OptTrans系统集成测试")
    print("="*50)
    
    # 1. 测试导入
    print("\n1. 测试模块导入...")
    try:
        from encoder import OptTransEncoder
        from decoder import OptTransDecoder
        print("   ✓ 模块导入成功")
    except Exception as e:
        print(f"   ✗ 模块导入失败: {e}")
        return False
    
    # 2. 测试基本功能
    print("\n2. 测试基本编码解码功能...")
    try:
        encoder = OptTransEncoder()
        decoder = OptTransDecoder()
        
        test_data = b"Integration Test Data"
        encoder.encode_data(test_data, "integration_test.png")
        decoded = decoder.decode_data("integration_test.png")
        
        if decoded == test_data:
            print("   ✓ 基本功能测试通过")
        else:
            print("   ✗ 基本功能测试失败")
            return False
    except Exception as e:
        print(f"   ✗ 基本功能测试失败: {e}")
        return False
    
    # 3. 测试不同数据类型
    print("\n3. 测试不同类型数据...")
    test_cases = [
        (b"", "空数据"),
        (b"Short", "短数据"),
        (b"Medium length data for testing" * 10, "中等长度数据"),
        (bytes(range(256)), "全字节范围数据")
    ]
    
    for data, desc in test_cases:
        try:
            encoder.encode_data(data, "temp_test.png")
            decoded = decoder.decode_data("temp_test.png")
            if decoded == data:
                print(f"   ✓ {desc}测试通过")
            else:
                print(f"   ✗ {desc}测试失败")
                return False
        except Exception as e:
            print(f"   ✗ {desc}测试出错: {e}")
            return False
    
    # 4. 清理
    for file in ["integration_test.png", "temp_test.png"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("\n" + "="*50)
    print("✓ 所有集成测试通过！")
    print("✓ 修改后的代码工作正常")
    print("✓ 没有循环导入问题")
    print("="*50)
    return True

if __name__ == "__main__":
    run_all_tests()