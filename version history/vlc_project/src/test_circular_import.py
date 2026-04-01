# test_circular_import.py
def test_imports():
    """测试模块导入是否会产生循环导入"""
    try:
        from encoder import OptTransEncoder
        print("✓ encoder导入成功")
        
        from decoder import OptTransDecoder
        print("✓ decoder导入成功")
        
        from data_codec import DataCodec
        print("✓ data_codec导入成功")
        
        # 测试各模块间的交互
        encoder = OptTransEncoder()
        decoder = OptTransDecoder()
        
        # 简单测试功能
        test_data = b"test"
        encoder.encode_data(test_data, "temp_test.png", 0, 1)
        decoded = decoder.decode_data("temp_test.png")
        
        import os
        if os.path.exists("temp_test.png"):
            os.remove("temp_test.png")
            
        if decoded == test_data:
            print("✓ 模块间交互测试通过")
        else:
            print("✗ 模块间交互测试失败")
        
        print("✓ 循环导入测试通过")
        return True
        
    except ImportError as e:
        print(f"✗ 循环导入测试失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        return False

if __name__ == "__main__":
    test_imports()