# 接收端运行脚本：小组可直接修改input_video_name/参数，无需改核心逻辑
from pathlib import Path
import config
from receiver_core import VLCReceiver
from utils import logger, convert_video_format, compare_bin_files

if __name__ == "__main__":
    # ========== 仅需修改此处参数 ==========
    input_video_name = "small_test_encoded.avi"  # 输入视频名（本地/拍摄）
    output_bin_name = "small_test_decoded.bin"   # 输出二进制文件名
    output_validity_name = "small_test_validity.txt"  # 有效性标记名
    brightness_threshold_high = 200  # 自定义高阈值
    brightness_threshold_low = 50    # 自定义低阈值
    # 可选：若输入是MP4，自动转换为AVI
    convert_mp4_to_avi = False
    # =====================================

    # 初始化接收端
    receiver = VLCReceiver()
    receiver.high_threshold = brightness_threshold_high
    receiver.low_threshold = brightness_threshold_low

    # 配置路径
    input_video_path = config.CAPTURED_VIDEO_DIR / input_video_name if convert_mp4_to_avi else config.ENCODED_VIDEO_DIR / input_video_name
    output_bin_path = config.OUTPUT_BIN_DIR / output_bin_name
    output_validity_path = config.OUTPUT_BIN_DIR / output_validity_name

    # 转换视频格式（若需要）
    if convert_mp4_to_avi and input_video_path.suffix.lower() == ".mp4":
        converted_video_path = config.CAPTURED_VIDEO_DIR / f"{input_video_path.stem}_converted{config.VIDEO_EXT}"
        if not convert_video_format(input_video_path, converted_video_path):
            logger.error("视频格式转换失败，退出解码！")
            exit(1)
        input_video_path = converted_video_path

    # 执行解码
    logger.info("========== 开始解码 ==========")
    success = receiver.decode(input_video_path, output_bin_path, output_validity_path)
    if success:
        logger.info("========== 解码成功 ==========")
        # 可选：对比原始文件（测试用）
        original_bin_path = config.INPUT_BIN_DIR / "small_test.bin"
        compare_bin_files(original_bin_path, output_bin_path)
    else:
        logger.error("========== 解码失败 ==========")