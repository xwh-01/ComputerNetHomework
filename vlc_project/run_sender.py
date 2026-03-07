# 发送端运行脚本：小组可直接修改input_bin_name/参数，无需改核心逻辑
from pathlib import Path
import config
from sender_core import VLCSender
from utils import logger

if __name__ == "__main__":
    # ========== 仅需修改此处参数 ==========
    input_bin_name = "small_test.bin"  # 输入二进制文件名（放在data/input_bin/）
    output_video_name = "small_test_encoded.avi"  # 输出视频名
    max_duration = 100  # 最大时长（覆盖config中的默认值）
    # =====================================

    # 初始化发送端
    sender = VLCSender()
    sender.max_duration = max_duration  # 自定义时长

    # 配置路径
    input_bin_path = config.INPUT_BIN_DIR / input_bin_name
    output_video_path = config.ENCODED_VIDEO_DIR / output_video_name

    # 执行编码
    logger.info("========== 开始编码 ==========")
    success = sender.encode(input_bin_path, output_video_path)
    if success:
        logger.info("========== 编码成功 ==========")
    else:
        logger.error("========== 编码失败 ==========")