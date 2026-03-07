# 发送端运行脚本：强制适配完整时长
from pathlib import Path
import config
from sender_core import VLCSender
from utils import logger

if __name__ == "__main__":
    # ========== 强制配置：确保时长足够 ==========
    input_bin_name = "small_test.bin"
    output_video_name = "small_test_encoded.avi"
    # 强制设置足够大的时长（100秒，远超62秒需求）
    max_duration = 100  

    # 初始化发送端+强制配置时长
    sender = VLCSender()
    sender.max_duration = max_duration  # 覆盖所有截断逻辑

    # 配置路径（确保路径无中文空格问题，这里用绝对路径兜底）
    input_bin_path = Path(config.INPUT_BIN_DIR) / input_bin_name
    output_video_path = Path(config.ENCODED_VIDEO_DIR) / output_video_name

    # 执行编码
    logger.info("========== 开始编码 ==========")
    # 手动打印关键信息：确认比特流长度
    bit_stream = sender.read_bin_file(input_bin_path)
    if bit_stream:
        logger.info(f"待传输比特流长度：{len(bit_stream)} 位（需{len(bit_stream)}帧）")
        logger.info(f"总需帧数：3+{len(bit_stream)}+3={3+len(bit_stream)+3} 帧 → 时长{3+len(bit_stream)+3}秒")
        # 生成帧+写入视频
        frames = sender.generate_frames(bit_stream)
        if frames:
            logger.info(f"实际生成帧数：{len(frames)} 帧")
            sender.write_video(frames, output_video_path)
            logger.info("========== 编码成功 ==========")
        else:
            logger.error("========== 生成帧失败 ==========")
    else:
        logger.error("========== 读取二进制文件失败 ==========")
