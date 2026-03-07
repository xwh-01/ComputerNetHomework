# 发送端核心：解耦编码逻辑，预留拓展接口
import cv2
import numpy as np
from pathlib import Path
import config
from utils import logger, check_file_exists

class VLCSender:
    def __init__(self):
        self.resolution = config.VIDEO_RESOLUTION
        self.fps = config.VIDEO_FPS
        self.codec = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        self.max_duration = config.MAX_VIDEO_DURATION

    def read_bin_file(self, bin_path: Path) -> list[int]:
        """读取二进制文件，转为比特流（拓展：支持加密读取）"""
        if not check_file_exists(bin_path):
            return []
        with open(bin_path, "rb") as f:
            bin_data = f.read()
        # 字节转8位比特流（补前导0）
        bit_stream = []
        for byte in bin_data:
            bit_stream += [int(bit) for bit in bin(byte)[2:].zfill(8)]
        logger.info(f"读取二进制文件完成：{len(bin_data)} 字节 → {len(bit_stream)} 位")
        return bit_stream

    def generate_frames(self, bit_stream: list[int]) -> list[np.ndarray]:
        """生成帧序列（同步帧+数据帧）（拓展：自定义编码规则，如彩色/纠错）"""
        frames = []
        height, width = self.resolution[1], self.resolution[0]

        # 1. 生成开头同步帧
        for _ in range(config.SYNC_FRAME_NUM):
            frame = np.ones((height, width, 3), dtype=np.uint8) * config.SYNC_FRAME_COLOR_START
            frames.append(frame)
        logger.info(f"生成开头同步帧：{config.SYNC_FRAME_NUM} 帧")

        # 2. 生成数据帧（可拓展：替换为彩色编码/纠错编码）
        for idx, bit in enumerate(bit_stream):
            frame_color = config.DATA_FRAME_COLOR_1 if bit == 1 else config.DATA_FRAME_COLOR_0
            frame = np.ones((height, width, 3), dtype=np.uint8) * frame_color
            frames.append(frame)
            if idx % 10 == 0:  # 每10帧打印进度
                logger.debug(f"生成数据帧进度：{idx+1}/{len(bit_stream)}")
        logger.info(f"生成数据帧：{len(bit_stream)} 帧")

        # 3. 生成结尾同步帧
        for _ in range(config.SYNC_FRAME_NUM):
            frame = np.ones((height, width, 3), dtype=np.uint8) * config.SYNC_FRAME_COLOR_END
            frames.append(frame)
        logger.info(f"生成结尾同步帧：{config.SYNC_FRAME_NUM} 帧")

        # 4. 处理时长限制
        total_frames = len(frames)
        total_duration = total_frames / self.fps
        if total_duration > self.max_duration:
            # 截断数据帧（保留同步帧）
            max_data_frames = int(self.max_duration * self.fps) - 2 * config.SYNC_FRAME_NUM
            if max_data_frames < 0:
                logger.error("最大时长过小，无法容纳同步帧！")
                return []
            # 重新生成数据帧（截断后）
            frames = frames[:config.SYNC_FRAME_NUM]  # 开头同步帧
            truncated_bit_stream = bit_stream[:max_data_frames]
            for bit in truncated_bit_stream:
                frame_color = config.DATA_FRAME_COLOR_1 if bit == 1 else config.DATA_FRAME_COLOR_0
                frame = np.ones((height, width, 3), dtype=np.uint8) * frame_color
                frames.append(frame)
            frames += [np.ones((height, width, 3), dtype=np.uint8) * config.SYNC_FRAME_COLOR_END for _ in range(config.SYNC_FRAME_NUM)]
            logger.warning(f"视频时长超限制（{total_duration:.1f}秒→{self.max_duration}秒），截断数据帧至{max_data_frames}帧")

        return frames

    def write_video(self, frames: list[np.ndarray], output_video_path: Path) -> bool:
        """写入视频文件（拓展：支持多格式/压缩）"""
        if not frames:
            logger.error("无帧数据，无法生成视频！")
            return False
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            self.codec,
            self.fps,
            self.resolution
        )
        if not video_writer.isOpened():
            logger.error(f"无法创建视频写入器：{output_video_path}")
            return False
        # 写入帧
        for idx, frame in enumerate(frames):
            video_writer.write(frame)
            if idx % 10 == 0:
                logger.debug(f"写入视频进度：{idx+1}/{len(frames)}")
        video_writer.release()
        logger.info(f"视频生成完成：{output_video_path}（时长：{len(frames)/self.fps:.1f}秒）")
        return True

    def encode(self, input_bin_path: Path, output_video_path: Path) -> bool:
        """完整编码流程（对外接口，拓展时仅需修改内部子函数）"""
        # 1. 读取二进制文件
        bit_stream = self.read_bin_file(input_bin_path)
        if not bit_stream:
            return False
        # 2. 生成帧序列
        frames = self.generate_frames(bit_stream)
        if not frames:
            return False
        # 3. 写入视频
        return self.write_video(frames, output_video_path)