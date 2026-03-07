# 接收端核心：解耦解码逻辑，预留拓展接口
import cv2
import numpy as np
from pathlib import Path
import config
from utils import logger, check_file_exists

class VLCReceiver:
    def __init__(self):
        self.resolution = config.VIDEO_RESOLUTION
        self.fps = config.VIDEO_FPS
        self.high_threshold = config.BRIGHTNESS_THRESHOLD_HIGH
        self.low_threshold = config.BRIGHTNESS_THRESHOLD_LOW
        self.grayscale = config.GRAYSCALE_CONVERT

    def extract_frames(self, video_path: Path) -> list[np.ndarray]:
        """提取视频帧（拓展：支持实时帧提取、格式兼容）"""
        if not check_file_exists(video_path):
            return []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"无法打开视频：{video_path}")
            return []
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"开始提取视频帧：共{total_frames}帧")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 转灰度图（拓展：彩色解码时注释此行）
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            if len(frames) % 10 == 0:
                logger.debug(f"提取帧进度：{len(frames)}/{total_frames}")
        cap.release()
        logger.info(f"提取帧完成：共{len(frames)}帧")
        return frames

    def calculate_brightness(self, frames: list[np.ndarray]) -> list[float]:
        """计算每帧平均亮度（拓展：区域亮度、彩色通道亮度）"""
        if not frames:
            logger.error("无帧数据，无法计算亮度！")
            return []
        brightness_list = []
        for frame in frames:
            # 灰度图：直接算平均；彩色图：算RGB均值
            if self.grayscale:
                brightness = np.mean(frame)
            else:
                brightness = np.mean(frame, axis=(0,1)).mean()  # RGB均值
            brightness_list.append(brightness)
        logger.info(f"亮度计算完成：共{len(brightness_list)}帧")
        return brightness_list

    def detect_sync_frames(self, brightness_list: list[float]) -> tuple[int, int]:
        """检测同步帧，返回数据帧的起止索引（拓展：自定义同步帧检测规则）"""
        sync_num = config.SYNC_FRAME_NUM
        start_idx = -1
        end_idx = -1

        # 检测开头同步帧（连续sync_num帧亮度>高阈值）
        for i in range(len(brightness_list) - sync_num + 1):
            window = brightness_list[i:i+sync_num]
            if all(b > self.high_threshold for b in window):
                start_idx = i + sync_num
                logger.info(f"检测到开头同步帧：起始索引{i}，数据帧起始索引{start_idx}")
                break
        if start_idx == -1:
            logger.error("未检测到开头同步帧！")
            return -1, -1

        # 检测结尾同步帧（连续sync_num帧亮度<低阈值）
        for i in range(start_idx, len(brightness_list) - sync_num + 1):
            window = brightness_list[i:i+sync_num]
            if all(b < self.low_threshold for b in window):
                end_idx = i
                logger.info(f"检测到结尾同步帧：起始索引{i}，数据帧结束索引{end_idx}")
                break
        if end_idx == -1:
            logger.warning("未检测到结尾同步帧，使用剩余所有帧作为数据帧")
            end_idx = len(brightness_list)

        return start_idx, end_idx

    def decode_bit_stream(self, brightness_list: list[float], start_idx: int, end_idx: int) -> tuple[list[int], list[bool]]:
        """解码比特流+标记有效性（拓展：纠错解码、加密解码）"""
        if start_idx == -1 or end_idx == -1:
            return [], []
        data_brightness = brightness_list[start_idx:end_idx]
        bit_stream = []
        bit_validity = []
        logger.info(f"开始解码数据帧：共{len(data_brightness)}帧")
        for idx, b in enumerate(data_brightness):
            if b > self.high_threshold:
                bit_stream.append(1)
                bit_validity.append(True)
            elif b < self.low_threshold:
                bit_stream.append(0)
                bit_validity.append(True)
            else:
                bit_stream.append(0)  # 无效位默认填0
                bit_validity.append(False)
            if idx % 10 == 0:
                logger.debug(f"解码进度：{idx+1}/{len(data_brightness)}")
        valid_count = sum(bit_validity)
        logger.info(f"解码完成：共{len(bit_stream)}位，有效位{valid_count}个（有效率：{valid_count/len(bit_stream)*100:.1f}%）")
        return bit_stream, bit_validity

    def write_bin_file(self, bit_stream: list[int], output_bin_path: Path) -> bool:
        """比特流转二进制文件（拓展：支持批量写入）"""
        if not bit_stream:
            logger.error("无比特流数据，无法生成二进制文件！")
            return False
        # 8位分组，不足补0
        bin_data = []
        for i in range(0, len(bit_stream), 8):
            byte_bits = bit_stream[i:i+8]
            if len(byte_bits) < 8:
                byte_bits += [0] * (8 - len(byte_bits))
            byte = int(''.join(map(str, byte_bits)), 2)
            bin_data.append(byte)
        # 写入文件
        with open(output_bin_path, "wb") as f:
            f.write(bytes(bin_data))
        logger.info(f"二进制文件生成完成：{output_bin_path}（{len(bin_data)}字节）")
        return True

    def write_validity(self, bit_stream: list[int], bit_validity: list[bool], output_txt_path: Path) -> bool:
        """写入位有效性标记（拓展：支持可视化报表）"""
        if len(bit_stream) != len(bit_validity):
            logger.error("比特流与有效性标记长度不匹配！")
            return False
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("位索引\t比特值\t有效性\n")
            for idx, (bit, valid) in enumerate(zip(bit_stream, bit_validity)):
                f.write(f"{idx}\t{bit}\t{valid}\n")
        logger.info(f"有效性标记生成完成：{output_txt_path}")
        return True

    def decode(self, input_video_path: Path, output_bin_path: Path, output_validity_path: Path) -> bool:
        """完整解码流程（对外接口，拓展时仅需修改内部子函数）"""
        # 1. 提取视频帧
        frames = self.extract_frames(input_video_path)
        if not frames:
            return False
        # 2. 计算亮度
        brightness_list = self.calculate_brightness(frames)
        if not brightness_list:
            return False
        # 3. 检测同步帧
        start_idx, end_idx = self.detect_sync_frames(brightness_list)
        if start_idx == -1:
            return False
        # 4. 解码比特流
        bit_stream, bit_validity = self.decode_bit_stream(brightness_list, start_idx, end_idx)
        if not bit_stream:
            return False
        # 5. 写入二进制文件
        if not self.write_bin_file(bit_stream, output_bin_path):
            return False
        # 6. 写入有效性标记
        if not self.write_validity(bit_stream, bit_validity, output_validity_path):
            return False
        return True