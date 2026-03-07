# 全局配置文件：所有参数集中管理，拓展时仅需修改此处
import os
from pathlib import Path

# ===================== 基础路径配置 =====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_BIN_DIR = DATA_DIR / "input_bin"
ENCODED_VIDEO_DIR = DATA_DIR / "encoded_video"
CAPTURED_VIDEO_DIR = DATA_DIR / "captured_video"
OUTPUT_BIN_DIR = DATA_DIR / "output_bin"

# 自动创建目录
for dir_path in [DATA_DIR, INPUT_BIN_DIR, ENCODED_VIDEO_DIR, CAPTURED_VIDEO_DIR, OUTPUT_BIN_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===================== 视频参数配置 =====================
# 基础视频参数（可拓展：支持自定义分辨率/帧率）
VIDEO_RESOLUTION = (640, 480)  # (宽, 高)
VIDEO_FPS = 1  # 帧率（可改为2/5fps，需同步调整解码逻辑）
VIDEO_CODEC = "XVID"  # 编码器（AVI格式）
VIDEO_EXT = ".avi"    # 输出视频格式（可拓展为.mp4）

# ===================== 编码规则配置 =====================
# 同步帧配置（可拓展：自定义同步帧数量/颜色）
SYNC_FRAME_NUM = 3  # 开头/结尾同步帧数量（如改为5帧）
SYNC_FRAME_COLOR_START = 255  # 开头同步帧（全白）
SYNC_FRAME_COLOR_END = 0      # 结尾同步帧（全黑）

# 数据帧编码规则（可拓展：彩色编码、纠错编码）
DATA_FRAME_COLOR_1 = 255  # 1对应白色
DATA_FRAME_COLOR_0 = 0    # 0对应黑色
MAX_VIDEO_DURATION = 30   # 最大视频时长（秒）

# ===================== 解码规则配置 =====================
# 亮度阈值（可拓展：自适应阈值、彩色阈值）
BRIGHTNESS_THRESHOLD_HIGH = 200  # 判定为1的亮度阈值
BRIGHTNESS_THRESHOLD_LOW = 50    # 判定为0的亮度阈值
GRAYSCALE_CONVERT = True         # 是否转灰度图（必开，可拓展为彩色解码）

# ===================== 日志配置 =====================
LOG_LEVEL = "INFO"  # DEBUG/INFO/WARNING/ERROR
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"