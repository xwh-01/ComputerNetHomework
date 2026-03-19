import cv2
import numpy as np
from PIL import Image
import reedsolo
import os
from typing import Optional, List, Tuple, Dict, Any

def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    poly = 0xA001
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
    return crc

class OptTransDecoderPillow:
    def __init__(self, blocks: int = 12, data_per_block: int = 215, rs_ecc_symbols: int = 40,
                 module_size: int = 6, matrix_size: int = 166):
        self.blocks = blocks
        self.data_per_block = data_per_block
        self.rs_ecc_symbols = rs_ecc_symbols
        self.matrix_size = 166
        self.margin = 4
        self.total_size = 174
        self.finder_size = 7

        try:
            self.rs = reedsolo.RSCodec(rs_ecc_symbols)
        except Exception as e:
            print(f"⚠️ ReedSolomon 初始化警告：{e}")
            self.rs = None

    def decode_image(self, image_source) -> Optional[bytes]:
        try:
            if isinstance(image_source, str):
                if not os.path.exists(image_source):
                    return None
                img = Image.open(image_source).convert('1')
            elif isinstance(image_source, Image.Image):
                img = image_source.convert('1')
            elif isinstance(image_source, np.ndarray):
                if len(image_source.shape) == 3:
                    image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image_source).convert('1')
            else:
                raise TypeError("不支持的输入类型")

            # 优先使用传统尺寸匹配模式，而不是透视矫正
            w, h = img.size
            if w == 1044:
                current_margin = 4
                target_w = 174
            elif w == 996:
                current_margin = 0
                target_w = 166
            else:
                img = img.resize((174, 174), resample=Image.NEAREST)
                current_margin = 4
                target_w = 174
                w, h = 174, 174

            if w != target_w:
                img = img.resize((target_w, target_w), resample=Image.NEAREST)
                w, h = target_w, target_w

            np_img = np.array(img)
            matrix = 1 - (np_img > 0).astype(int)

            if current_margin > 0:
                m = current_margin
                if matrix.shape[0] >= m+166 and matrix.shape[1] >= m+166:
                    matrix = matrix[m:m+166, m:m+166]

            # 解析控制区
            ctrl_info = self._parse_control_robust(matrix)
            best_mask = None
            data_len = None

            if ctrl_info:
                best_mask = ctrl_info['mask_pattern']
                data_len = ctrl_info['data_len']
            else:
                best_mask, data_len = self._blind_test_mask(matrix)

                if best_mask is None:
                    best_mask = self._try_all_masks_find_best(matrix)
                    if best_mask is not None:
                        data_len = 2580
                    else:
                        # 尝试所有可能的掩码，不依赖头部检测
                        for test_mask in range(8):
                            try:
                                unmasked = self._remove_mask(matrix, test_mask)
                                data_bits = self._snake_extract(unmasked)
                                data_bytes_list = self._bits_to_bytes(data_bits)
                                raw_data = bytes(data_bytes_list)
                                corrected_data = self._apply_reed_solomon(raw_data, 2580)
                                if corrected_data:
                                    best_mask = test_mask
                                    data_len = 2580
                                    break
                            except:
                                continue
                        
                        if best_mask is None:
                            return None

            unmasked_matrix = self._remove_mask(matrix, best_mask)
            data_bits = self._snake_extract(unmasked_matrix)
            data_bytes_list = self._bits_to_bytes(data_bits)
            raw_data = bytes(data_bytes_list)

            corrected_data = self._apply_reed_solomon(raw_data, data_len)

            if corrected_data:
                return corrected_data
            else:
                return raw_data[:data_len] if data_len else raw_data

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def _perspective_warp(self, binary_np: np.ndarray) -> Optional[np.ndarray]:
        h, w = binary_np.shape
        total_pixels = h * w
        min_area = total_pixels * 0.001
        max_area = total_pixels * 0.1

        contours, _ = cv2.findContours(binary_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        locator_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
                if len(approx) == 4:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        locator_candidates.append({'center': (cx, cy), 'area': area})

        if len(locator_candidates) < 3:
            return None

        locator_candidates.sort(key=lambda x: x['area'], reverse=True)
        top_3 = locator_candidates[:3]
        points = [p['center'] for p in top_3]

        def angle_between(p1, p2, p3):
            v1 = (p2[0]-p1[0], p2[1]-p1[1])
            v2 = (p3[0]-p1[0], p3[1]-p1[1])
            return v1[0]*v2[0] + v1[1]*v2[1]

        best_dot = float('inf')
        corner_idx = -1
        for i in range(3):
            p1 = points[i]
            p2 = points[(i+1)%3]
            p3 = points[(i+2)%3]
            dot = abs(angle_between(p1, p2, p3))
            if dot < best_dot:
                best_dot = dot
                corner_idx = i

        if corner_idx == -1:
            return None

        tl = points[corner_idx]
        others = [points[(corner_idx+1)%3], points[(corner_idx+2)%3]]

        v1 = (others[0][0]-tl[0], others[0][1]-tl[1])
        v2 = (others[1][0]-tl[0], others[1][1]-tl[1])

        if abs(v1[0]) > abs(v1[1]):
            tr = others[0]
            bl = others[1]
        else:
            tr = others[1]
            bl = others[0]

        br_x = bl[0] + (tr[0] - tl[0])
        br_y = bl[1] + (tr[1] - tl[1])
        br = (br_x, br_y)

        src_pts = np.float32([tl, tr, bl, br])
        target_size = 174
        dst_pts = np.float32([[0, 0], [target_size, 0], [0, target_size], [target_size, target_size]])

        try:
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_cv = cv2.warpPerspective(binary_np, matrix, (target_size, target_size), flags=cv2.INTER_NEAREST)
            warped_np = (warped_cv > 127).astype(int)
            return warped_np
        except Exception:
            return None

    def _is_control_module(self, i, j):
        CONTROL_START = 12
        BLOCK_SIZE = 32
        if i == 4 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE: return True
        if i == 5 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE: return True
        if j == 4 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE: return True
        if j == 5 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE: return True
        if i == 161 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE: return True
        if i == 162 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE: return True
        if j == 161 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE: return True
        if j == 162 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE: return True
        return False

    def _parse_control_robust(self, matrix: np.ndarray) -> Optional[Dict[str, Any]]:
        if matrix.shape[0] < 166 or matrix.shape[1] < 166:
            return None

        def read_copy_with_offset(copy_type, row_offset=0, col_offset=0):
            bits = []
            if copy_type == 'top_left':
                rows_1 = [4 + row_offset, 5 + row_offset]
                cols_1 = list(range(12 + col_offset, 44 + col_offset))
                cols_2 = [4 + col_offset, 5 + col_offset]
                rows_2 = list(range(12 + row_offset, 44 + row_offset))
            else:
                rows_1 = [161 + row_offset, 162 + row_offset]
                cols_1 = list(range(12 + col_offset, 44 + col_offset))
                cols_2 = [161 + col_offset, 162 + col_offset]
                rows_2 = list(range(12 + row_offset, 44 + row_offset))

            if any(r < 0 or r >= 166 for r in rows_1 + rows_2) or \
               any(c < 0 or c >= 166 for c in cols_1 + cols_2):
                return None

            for row in rows_1:
                for col in cols_1:
                    bits.append(matrix[row, col])
            for col in cols_2:
                for row in rows_2:
                    bits.append(matrix[row, col])
            return bits

        for copy_type in ['top_left', 'bottom_right']:
            for row_offset in range(-2, 3):
                for col_offset in range(-2, 3):
                    bits = read_copy_with_offset(copy_type, row_offset, col_offset)
                    if bits is None:
                        continue
                    bytes_data = self._bits_to_bytes(bits)
                    if len(bytes_data) >= 16:
                        crc_calc = _crc16(bytes(bytes_data[:14]))
                        crc_read = (bytes_data[14] << 8) | bytes_data[15]
                        if crc_calc == crc_read:
                            mask_pattern = bytes_data[8] & 0x07
                            data_len = (bytes_data[5] << 16) | (bytes_data[6] << 8) | bytes_data[7]
                            return {'mask_pattern': mask_pattern, 'data_len': data_len}
        return None

    def _blind_test_mask(self, matrix: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        for m in range(8):
            try:
                unmasked = self._remove_mask(matrix, m)
                ctrl_info = self._parse_control_robust(unmasked)
                if ctrl_info:
                    return m, ctrl_info['data_len']
            except:
                continue
        return None, None

    def _try_all_masks_find_best(self, matrix: np.ndarray) -> Optional[int]:
        for test_mask in range(8):
            unmasked = self._remove_mask(matrix, test_mask)
            data_bits = self._snake_extract(unmasked)
            data_bytes_list = self._bits_to_bytes(data_bits)
            raw_data = bytes(data_bytes_list)
            if len(raw_data) >= 4:
                if raw_data.startswith(b'\x89PNG') or raw_data.startswith(b'\xff\xd8') or raw_data.startswith(b'%PDF'):
                    return test_mask
        return None

    def _remove_mask(self, matrix: np.ndarray, mask_type: int) -> np.ndarray:
        h, w = matrix.shape
        masked = np.copy(matrix)
        for i in range(h):
            for j in range(w):
                if (i < 7 and j < 7) or (i < 7 and j >= w - 7) or (i >= w - 7 and j < 7):
                    continue
                mask_bit = 0
                if mask_type == 0:
                    mask_bit = 1 if (i + j) % 2 == 0 else 0
                elif mask_type == 1:
                    mask_bit = 1 if i % 2 == 0 else 0
                elif mask_type == 2:
                    mask_bit = 1 if j % 3 == 0 else 0
                elif mask_type == 3:
                    mask_bit = 1 if (i + j) % 3 == 0 else 0
                elif mask_type == 4:
                    mask_bit = 1 if (i // 2 + j // 3) % 2 == 0 else 0
                elif mask_type == 5:
                    mask_bit = 1 if ((i * j) % 2) + ((i * j) % 3) == 0 else 0
                elif mask_type == 6:
                    mask_bit = 1 if (((i * j) % 2) + ((i * j) % 3)) % 2 == 0 else 0
                elif mask_type == 7:
                    mask_bit = 1 if (((i + j) % 2) + ((i * j) % 3)) % 2 == 0 else 0
                if mask_bit:
                    masked[i, j] = 1 - masked[i, j]
        return masked

    def _snake_extract(self, matrix: np.ndarray) -> List[int]:
        bits = []
        h, w = matrix.shape
        # 总共需要提取 3060 字节 × 8 位 = 24480 位数据
        max_bits = 3060 * 8
        for row in range(h - 1, -1, -1):
            if row % 2 == 0:
                cols = range(w - 1, -1, -1)
            else:
                cols = range(w)
            for col in cols:
                if not (self._is_finder_module(row, col, w) or self._is_control_module(row, col)):
                    bits.append(matrix[row, col])
                    if len(bits) >= max_bits:
                        break
            if len(bits) >= max_bits:
                break
        return bits

    def _is_finder_module(self, i, j, w):
        if (i < 7 and j < 7): return True
        if (i < 7 and j >= w - 7): return True
        if (i >= w - 7 and j < 7): return True
        return False

    def _bits_to_bytes(self, bits: List[int]) -> List[int]:
        bytes_list = []
        for i in range(0, len(bits), 8):
            chunk = bits[i:i+8]
            if len(chunk) < 8:
                chunk += [0] * (8 - len(chunk))
            byte_val = 0
            for bit in chunk:
                byte_val = (byte_val << 1) | bit
            bytes_list.append(byte_val)
        return bytes_list

    def _apply_reed_solomon(self, raw_data: bytes, expected_len: Optional[int]) -> Optional[bytes]:
        if self.rs is None:
            return raw_data[:expected_len] if expected_len else raw_data

        block_size = self.data_per_block + self.rs_ecc_symbols
        corrected_blocks = []
        total_blocks = len(raw_data) // block_size

        if total_blocks == 0:
            return None

        for i in range(total_blocks):
            start = i * block_size
            end = start + block_size
            block = raw_data[start:end]
            try:
                decoded_result = self.rs.decode(block)
                if isinstance(decoded_result, tuple):
                    decoded_block = decoded_result[0]
                else:
                    decoded_block = decoded_result
                corrected_blocks.append(decoded_block)
            except Exception as e:
                corrected_blocks.append(block[:self.data_per_block])

        result = b''.join(corrected_blocks)
        if expected_len:
            return result[:expected_len]
        return result

def detect_file_extension(data: bytes) -> str:
    if data.startswith(b'\x89PNG\r\n\x1a\n'): return '.png'
    elif data.startswith(b'\xff\xd8\xff'): return '.jpg'
    elif data.startswith(b'%PDF'): return '.pdf'
    elif data.startswith(b'GIF8'): return '.gif'
    else: return '.bin'

def decode_to_file(image_path: str, output_path: Optional[str] = None) -> Optional[str]:
    decoder = OptTransDecoderPillow()
    data = decoder.decode_image(image_path)
    if not data:
        print("❌ 解码失败，未生成文件")
        return None
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = detect_file_extension(data)
        output_path = f"restored_{base_name}{ext}"
    with open(output_path, 'wb') as f:
        f.write(data)
    print(f"💾 文件已保存至：{output_path}")
    return output_path

def calculate_accuracy(original_path: str, decoded_path: str) -> float:
    """计算解码文件与原始文件的正确率"""
    try:
        with open(original_path, 'rb') as f1, open(decoded_path, 'rb') as f2:
            original_data = f1.read()
            decoded_data = f2.read()
        
        min_len = min(len(original_data), len(decoded_data))
        correct_bytes = sum(1 for a, b in zip(original_data[:min_len], decoded_data[:min_len]) if a == b)
        accuracy = (correct_bytes / len(original_data)) * 100 if original_data else 0.0
        return accuracy
    except Exception as e:
        print(f"⚠️ 计算正确率时出错：{e}")
        return 0.0

def decode_video_to_file(video_path: str, output_path: Optional[str] = None, original_path: Optional[str] = None) -> Optional[str]:
    decoder = OptTransDecoderPillow()
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return None
    
    frames_data = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"🔍 处理视频帧 {frame_count + 1}")
        
        # 增强视频帧的预处理
        frame = cv2.resize(frame, (1044, 1044), interpolation=cv2.INTER_NEAREST)
        
        # 尝试多次解码，提高成功率
        max_attempts = 3
        success = False
        
        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"  尝试第 {attempt + 1} 次解码...")
            
            # 尝试不同的预处理方法
            if attempt == 1:
                # 增加对比度
                frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
            elif attempt == 2:
                # 应用高斯模糊减少噪声
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
            data = decoder.decode_image(frame)
            if data:
                frames_data.append(data)
                print(f"✅ 帧 {frame_count + 1} 解码成功，数据长度：{len(data)} 字节")
                success = True
                break
        
        if not success:
            print(f"❌ 帧 {frame_count + 1} 解码失败")
        
        frame_count += 1
    
    cap.release()
    
    if not frames_data:
        print("❌ 视频中未解码到任何数据")
        return None
    
    # 合并所有帧的数据
    combined_data = b''.join(frames_data)
    print(f"📦 合并后数据总长度：{len(combined_data)} 字节")
    
    # 检查原始文件长度并截断多余数据
    if original_path and os.path.exists(original_path):
        with open(original_path, 'rb') as f:
            original_data = f.read()
        print(f"📦 原始文件长度：{len(original_data)} 字节")
        # 截断多余数据
        if len(combined_data) > len(original_data):
            combined_data = combined_data[:len(original_data)]
            print(f"📦 截断后数据长度：{len(combined_data)} 字节")
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        ext = detect_file_extension(combined_data)
        output_path = f"restored_{base_name}{ext}"
    
    with open(output_path, 'wb') as f:
        f.write(combined_data)
    
    print(f"💾 文件已保存至：{output_path}")
    print(f"📊 处理了 {frame_count} 帧，成功解码 {len(frames_data)} 帧")
    
    # 计算并输出正确率
    if original_path and os.path.exists(original_path):
        accuracy = calculate_accuracy(original_path, output_path)
        print(f"🎯 解码正确率：{accuracy:.2f}%")
        
        # 输出详细的错误信息
        with open(original_path, 'rb') as f1, open(output_path, 'rb') as f2:
            original_data = f1.read()
            decoded_data = f2.read()
        
        min_len = min(len(original_data), len(decoded_data))
        error_count = sum(1 for a, b in zip(original_data[:min_len], decoded_data[:min_len]) if a != b)
        print(f"📊 错误字节数：{error_count}，总字节数：{len(original_data)}")
    
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        decode_to_file(sys.argv[1])
    else:
        print("用法：python decoder_pillow.py <image_path>")
