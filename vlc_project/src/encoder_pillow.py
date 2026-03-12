from PIL import Image
from reedsolo import RSCodec
import struct

class OptTransEncoderPillow:
    def __init__(self, version=1, module_size=8, rs_ecc_symbols=10, matrix_size=145):
        self.version = version
        self.module_size = module_size
        self.rs_ecc_symbols = rs_ecc_symbols
        self.matrix_size = matrix_size
        self.rs = RSCodec(rs_ecc_symbols)
        self.control_rs = RSCodec(10)
    
    def _generate_finder_pattern(self):
        pattern = [[0]*7 for _ in range(7)]
        for i in range(7):
            for j in range(7):
                if i == 0 or i == 6 or j == 0 or j == 6:
                    pattern[i][j] = 1
                elif 2 <= i <= 4 and 2 <= j <= 4:
                    pattern[i][j] = 1
        return pattern
    
    def _generate_timing_pattern(self):
        length = self.matrix_size - 14
        timing = [i % 2 for i in range(length)]
        return timing
    
    def _is_control_module(self, i, j):
        if i < 7 and j < 7:
            return False
        if i < 7 and 7 <= j < 23:
            return True
        if j < 7 and 7 <= i < 23:
            return True
        return False
    
    def _is_data_module(self, i, j):
        if i < 7 and j < 7:
            return False
        if i < 7 and j >= (self.matrix_size - 7):
            return False
        if i >= (self.matrix_size - 7) and j < 7:
            return False
        if self._is_control_module(i, j):
            return False
        return True
    
    def _encode_control_area(self, data_len, mask_pattern=0, ecc_level=0):
        version = self.version
        check_method = 0
        check_value = 0
        frame_num = 0
        total_frames = 1
        frame_rate = 0
        encode_mode = 0
        module_size_byte = self.module_size
        reserved = 0
        
        control_data = bytearray(32)
        control_data[0] = version & 0xFF
        control_data[1] = (data_len >> 16) & 0xFF
        control_data[2] = (data_len >> 8) & 0xFF
        control_data[3] = data_len & 0xFF
        control_data[4] = (check_method << 4) | (check_value >> 28) & 0x0F
        control_data[5] = (check_value >> 20) & 0xFF
        control_data[6] = (check_value >> 12) & 0xFF
        control_data[7] = (check_value >> 4) & 0xFF
        control_data[8] = ((check_value & 0x0F) << 4) | ((frame_num >> 12) & 0x0F)
        control_data[9] = (frame_num >> 4) & 0xFF
        control_data[10] = ((frame_num & 0x0F) << 4) | ((total_frames >> 12) & 0x0F)
        control_data[11] = (total_frames >> 4) & 0xFF
        control_data[12] = ((total_frames & 0x0F) << 4) | frame_rate
        control_data[13] = (encode_mode << 4) | (ecc_level & 0x0F)
        control_data[14] = (mask_pattern << 4) | ((module_size_byte >> 4) & 0x0F)
        control_data[15] = module_size_byte & 0xFF
        control_data[16] = (reserved >> 16) & 0xFF
        control_data[17] = (reserved >> 8) & 0xFF
        control_data[18] = reserved & 0xFF
        
        crc = 0xFFFF
        for byte in control_data[:19]:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        control_data[19] = crc & 0xFF
        control_data[20] = (crc >> 8) & 0xFF
        
        for i in range(21, 32):
            control_data[i] = 0
        
        encoded_control = self.control_rs.encode(bytes(control_data))
        return encoded_control
    
    def _write_control_area(self, matrix, control_bits):
        bit_idx = 0
        for j in range(7, 23):
            if bit_idx < len(control_bits):
                matrix[0][j] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[1][j] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[2][j] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[3][j] = control_bits[bit_idx]
                bit_idx += 1
        
        for i in range(7, 23):
            if bit_idx < len(control_bits):
                matrix[i][0] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[i][1] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[i][2] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[i][3] = control_bits[bit_idx]
                bit_idx += 1
        
        bit_idx = 0
        for j in range(7, 23):
            if bit_idx < len(control_bits):
                matrix[4][j] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[5][j] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[6][j] = control_bits[bit_idx]
                bit_idx += 1
        
        for i in range(7, 23):
            if bit_idx < len(control_bits):
                matrix[i][4] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[i][5] = control_bits[bit_idx]
                bit_idx += 1
            if bit_idx < len(control_bits):
                matrix[i][6] = control_bits[bit_idx]
                bit_idx += 1
    
    def _snake_fill(self, data_bits):
        matrix = [[0]*self.matrix_size for _ in range(self.matrix_size)]
        bit_idx = 0
        for row in range(self.matrix_size - 1, -1, -1):
            if row % 2 == 0:
                cols = range(self.matrix_size - 1, -1, -1)
            else:
                cols = range(self.matrix_size)
            for col in cols:
                if self._is_data_module(row, col) and bit_idx < len(data_bits):
                    matrix[row][col] = data_bits[bit_idx]
                    bit_idx += 1
        return matrix
    
    def _mask_pattern_0(self, i, j):
        return (i + j) % 2 == 0
    
    def _mask_pattern_1(self, i, j):
        return i % 2 == 0
    
    def _mask_pattern_2(self, i, j):
        return j % 3 == 0
    
    def _mask_pattern_3(self, i, j):
        return (i + j) % 3 == 0
    
    def _mask_pattern_4(self, i, j):
        return (i // 2 + j // 3) % 2 == 0
    
    def _mask_pattern_5(self, i, j):
        return ((i * j) % 2) + ((i * j) % 3) == 0
    
    def _mask_pattern_6(self, i, j):
        return (((i * j) % 2) + ((i * j) % 3)) % 2 == 0
    
    def _mask_pattern_7(self, i, j):
        return (((i + j) % 2) + ((i * j) % 3)) % 2 == 0
    
    def _get_mask_func(self, mask_pattern):
        mask_funcs = [
            self._mask_pattern_0,
            self._mask_pattern_1,
            self._mask_pattern_2,
            self._mask_pattern_3,
            self._mask_pattern_4,
            self._mask_pattern_5,
            self._mask_pattern_6,
            self._mask_pattern_7
        ]
        return mask_funcs[mask_pattern % 8]
    
    def _apply_mask(self, matrix, mask_pattern):
        masked = [row[:] for row in matrix]
        mask_func = self._get_mask_func(mask_pattern)
        for i in range(len(masked)):
            for j in range(len(masked[0])):
                if self._is_data_module(i, j) or self._is_control_module(i, j):
                    if mask_func(i, j):
                        masked[i][j] = 1 - masked[i][j]
        return masked
    
    def _calculate_mask_penalty(self, matrix):
        penalty = 0
        
        for i in range(self.matrix_size):
            run_length = 1
            for j in range(1, self.matrix_size):
                if matrix[i][j] == matrix[i][j-1]:
                    run_length += 1
                else:
                    if run_length >= 5:
                        penalty += run_length - 2
                    run_length = 1
            if run_length >= 5:
                penalty += run_length - 2
        
        for j in range(self.matrix_size):
            run_length = 1
            for i in range(1, self.matrix_size):
                if matrix[i][j] == matrix[i-1][j]:
                    run_length += 1
                else:
                    if run_length >= 5:
                        penalty += run_length - 2
                    run_length = 1
            if run_length >= 5:
                penalty += run_length - 2
        
        for i in range(self.matrix_size - 1):
            for j in range(self.matrix_size - 1):
                if (matrix[i][j] == matrix[i][j+1] == 
                    matrix[i+1][j] == matrix[i+1][j+1]):
                    penalty += 3
        
        dark_modules = 0
        total_modules = 0
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if self._is_data_module(i, j) or self._is_control_module(i, j):
                    total_modules += 1
                    if matrix[i][j] == 0:
                        dark_modules += 1
        
        if total_modules > 0:
            ratio = dark_modules / total_modules
            deviation = abs(ratio - 0.5)
            penalty += int(deviation * 100)
        
        return penalty
    
    def _encode_data(self, data):
        data_len = len(data)
        len_bytes = bytes([data_len & 0xFF, (data_len >> 8) & 0xFF])
        full_data = len_bytes + data
        encoded = self.rs.encode(full_data)
        return encoded
    
    def encode_data(self, data, output_image):
        matrix = [[0]*self.matrix_size for _ in range(self.matrix_size)]
        
        finder = self._generate_finder_pattern()
        for i in range(7):
            for j in range(7):
                matrix[i][j] = finder[i][j]
                matrix[i][self.matrix_size-7+j] = finder[i][j]
                matrix[self.matrix_size-7+i][j] = finder[i][j]
        
        timing = self._generate_timing_pattern()
        for i in range(len(timing)):
            matrix[7][7+i] = timing[i]
            matrix[7+i][7] = timing[i]
        
        data_bytes = self._encode_data(data)
        
        data_bits = []
        for byte in data_bytes:
            bits = [(byte >> (7 - i)) & 1 for i in range(8)]
            data_bits.extend(bits)
        
        data_matrix = self._snake_fill(data_bits)
        
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if self._is_data_module(i, j):
                    matrix[i][j] = data_matrix[i][j]
        
        best_mask = 0
        min_penalty = float('inf')
        
        for mask_pattern in range(8):
            test_matrix = [row[:] for row in matrix]
            
            control_bytes_test = self._encode_control_area(len(data), mask_pattern)
            control_bits_test = []
            for byte in control_bytes_test:
                bits = [(byte >> (7 - i)) & 1 for i in range(8)]
                control_bits_test.extend(bits)
            
            test_matrix_control = [row[:] for row in test_matrix]
            self._write_control_area(test_matrix_control, control_bits_test)
            
            masked_test = self._apply_mask(test_matrix_control, mask_pattern)
            penalty = self._calculate_mask_penalty(masked_test)
            
            if penalty < min_penalty:
                min_penalty = penalty
                best_mask = mask_pattern
        
        control_bytes = self._encode_control_area(len(data), mask_pattern=best_mask)
        control_bits = []
        for byte in control_bytes:
            bits = [(byte >> (7 - i)) & 1 for i in range(8)]
            control_bits.extend(bits)
        self._write_control_area(matrix, control_bits)
        
        matrix = self._apply_mask(matrix, best_mask)
        
        padded_size = self.matrix_size + 2 * 4
        padded = [[1]*padded_size for _ in range(padded_size)]
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                padded[4+i][4+j] = matrix[i][j]
        
        image_size = padded_size * self.module_size
        img = Image.new('RGB', (image_size, image_size), color='white')
        pixels = img.load()
        
        for i in range(padded_size):
            for j in range(padded_size):
                color = (0, 0, 0) if padded[i][j] == 0 else (255, 255, 255)
                for y in range(i * self.module_size, (i + 1) * self.module_size):
                    for x in range(j * self.module_size, (j + 1) * self.module_size):
                        pixels[x, y] = color
        
        img.save(output_image)
        return img
    
    def encode_file(self, input_file, output_image):
        with open(input_file, 'rb') as f:
            data = f.read()
        return self.encode_data(data, output_image)
