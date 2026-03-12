from PIL import Image
from reedsolo import RSCodec

class OptTransDecoderPillow:
    def __init__(self, version=1, module_size=8, rs_ecc_symbols=10, matrix_size=145):
        self.version = version
        self.module_size = module_size
        self.rs_ecc_symbols = rs_ecc_symbols
        self.matrix_size = matrix_size
        self.rs = RSCodec(rs_ecc_symbols)
        self.control_rs = RSCodec(10)
    
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
    
    def _read_control_area(self, matrix):
        control_bits1 = []
        for j in range(7, 23):
            control_bits1.append(matrix[0][j])
            control_bits1.append(matrix[1][j])
            control_bits1.append(matrix[2][j])
            control_bits1.append(matrix[3][j])
        
        for i in range(7, 23):
            control_bits1.append(matrix[i][0])
            control_bits1.append(matrix[i][1])
            control_bits1.append(matrix[i][2])
            control_bits1.append(matrix[i][3])
        
        control_bits2 = []
        for j in range(7, 23):
            control_bits2.append(matrix[4][j])
            control_bits2.append(matrix[5][j])
            control_bits2.append(matrix[6][j])
        
        for i in range(7, 23):
            control_bits2.append(matrix[i][4])
            control_bits2.append(matrix[i][5])
            control_bits2.append(matrix[i][6])
        
        control_bytes1 = self._bits_to_bytes(control_bits1)
        control_bytes2 = self._bits_to_bytes(control_bits2)
        
        try:
            decoded1 = self.control_rs.decode(bytes(control_bytes1))
            if isinstance(decoded1, tuple):
                decoded1 = decoded1[0]
            return decoded1
        except:
            pass
        
        try:
            decoded2 = self.control_rs.decode(bytes(control_bytes2))
            if isinstance(decoded2, tuple):
                decoded2 = decoded2[0]
            return decoded2
        except:
            pass
        
        return None
    
    def _parse_control_area(self, control_data):
        if control_data is None or len(control_data) < 21:
            return None
        
        version = control_data[0]
        data_len = (control_data[1] << 16) | (control_data[2] << 8) | control_data[3]
        mask_pattern = (control_data[14] >> 4) & 0x0F
        
        return {
            'version': version,
            'data_len': data_len,
            'mask_pattern': mask_pattern
        }
    
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
    
    def _remove_mask(self, matrix, mask_pattern=0):
        unmasked = [row[:] for row in matrix]
        mask_func = self._get_mask_func(mask_pattern)
        for i in range(len(unmasked)):
            for j in range(len(unmasked[0])):
                if self._is_data_module(i, j) or self._is_control_module(i, j):
                    if mask_func(i, j):
                        unmasked[i][j] = 1 - unmasked[i][j]
        return unmasked
    
    def _snake_extract(self, matrix):
        bits = []
        for row in range(self.matrix_size - 1, -1, -1):
            if row % 2 == 0:
                cols = range(self.matrix_size - 1, -1, -1)
            else:
                cols = range(self.matrix_size)
            for col in cols:
                if self._is_data_module(row, col):
                    bits.append(matrix[row][col])
        return bits
    
    def _bits_to_bytes(self, bits):
        bytes_list = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte = 0
                for j in range(8):
                    byte = (byte << 1) | bits[i + j]
                bytes_list.append(byte)
        return bytes_list
    
    def _decode_data(self, data):
        max_data_len = len(data) - self.rs_ecc_symbols
        for data_len in range(max_data_len, 1, -1):
            try:
                test_data = data[:data_len + self.rs_ecc_symbols]
                decoded = self.rs.decode(test_data)
                if isinstance(decoded, tuple):
                    decoded = decoded[0]
                
                if len(decoded) >= 2:
                    actual_len = decoded[0] | (decoded[1] << 8)
                    if 2 + actual_len <= len(decoded):
                        return decoded[2:2+actual_len]
            except:
                continue
        return None
    
    def _extract_matrix_direct(self, image):
        img = image.convert('L')
        
        border = 4 * self.module_size
        w, h = img.size
        
        img_cropped = img.crop((border, border, w - border, h - border))
        
        target_size = self.matrix_size * 20
        img_resized = img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        pixels = img_resized.load()
        matrix = [[0]*self.matrix_size for _ in range(self.matrix_size)]
        grid_size = target_size // self.matrix_size
        
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                x = j * grid_size + grid_size // 2
                y = i * grid_size + grid_size // 2
                pixel = pixels[x, y]
                matrix[i][j] = 0 if pixel < 128 else 1
        
        return matrix
    
    def decode_image(self, input_image):
        try:
            image = Image.open(input_image)
        except:
            return None
        
        matrix = self._extract_matrix_direct(image)
        
        mask_pattern = 0
        control_info = None
        
        for test_mask in range(8):
            test_matrix = self._remove_mask(matrix, test_mask)
            control_data = self._read_control_area(test_matrix)
            test_control_info = self._parse_control_area(control_data)
            if test_control_info is not None:
                mask_pattern = test_mask
                control_info = test_control_info
                break
        
        if control_info is not None:
            mask_pattern = control_info['mask_pattern']
        
        matrix = self._remove_mask(matrix, mask_pattern)
        
        bits = self._snake_extract(matrix)
        bytes_list = self._bits_to_bytes(bits)
        
        if len(bytes_list) < self.rs_ecc_symbols:
            return None
        
        data = bytes(bytes_list)
        decoded = self._decode_data(data)
        
        return decoded
    
    def decode_to_file(self, input_image, output_file):
        data = self.decode_image(input_image)
        if data is not None:
            with open(output_file, 'wb') as f:
                f.write(data)
            return True
        return False
