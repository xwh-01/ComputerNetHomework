#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.encoder_pillow import OptTransEncoderPillow
from src.decoder_pillow import OptTransDecoderPillow
# 👇 新增：导入独立的 decode_to_file 函数
from src.decoder_pillow import decode_to_file, decode_video_to_file

def main():
    parser = argparse.ArgumentParser(description='OptTrans: Binary file to image/video encoder/decoder')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    encode_parser = subparsers.add_parser('encode', help='Encode a binary file to an image')
    encode_parser.add_argument('input_file', help='Input binary file')
    encode_parser.add_argument('output_image', help='Output image file')
    encode_parser.add_argument('--module-size', type=int, default=8, help='Size of each module in pixels (default: 8)')
    encode_parser.add_argument('--ecc', type=int, default=10, help='Number of RS error correction symbols (default: 10)')
    encode_parser.add_argument('--matrix-size', type=int, default=145, help='Matrix size (default: 145, larger = more data)')
    
    encode_video_parser = subparsers.add_parser('encode-video', help='Encode a binary file to a video')
    encode_video_parser.add_argument('input_file', help='Input binary file')
    encode_video_parser.add_argument('output_video', help='Output video file')
    encode_video_parser.add_argument('--fps', type=int, default=1, help='Frames per second (default: 1)')
    
    decode_parser = subparsers.add_parser('decode', help='Decode an image to a binary file')
    decode_parser.add_argument('input_image', help='Input image file')
    decode_parser.add_argument('output_file', help='Output binary file')
    decode_parser.add_argument('--module-size', type=int, default=8, help='Size of each module in pixels (default: 8)')
    decode_parser.add_argument('--ecc', type=int, default=10, help='Number of RS error correction symbols (default: 10)')
    decode_parser.add_argument('--matrix-size', type=int, default=145, help='Matrix size (must match encode)')
    
    decode_video_parser = subparsers.add_parser('decode-video', help='Decode a video to a binary file')
    decode_video_parser.add_argument('input_video', help='Input video file')
    decode_video_parser.add_argument('output_file', help='Output binary file')
    decode_video_parser.add_argument('--original', help='Original file path for accuracy calculation')
    
    args = parser.parse_args()
    
    if args.command == 'encode':
        encoder = OptTransEncoderPillow()
        encoder.encode_file(args.input_file, args.output_image)
        print(f"Encoded {args.input_file} to {args.output_image}")
        print(f"  Matrix size: 166x166")
        print(f"  Module size: 6px")
        print(f"  Output size: 996x996px")
    elif args.command == 'encode-video':
        encoder = OptTransEncoderPillow()
        total_frames = encoder.encode_file_to_video(args.input_file, args.output_video, fps=args.fps)
        print(f"Encoded {args.input_file} to {args.output_video}")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {args.fps}")
    elif args.command == 'decode':
        # 👇 注释/删除原有错误调用（2行）
        # decoder = OptTransDecoderPillow(module_size=args.module_size, rs_ecc_symbols=args.ecc, matrix_size=args.matrix_size)
        # success = decoder.decode_to_file(args.input_image, args.output_file)
        
        # 👇 新增：调用独立函数，赋值给 success
        success = decode_to_file(args.input_image, args.output_file) is not None
        
        if success:
            print(f"Decoded {args.input_image} to {args.output_file}")
        else:
            print(f"Failed to decode {args.input_image}")
            sys.exit(1)
    elif args.command == 'decode-video':
        success = decode_video_to_file(args.input_video, args.output_file, args.original) is not None
        
        if success:
            print(f"Decoded {args.input_video} to {args.output_file}")
        else:
            print(f"Failed to decode {args.input_video}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
