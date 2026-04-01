#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.video_transport import OptTransVideoTransport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode OptTrans payloads into videos and decode them back")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode a file into a video")
    encode_parser.add_argument("input_file", help="Binary file to encode")
    encode_parser.add_argument("output_video", help="Output video path, e.g. output.mp4 or output.avi")
    encode_parser.add_argument("--fps", type=int, default=7, help="Video FPS for the phone-friendly profile, default: 7")
    encode_parser.add_argument(
        "--marker-frames",
        type=int,
        default=8,
        help="Number of repeated START/END marker frames, default: 8",
    )
    encode_parser.add_argument(
        "--data-frames",
        type=int,
        default=2,
        help="Number of repeated video frames for each data frame, default: 2",
    )

    decode_parser = subparsers.add_parser("decode", help="Decode a video back into a file")
    decode_parser.add_argument("input_video", help="Video file to decode")
    decode_parser.add_argument("output_file", help="Output file path")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    transport = OptTransVideoTransport()

    if args.command == "encode":
        input_path = Path(args.input_file)
        if not input_path.exists():
            parser.error(f"Input file does not exist: {args.input_file}")
        frame_count = transport.encode_file_to_video(
            args.input_file,
            args.output_video,
            fps=args.fps,
            marker_frames=args.marker_frames,
            data_frames=args.data_frames,
        )
        print(f"Video encode succeeded: {frame_count} data frames written to {args.output_video}")
        return 0

    if args.command == "decode":
        input_path = Path(args.input_video)
        if not input_path.exists():
            parser.error(f"Input video does not exist: {args.input_video}")
        byte_count = transport.decode_video_to_file(args.input_video, args.output_file)
        print(f"Video decode succeeded: wrote {byte_count} bytes to {args.output_file}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
