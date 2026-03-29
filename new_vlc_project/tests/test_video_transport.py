import tempfile
import unittest
from pathlib import Path

from src.video_transport import OptTransVideoTransport


class TestVideoTransport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.transport = OptTransVideoTransport()
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_fast_grid_decode_on_generated_frame(self):
        payload = bytes((index * 7) % 256 for index in range(512))
        image = self.transport.encoder.build_image(payload, frame_num=0, total_frames=1)
        frame = self.transport._to_bgr_frame(image)
        decoded, method = self.transport._decode_fast_aligned_frame(frame)

        self.assertIsNotNone(decoded)
        self.assertTrue(method.startswith("fast-grid"))
        self.assertEqual(decoded.data, payload)
        self.assertEqual(decoded.data_len, len(payload))
        self.assertEqual(decoded.total_frames, 1)

    def test_video_round_trip_mp4(self):
        payload = bytes(index % 251 for index in range(3000))
        input_path = self.root / "input.bin"
        video_path = self.root / "roundtrip.mp4"
        output_path = self.root / "output.bin"
        input_path.write_bytes(payload)

        frame_count = self.transport.encode_file_to_video(
            str(input_path),
            str(video_path),
            fps=6,
            marker_frames=6,
            data_frames=2,
        )
        decoded_bytes = self.transport.decode_video_to_file(str(video_path), str(output_path))

        self.assertGreaterEqual(frame_count, 2)
        self.assertEqual(decoded_bytes, len(payload))
        self.assertEqual(output_path.read_bytes(), payload)

    def test_phone_shot_video_decode(self):
        input_video = self.repo_root / "examples" / "111.mp4"
        expected_output = self.repo_root / "examples" / "input.bin"
        if not input_video.exists() or not expected_output.exists():
            self.skipTest("phone-shot sample video is not available")

        output_path = self.root / "phone_shot.bin"
        decoded_bytes = self.transport.decode_video_to_file(str(input_video), str(output_path))

        self.assertEqual(decoded_bytes, len(expected_output.read_bytes()))
        self.assertEqual(output_path.read_bytes(), expected_output.read_bytes())


if __name__ == "__main__":
    unittest.main()
