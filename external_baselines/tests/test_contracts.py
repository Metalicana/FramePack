import copy
import json
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import convert_poses
from study import fingerprint, resume_valid, sha, validate_video, logged
from collect_metrics import quality, QUALITY
from initial_image import read_initial_image


class CameraTests(unittest.TestCase):
    def test_known_translation_rotation_and_no_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'poses.json'
            path.write_text(json.dumps({'CineCameraActor': {
                '0': {'position': [0, 0, 0], 'rotation': [0, 0, 0]},
                '1': {'position': [100, 0, 0], 'rotation': [0, 0, 90]}}}))
            calibration = dict(verified=True, evidence='synthetic test only',
                opencv_camera_to_ue_camera=[[0, 0, 1], [1, 0, 0], [0, -1, 0]],
                position_units_per_meter=100, normalized_intrinsics_after_resize=[1, 1, .5, .5])
            # This fixture contains no GT images or image directory.
            poses = convert_poses(path, 0, 2, calibration)
            np.testing.assert_allclose(poses[0, 4:].reshape(3, 4), np.eye(4)[:3])
            w2c = np.eye(4)
            w2c[:3] = poses[1, 4:].reshape(3, 4)
            c2w = np.linalg.inv(w2c)
            np.testing.assert_allclose(c2w[:3, 3], [0, 0, 1], atol=1e-6)
            np.testing.assert_allclose(c2w[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-6)
            calibration['verified'] = False
            with self.assertRaises(ValueError):
                convert_poses(path, 0, 2, calibration)


class RunTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe unavailable')
    def test_real_encoded_video_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'synthetic.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=256x256:r=30',
                            '-frames:v', '2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
            result = validate_video(video, {'fps': 30, 'start_frame': 7}, 2)
            self.assertEqual(int(result['nb_read_frames']), 2)
            with self.assertRaises(ValueError):
                validate_video(video, {'fps': 30, 'start_frame': 7}, 3)

    def test_cli_failure_exports_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'study_root': str(root/'study'), 'source_manifest': str(root/'missing.jsonl')}))
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'study.py'),
                                   'run', '--config', str(config)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertTrue((root/'study/tables/coverage.csv').is_file())
            self.assertTrue((root/'study/tables/scores.csv').is_file())
            self.assertTrue((root/'study/failure.json').is_file())

    def test_only_initial_image_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / '0007.png'
            future = Path(directory) / '0008.png'
            Image.new('RGB', (640, 352), (10, 20, 30)).save(initial)
            future.write_bytes(b'future image intentionally unreadable')
            original = Image.open
            accesses = []
            def guarded(path):
                accesses.append(Path(path))
                if Path(path) != initial:
                    raise AssertionError('Future GT accessed')
                return original(path)
            with patch('initial_image.Image.open', side_effect=guarded):
                pixels = read_initial_image(initial)
            self.assertEqual(accesses, [initial])
            self.assertEqual(pixels.shape, (256, 256, 3))
            np.testing.assert_array_equal(pixels[0, 0], [10, 20, 30])

    def test_resume_rejects_changed_output_or_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'video.mp4'
            video.write_bytes(b'original')
            identity = fingerprint({'seed': 42, 'checkpoint': 'a'})
            receipt = {'identity': identity, 'video_sha256': sha(video)}
            self.assertTrue(resume_valid(receipt, identity, video))
            self.assertFalse(resume_valid(receipt, fingerprint({'seed': 43, 'checkpoint': 'a'}), video))
            video.write_bytes(b'changed')
            self.assertFalse(resume_valid(receipt, identity, video))

    def test_alignment_rejects_shifted_timestamps(self):
        stream = json.dumps({'streams': [dict(nb_read_frames=2, width=256, height=256, r_frame_rate='30/1')]})
        frames = json.dumps({'frames': [{'best_effort_timestamp_time': '.1'}, {'best_effort_timestamp_time': '.133333'}]})
        with patch('study.capture', side_effect=[stream, '', frames]):
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                validate_video(Path('fake.mp4'), {'fps': 30, 'start_frame': 7}, 2)

    def test_frame_map_preserves_start(self):
        stream = json.dumps({'streams': [dict(nb_read_frames=2, width=256, height=256, r_frame_rate='30/1')]})
        frames = json.dumps({'frames': [{'best_effort_timestamp_time': '0'}, {'best_effort_timestamp_time': '.033333'}]})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'map.csv'
            with patch('study.capture', side_effect=[stream, '', frames]):
                validate_video(Path('fake.mp4'), {'fps': 30, 'start_frame': 7}, 2, output)
            self.assertIn('1,8,.033333', output.read_text())

    def test_subprocess_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                logged([sys.executable, '-c', 'raise SystemExit(7)'], Path(directory)/'failure.log', '0')


class QualityTests(unittest.TestCase):
    def test_exact_cohort_and_group_fvd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(scene='scene', start_frame=i, duration_sec=180, num_frames=5397,
                         output_prefix=f'{i}_') for i in range(15)]
            records = [dict(r, output=f'/dfot_re10k/{i}_custom.mp4', status='completed',
                            num_frames_expected=5397, frames_seen=5397, lpips_alex=.5) for i, r in enumerate(rows)]
            (root/'summary.json').write_text(json.dumps(dict(metric_config=dict(QUALITY, max_frames=None),
                by_duration={'180': dict(completed_or_short=15, fvd_clips=60, fvd=123., fvd_detector_path='/detector')})))
            def write(items):
                (root/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in items))
            write(records)
            self.assertEqual(quality(root, rows, 'dfot_re10k'), {'LPIPS': .5, 'FVD': 123.})
            write(records[:-1] + [records[0]])
            with self.assertRaises(ValueError):
                quality(root, rows, 'dfot_re10k')
            bad = copy.deepcopy(records)
            bad[0]['start_frame'] += 1
            write(bad)
            with self.assertRaises(ValueError):
                quality(root, rows, 'dfot_re10k')


if __name__ == '__main__':
    unittest.main()
