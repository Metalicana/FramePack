import copy
import unittest
from external_baselines.prepare_manifest import identity, remap


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(scene='scene', start_frame=7, duration_sec=180,
                          fps=30, num_frames=5397, input_image='/old/scene/7.png',
                          pose_path='/old/scene/poses.json', gt_frames_dir='/old/scene',
                          prompt='a scene', output_prefix=f'row{i}_', split_id='test',
                          split_seed=0, extra={'untouched': True}) for i in range(15)]

    def test_roundtrip_and_identity(self):
        result = remap(self.rows, '/old', '/new', check_paths=False)
        self.assertEqual(identity(self.rows, '/old'), identity(result, '/new'))
        self.assertEqual(remap(result, '/new', '/old', check_paths=False), self.rows)

    def test_start_and_order_affect_identity(self):
        changed = copy.deepcopy(self.rows)
        changed[0]['start_frame'] += 1
        self.assertNotEqual(identity(changed, '/old'), identity(self.rows, '/old'))
        self.assertNotEqual(identity(self.rows[::-1], '/old'), identity(self.rows, '/old'))

    def test_relative_filename_affects_identity(self):
        changed = copy.deepcopy(self.rows)
        changed[0]['input_image'] = '/old/scene/8.png'
        self.assertNotEqual(identity(changed, '/old'), identity(self.rows, '/old'))

    def test_rejects_invalid_inputs(self):
        for key, value in [('input_image', '/old/../outside.png'),
                           ('pose_path', '/elsewhere/poses.json'),
                           ('output_prefix', '../escape'), ('fps', 0),
                           ('num_frames', 1.5)]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.rows)
                changed[0][key] = value
                with self.assertRaises(ValueError):
                    remap(changed, '/old', '/new', check_paths=False)

    def test_cohort_and_missing_paths(self):
        with self.assertRaises(ValueError):
            remap(self.rows[:14], '/old', '/new', check_paths=False)
        with self.assertRaises(ValueError):
            remap(self.rows, '/old', '/nonexistent-external-baseline-test')


if __name__ == '__main__':
    unittest.main()
