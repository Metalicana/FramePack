"""Explicit CaM (UE) to DFoT camera conversion; no image access."""
import json
import math
import numpy as np


def nominal_calibration(width, height):
    """Paper-based smoke intrinsics for our full-image 256-square resize.

    Coordinates are measured from image edges; pixel centers are i+0.5,
    matching DFoT rays and Pillow's resize sampling convention.
    """
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    focal = width / (2 * math.tan(math.radians(52.67) / 2))
    native = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
    transform = np.diag([256/width, 256/height, 1.])
    resized = transform @ native
    return dict(verified=False, status='nominal_paper_smoke_only',
        evidence='https://arxiv.org/html/2506.03141v2#A2',
        published_settings={'focal_length_mm': 24, 'fov_degrees': 52.67, 'aperture_f_number': 10},
        assumptions=['52.67 degrees is horizontal FOV', 'native PNG preserves full renderer FOV',
                     'square native pixels', 'centered principal point', 'zero lens distortion',
                     'UE forward/right/up axes and MemCam rotation convention', 'positions are centimeters'],
        native_image_size=[width, height], output_image_size=[256, 256],
        pixel_coordinates='image edges; pixel centers i+0.5',
        preprocessing='Pillow bicubic full-image resize; no crop or padding',
        native_K=native.tolist(), pixel_transform=transform.tolist(), resized_K=resized.tolist(),
        position_units_per_meter=100,
        opencv_camera_to_ue_camera=[[0, 0, 1], [1, 0, 0], [0, -1, 0]],
        normalized_intrinsics_after_resize=[resized[0, 0]/256, resized[1, 1]/256, .5, .5])


def convert_poses(path, start, count, calibration, *, allow_nominal=False):
    nominal_allowed = (allow_nominal and calibration.get('status') == 'nominal_paper_smoke_only'
                       and bool(calibration.get('assumptions')))
    if not calibration.get('evidence') or not (calibration.get('verified') or nominal_allowed):
        raise ValueError('Camera calibration requires verified=true and evidence')
    basis = np.asarray(calibration['opencv_camera_to_ue_camera'], dtype=float)
    if basis.shape != (3, 3) or not np.allclose(basis.T @ basis, np.eye(3)):
        raise ValueError('Camera basis must be an orthogonal 3x3 matrix')
    scale = float(calibration['position_units_per_meter'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid camera position scale')
    intrinsics = np.asarray(calibration['normalized_intrinsics_after_resize'], dtype=float)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or (intrinsics[:2] <= 0).any():
        raise ValueError('Expected normalized fx,fy,cx,cy after full-image resize')
    camera = json.loads(path.read_text())['CineCameraActor']
    keys = sorted(camera, key=int)[start:start + count]
    if len(keys) != count or [int(k) for k in keys] != list(range(start, start + count)):
        raise ValueError('Camera keys must match explicit consecutive dataset indices')
    c2ws = []
    for key in keys:
        p = camera[key]
        pitch, roll, yaw = np.deg2rad(p['rotation'])
        cp, sp, cr, sr, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll), np.cos(yaw), np.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        c2w = np.eye(4)
        # Change both world and camera basis, preserving handedness of rotations.
        c2w[:3, :3] = basis.T @ (rz @ ry @ rx) @ basis
        c2w[:3, 3] = basis.T @ np.asarray(p['position']) / scale
        c2ws.append(c2w)
    c2ws = np.stack(c2ws)
    if not np.isfinite(c2ws).all():
        raise ValueError('Nonfinite pose')
    # DFoT performs its own reference-camera normalization in _process_conditions.
    w2cs = np.linalg.inv(c2ws)
    return np.concatenate([np.tile(intrinsics, (count, 1)), w2cs[:, :3].reshape(count, 12)], axis=1).astype('float32')
