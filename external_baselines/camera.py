"""Explicit CaM (UE) to DFoT camera conversion; no image access."""
import json
import numpy as np


def convert_poses(path, start, count, calibration):
    if not calibration.get('verified') or not calibration.get('evidence'):
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
