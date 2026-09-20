"""Run with MemCam Python; validate whole-cohort 180s quality and VBench."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys


QUALITY = dict(frame_stride=30, learned_image_size=224, fvd_clip_length=16,
               fvd_clips_per_video=4, fvd_frame_stride=4, fvd_image_size=224,
               fvd_backend='styleganv_i3d', fvd_eps=1e-6)


def quality(directory, rows, run):
    summary = json.loads((directory / 'summary.json').read_text())
    config = summary['metric_config']
    for key, value in QUALITY.items():
        if config.get(key) != value:
            raise ValueError(f'Incompatible metric setting: {key}')
    if 'max_frames' not in config or config['max_frames'] not in (None, 0):
        raise ValueError('Truncated/unknown evaluation length')
    records = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    expected = {r['output_prefix'] + 'custom.mp4': r for r in rows}
    if len(records) != 15 or {Path(r['output']).name for r in records} != set(expected):
        raise ValueError('Quality metrics contain a different or duplicate cohort')
    for record in records:
        row = expected[Path(record['output']).name]
        if record['status'] != 'completed' or Path(record['output']).parent.name != run:
            raise ValueError('Failed/short/mixed-run quality metrics')
        for key in ('scene', 'start_frame', 'duration_sec'):
            if record[key] != row[key]:
                raise ValueError(f'Quality identity mismatch: {key}')
        if record.get('num_frames_expected') != row['num_frames'] or record.get('frames_seen') != row['num_frames']:
            raise ValueError('Quality metrics did not cover the full video')
        if not isinstance(record.get('lpips_alex'), (float, int)) or not math.isfinite(record['lpips_alex']):
            raise ValueError('Missing LPIPS')
    group = summary['by_duration']['180']
    if group['completed_or_short'] != 15 or group['fvd_clips'] != 60 or not math.isfinite(group['fvd']):
        raise ValueError('FVD must be one complete 60-clip cohort')
    if not group.get('fvd_detector_path'):
        raise ValueError('Missing FVD detector provenance')
    return {'LPIPS': mean(r['lpips_alex'] for r in records), 'FVD': group['fvd']}


def main():
    p = argparse.ArgumentParser()
    for name in ('memcam-repo', 'manifest', 'quality-dir', 'vbench-dir', 'run', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.memcam_repo) / 'utils'))
    from run_budget_metric_grid import validate_bench
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    scores = quality(Path(args.quality_dir), rows, args.run)
    scores.update({v['metric']: v['value'] for v in validate_bench(
        args.vbench_dir, [r['output_prefix'] + 'custom.mp4' for r in rows], args.run)})
    if any(not 0 <= value <= 1 for key, value in scores.items() if key not in ('FVD', 'LPIPS')):
        raise ValueError('VBench aggregate outside [0,1]')
    Path(args.output).write_text(json.dumps(scores, indent=2) + '\n')


if __name__ == '__main__':
    main()
