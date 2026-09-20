"""Local/cluster entry point. Never uses SSH, submits jobs, or trains models."""
import argparse
import csv
import fcntl
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from prepare_manifest import identity, remap
from collect_metrics import QUALITY, quality

HERE = Path(__file__).resolve().parent
RUN = 'dfot_re10k'
DIMENSIONS = ['subject_consistency', 'background_consistency', 'motion_smoothness',
              'dynamic_degree', 'aesthetic_quality', 'imaging_quality']


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def capture(command, cwd=None):
    return subprocess.check_output(list(map(str, command)), cwd=cwd, text=True, stderr=subprocess.STDOUT)


def logged(command, path, gpu, cwd=None):
    print('Running:', ' '.join(map(str, command)), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    with Path(path).open('a') as log:
        proc = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        proc.stdout.close()
        if proc.wait():
            raise RuntimeError(f'Command failed ({proc.returncode}); see {path}')


def code_identity(repo):
    repo = Path(repo)
    sources = {str(p.relative_to(repo)): sha(p) for p in sorted(repo.rglob('*'))
               if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')
               and not any(v in p.parts for v in ('.git', '.venv', '__pycache__'))}
    return {'commit': capture(['git', 'rev-parse', 'HEAD'], repo).strip(),
            'source_sha256': fingerprint(sources)}


def audit(cfg, root):
    report = {'hostname': platform.node(), 'platform': platform.platform(),
              'cpu': platform.processor(), 'free_bytes': shutil.disk_usage(root).free,
              'framepack': 'excluded: inspected stock original/F1 workers have no pose input',
              'paths': {k: {'path': v, 'exists': Path(v).exists()} for k, v in cfg.items()
                        if isinstance(v, str) and k.endswith(('_repo', '_python', '_manifest', '_checkpoint', '_root'))}}
    for name, command in [('gpus', ['nvidia-smi']), ('conda', ['conda', 'env', 'list'])]:
        try:
            report[name] = capture(command)
        except (OSError, subprocess.CalledProcessError) as exc:
            report[name] = str(exc)
    report['repositories'] = {}
    for key in ('dfot_repo', 'memcam_repo', 'vbench_repo'):
        try:
            report['repositories'][key] = code_identity(cfg[key])
        except (OSError, subprocess.CalledProcessError) as exc:
            report['repositories'][key] = {'error': str(exc)}
    report['checkpoint_candidates'] = []
    for directory in cfg.get('checkpoint_search_roots', []):
        for p in Path(directory).rglob('*'):
            if (p.is_file() and p.suffix in ('.ckpt', '.safetensors')
                    and '.no_exist' not in p.parts and p.stat().st_size > 0):
                report['checkpoint_candidates'].append({'path': str(p), 'bytes': p.stat().st_size})
    try:
        rows = [json.loads(line) for line in Path(cfg['source_manifest']).read_text().splitlines() if line.strip()]
        report['manifest'] = {'rows': len(rows), 'sha256': sha(cfg['source_manifest']),
                              'first_row': rows[0] if rows else None}
        mapped = remap(rows, cfg['source_dataset_root'], cfg['dataset_root'], check_paths=False)
        if mapped and Path(mapped[0]['pose_path']).is_file():
            data = load(mapped[0]['pose_path'])
            camera = data['CineCameraActor']
            report['first_pose_metadata'] = camera[sorted(camera, key=int)[0]]
    except (OSError, ValueError, KeyError) as exc:
        report['manifest_error'] = str(exc)
    save(root / 'audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


def prepare(cfg, root):
    source = Path(cfg['source_manifest'])
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    mapped = remap(rows, cfg['source_dataset_root'], cfg['dataset_root'])
    if any(r['duration_sec'] != 180 or r['fps'] != 30 or r['num_frames'] != 5397 for r in rows):
        raise ValueError('This launcher requires the canonical 15 x 5397-frame, 30 FPS, nominal 180s cohort')
    frozen = root / 'inputs' / 'manifest.original.jsonl'
    frozen.parent.mkdir(exist_ok=True)
    if frozen.exists() and frozen.read_bytes() != source.read_bytes():
        raise ValueError('Original manifest changed; use a new study directory')
    frozen.write_bytes(source.read_bytes())
    dest = root / 'inputs' / 'manifest.remapped.jsonl'
    contents = ''.join(json.dumps(r) + '\n' for r in mapped)
    if dest.exists() and dest.read_text() != contents:
        raise ValueError('Remapping changed; use a new study directory')
    dest.write_text(contents)
    save(root / 'inputs' / 'manifest.identity.json', {
        'source_sha256': sha(source), 'remapped_sha256': sha(dest),
        'cohort_sha256': identity(rows, cfg['source_dataset_root'])})
    return mapped, dest


def preflight(cfg, rows, root, gpu):
    from camera import convert_poses
    if type(cfg['smoke_frames']) is not int or not 32 <= cfg['smoke_frames'] < 5397:
        raise ValueError('smoke_frames must be an integer in [32,5396]')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0,2**32)')
    for key in ('dfot_python', 'metric_python', 'vbench_python'):
        p = Path(cfg[key])
        if not p.is_absolute() or not p.is_file() or not os.access(p, os.X_OK):
            raise ValueError(f'{key} must be an existing absolute Python executable')
    for binary in ('ffmpeg', 'ffprobe', 'nvidia-smi'):
        if not shutil.which(binary):
            raise ValueError(f'Missing executable: {binary}')
    if ',' in str(gpu) or not str(gpu).strip():
        raise ValueError('Assign exactly one GPU; generation and metrics run sequentially')
    if shutil.disk_usage(root).free < cfg['minimum_free_gib'] * 1024**3:
        raise ValueError('Insufficient disk space for configured minimum_free_gib')
    if not Path(cfg['dfot_checkpoint']).is_file():
        raise ValueError('Missing official DFoT_RE10K.ckpt; no checkpoint will be downloaded')
    if not cfg.get('checkpoint_provenance') or 'REPLACE' in cfg['checkpoint_provenance']:
        raise ValueError('Record checkpoint source/revision/training domain')
    calibration = load(cfg['calibration_file'])
    for i, row in enumerate(rows):
        poses = convert_poses(Path(row['pose_path']), row['start_frame'], row['num_frames'], calibration[row['scene']])
        save(root / 'inputs' / f'camera_preview_{i:02d}.json',
             {'first': poses[0].tolist(), 'second': poses[1].tolist(), 'last': poses[-1].tolist(),
              'shape': list(poses.shape), 'format': 'normalized fx fy cx cy + flattened w2c'})
        gt = Path(row['gt_frames_dir'])
        if gt.resolve() != (Path(cfg['dataset_root']) / 'frames' / row['scene']).resolve():
            raise ValueError('Evaluator dataset_root would override GT directory with a different path')
        for k in range(row['start_frame'], row['start_frame'] + row['num_frames']):
            if not (gt / f'{k:04d}.png').is_file():
                raise ValueError(f'Missing GT index {k} in {gt}')
    reference = load(cfg['quality_reference_summary'])
    if '180' not in reference.get('by_duration', {}):
        raise ValueError('Quality reference must contain the nominal 180-second result group')
    for key, value in QUALITY.items():
        if reference['metric_config'].get(key) != value:
            raise ValueError(f'Source quality configuration differs: {key}')
    if not Path(cfg['fvd_detector']).is_file():
        raise ValueError('Missing local StyleGAN-V I3D detector')
    expected_detector = cfg.get('fvd_detector_sha256')
    if not expected_detector or sha(cfg['fvd_detector']) != expected_detector:
        raise ValueError('Set fvd_detector_sha256 from the verified source-run detector')
    if not Path(cfg['vbench_reference_config']).is_file():
        raise ValueError('Missing saved VBench source configuration')
    bench_reference = load(cfg['vbench_reference_config'])
    if (bench_reference.get('mode') != 'custom_input'
            or set(bench_reference.get('dimensions', [])) != set(DIMENSIONS)
            or bench_reference.get('imaging_quality_preprocessing_mode') != cfg['vbench_imaging_preprocessing']
            or not bench_reference.get('source_evidence')):
        raise ValueError('VBench reference must record matching mode/dimensions/preprocessing and source_evidence')
    if not bench_reference.get('checkpoint_sha256'):
        raise ValueError('VBench reference must identify local detector weights by absolute path and SHA256')
    for path, expected_hash in bench_reference['checkpoint_sha256'].items():
        if not Path(path).is_absolute() or sha(path) != expected_hash:
            raise ValueError(f'VBench checkpoint missing or changed: {path}')
    print('Checking environment imports and selected GPU; imports have no fixed timeout', flush=True)
    for key, imports in [('dfot_python', 'import torch,hydra,omegaconf,lightning,numpy,PIL'),
                         ('metric_python', 'import torch,lpips,imageio'),
                         ('vbench_python', 'import torch,vbench')]:
        logged([cfg[key], '-c', imports + '; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'],
               root / 'logs' / f'{key}.log', gpu,
               cfg['vbench_repo'] if key == 'vbench_python' else None)
    logged([cfg['dfot_python'], '-c',
            'import json,sys; from PIL import Image; '
            'rows=[json.loads(s) for s in open(sys.argv[1]) if s.strip()]; '
            '[Image.open(r["input_image"]).verify() for r in rows]; print("Initial images verified")',
            root / 'inputs' / 'manifest.remapped.jsonl'], root / 'logs' / 'images.log', gpu)
    for key in ('quality_reference_summary', 'vbench_reference_config', 'calibration_file'):
        target = root / 'inputs' / (key + Path(cfg[key]).suffix)
        if target.exists() and sha(target) != sha(cfg[key]):
            raise ValueError(f'{key} changed; use a new study directory')
        shutil.copyfile(cfg[key], target)
    provenance = {'config': cfg, 'checkpoint_sha256': sha(cfg['dfot_checkpoint']),
        'calibration_sha256': sha(cfg['calibration_file']),
        'code': {k: code_identity(cfg[k]) for k in ('dfot_repo', 'memcam_repo', 'vbench_repo')},
        'adapter_sha256': fingerprint({p.name: sha(p) for p in sorted(HERE.glob('*.py'))}),
        'manifest': load(root / 'inputs' / 'manifest.identity.json'),
        'metric_sources': {key: sha(cfg[key]) for key in ('quality_reference_summary', 'vbench_reference_config')},
        'environments': {key: capture([cfg[key], '-m', 'pip', 'freeze']) for key in ('dfot_python', 'metric_python', 'vbench_python')},
        'hardware': capture(['nvidia-smi']), 'cpu': platform.processor(),
        'protocol': {'observed_frames': 1, 'prompt_supported': False, 'camera_input': True,
                     'spatial_transform': 'full-image bicubic resize to 256 square, no crop',
                     'frame_mapping': 'output k -> GT start_frame+k, t=k/30',
                     'training_domain': 'RealEstate10K; not a trained CaM baseline',
                     'dtype': 'float32', 'generation_seed': cfg['seed']}}
    # Hardware utilization varies; it is recorded but is not part of resume identity.
    compatible = {k: v for k, v in provenance.items() if k not in ('hardware', 'cpu')}
    run_id = fingerprint(compatible)
    old = root / 'provenance.json'
    if old.exists() and load(old)['compatibility_sha256'] != run_id:
        raise ValueError('Config/code/environment/checkpoint changed; use a new study directory')
    save(old, dict(provenance, compatibility_sha256=run_id))
    return run_id, calibration


def validate_video(path, row, count, frame_map=None):
    raw = json.loads(capture(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate', '-of', 'json', path]))
    stream = raw['streams'][0]
    if int(stream['nb_read_frames']) != count or Fraction(stream['r_frame_rate']) != Fraction(str(row['fps'])):
        raise ValueError('Wrong decoded frame count or FPS')
    if (stream['width'], stream['height']) != (256, 256):
        raise ValueError('Unexpected native DFoT resolution')
    capture(['ffmpeg', '-v', 'error', '-xerror', '-i', path, '-f', 'null', '-'])
    frames = json.loads(capture(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', path]))['frames']
    if len(frames) != count or any(abs(float(f['best_effort_timestamp_time']) - k / row['fps']) > 0.0001 for k, f in enumerate(frames)):
        raise ValueError('Nonuniform or shifted output timestamps')
    if frame_map:
        with Path(frame_map).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['output_index', 'gt_dataset_index', 'timestamp_seconds'])
            writer.writerows((k, row['start_frame'] + k, f['best_effort_timestamp_time']) for k, f in enumerate(frames))
    return stream


def resume_valid(receipt, expected, video):
    return (receipt.get('identity') == expected and video.is_file()
            and receipt.get('video_sha256') == sha(video))


def generate(cfg, row, index, count, run_id, calibration, root, gpu, smoke=False):
    directory = root / ('smoke' if smoke else RUN)
    directory.mkdir(exist_ok=True)
    video = directory / (row['output_prefix'] + 'custom.mp4')
    receipt_path = video.with_suffix('.receipt.json')
    expected = fingerprint({'run_id': run_id, 'row': row, 'frames': count,
                            'initial_sha256': sha(row['input_image']), 'pose_sha256': sha(row['pose_path'])})
    if receipt_path.exists() and resume_valid(load(receipt_path), expected, video):
        validate_video(video, row, count)
        print(f'KEEP row {index}: {video}', flush=True)
        return load(receipt_path)
    # Never overwrite a completed incompatible video or an unrelated file.
    if video.exists():
        raise ValueError(f'Existing video lacks a compatible receipt: {video}')
    attempt = directory / f'attempt_{index:02d}_{time.time_ns()}'
    attempt.mkdir()
    job = dict(row={k: row[k] for k in ('input_image', 'pose_path', 'start_frame', 'fps')},
               frames=count, seed=cfg['seed'], dfot_repo=cfg['dfot_repo'],
               checkpoint=cfg['dfot_checkpoint'], calibration=calibration[row['scene']],
               output=str(attempt / 'partial.mp4'), resolved_config=str(attempt / 'resolved.yaml'),
               camera_path=str(attempt / 'camera.npy'), resources=str(attempt / 'resources.json'))
    save(attempt / 'job.json', job)
    began = time.monotonic()
    logged([cfg['dfot_python'], '-u', HERE / 'dfot_worker.py', '--job', attempt / 'job.json'],
           attempt / 'generation.log', gpu, cfg['dfot_repo'])
    probe = validate_video(Path(job['output']), row, count, attempt / 'frame_map.csv')
    Path(job['output']).replace(video)
    receipt = dict(identity=expected, video_sha256=sha(video), frames=count,
                   nominal_duration=row['duration_sec'], actual_duration=count/row['fps'],
                   timestamp_span=(count-1)/row['fps'], resources=load(job['resources']),
                   attempt=str(attempt), elapsed_seconds=time.monotonic()-began, probe=probe)
    save(receipt_path, receipt)
    return receipt


def collect(cfg, manifest, quality_dir, bench_dir, run, output, root, gpu):
    logged([cfg['metric_python'], HERE / 'collect_metrics.py', '--memcam-repo', cfg['memcam_repo'],
            '--manifest', manifest, '--quality-dir', quality_dir, '--vbench-dir', bench_dir,
            '--run', run, '--output', output], root / 'logs' / 'collect.log', gpu)
    return load(output)


def evaluate(cfg, rows, manifest, root, gpu):
    qroot, bench = root / 'metrics' / 'quality', root / 'metrics' / 'vbench'
    qdir = qroot / RUN
    bench.mkdir(parents=True, exist_ok=True)
    input_identity = {r['output_prefix']: sha(root / RUN / (r['output_prefix'] + 'custom.mp4')) for r in rows}
    selection = root / 'metrics' / 'input_identity.json'
    if selection.exists() and load(selection) != input_identity:
        raise ValueError('Metric input videos changed; refuse stale metric reuse')
    save(selection, input_identity)
    # Staging contains exactly the canonical cohort, never attempt previews.
    stage = root / 'metrics' / 'inputs' / RUN
    stage.mkdir(parents=True, exist_ok=True)
    expected = [r['output_prefix'] + 'custom.mp4' for r in rows]
    for name in expected:
        target = (root / RUN / name).resolve()
        dest = stage / name
        if not dest.exists():
            dest.symlink_to(target)
        if dest.resolve() != target:
            raise ValueError('Incorrect staged video')
    if sorted(p.name for p in stage.iterdir()) != sorted(expected):
        raise ValueError('Unexpected VBench staged files')
    output = root / 'metrics' / 'scores.json'
    # Re-validate metric artifacts on every resume, including exact identities.
    try:
        return collect(cfg, manifest, qdir, bench, RUN, output, root, gpu)
    except (RuntimeError, OSError, ValueError):
        pass
    quality_valid = False
    if qdir.exists():
        try:
            quality(qdir, rows, RUN)
            quality_valid = True
        except (OSError, ValueError, KeyError):
            qdir.rename(qdir.with_name(RUN + f'.failed_{time.time_ns()}'))
    if not quality_valid:
        command = [cfg['metric_python'], '-u', Path(cfg['memcam_repo']) / 'utils/evaluate_context_memory_prefix_curves.py',
            '--manifest', manifest, '--dataset_root', cfg['dataset_root'], '--model_output_dir', root / RUN,
            '--metrics_dir', qroot, '--run_name', RUN, '--source_duration', '180', '--eval_durations', '180',
            '--learned_metrics', 'lpips,fvd', '--metric_device', 'cuda', '--metric_batch_size', '8',
            '--fvd_detector_path', cfg['fvd_detector'], '--no_fvd_download', '--write_frame_metrics', '--strict']
        for key, value in QUALITY.items():
            command += ['--' + key, str(value)]
        logged(command, root / 'logs' / 'quality.log', gpu, cfg['memcam_repo'])
    try:
        return collect(cfg, manifest, qdir, bench, RUN, output, root, gpu)
    except (RuntimeError, OSError, ValueError):
        if list(bench.iterdir()):
            bench.rename(bench.with_name(f'vbench.failed_{time.time_ns()}'))
            bench.mkdir()
    if not list(bench.glob('*_eval_results.json')):
        logged([cfg['vbench_python'], '-u', Path(cfg['vbench_repo']) / 'evaluate.py',
            '--videos_path', stage, '--output_path', bench, '--mode', 'custom_input',
            '--imaging_quality_preprocessing_mode', cfg['vbench_imaging_preprocessing'],
            '--load_ckpt_from_local', 'True', '--dimension', *DIMENSIONS],
            root / 'logs' / 'vbench.log', gpu, cfg['vbench_repo'])
    return collect(cfg, manifest, qdir, bench, RUN, output, root, gpu)


def export(root, rows, coverage, scores):
    tables = root / 'tables'
    tables.mkdir(exist_ok=True)
    with (tables / 'coverage.csv').open('w', newline='') as handle:
        fields = ['model', 'row', 'scene', 'status', 'frames', 'metrics_complete', 'error']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coverage)
        writer.writerows(dict(model='framepack', row=i, scene=r['scene'], status='excluded_no_camera_input',
                              frames=0, metrics_complete=False, error='') for i, r in enumerate(rows))
    fields = ['model', 'checkpoint', 'camera_input', 'initial_observed_frames', 'nominal_duration',
              'actual_duration', 'N', 'LPIPS', 'FVD', *DIMENSIONS, 'startup_checkpoint_seconds_mean',
              'steady_generation_seconds_mean', 'video_write_seconds_mean', 'peak_cuda_allocated_bytes_max',
              'peak_cuda_reserved_bytes_max', 'peak_process_rss_bytes_max']
    with (tables / 'scores.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(scores)
    def tex(value):
        return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    columns = ['model', 'N', 'LPIPS', 'FVD', *DIMENSIONS]
    lines = [r'\begin{tabular}{l' + 'r' * (len(columns)-1) + '}',
             ' & '.join(tex(c) for c in columns) + r' \\ \hline']
    for row in scores:
        lines.append(' & '.join(tex(f'{row[c]:.4f}' if isinstance(row.get(c), float) else row.get(c, '--')) for c in columns) + r' \\')
    lines.append(r'\end{tabular}')
    (tables / 'comparison.tex').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['audit', 'smoke', 'run'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--approved-hours', type=float)
    p.add_argument('--protocol-approved', action='store_true')
    args = p.parse_args()
    cfg = load(args.config)
    root = Path(cfg['study_root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    rows, coverage, scores = [], [], []
    exit_code = 1
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.stage == 'audit':
                audit(cfg, root)
                return 0
            rows, manifest = prepare(cfg, root)
            coverage = [dict(model=RUN, row=i, scene=r['scene'], status='pending', frames=0,
                             metrics_complete=False, error='') for i, r in enumerate(rows)]
            run_id, calibration = preflight(cfg, rows, root, args.gpu)
            if args.stage == 'smoke':
                receipt = generate(cfg, rows[0], 0, cfg['smoke_frames'], run_id, calibration, root, args.gpu, smoke=True)
                resources = receipt['resources']
                estimate = resources['steady_generation_seconds'] / (cfg['smoke_frames']-1) * sum(r['num_frames']-1 for r in rows) / 3600
                save(root / 'smoke' / 'estimate.json', dict(compatibility_sha256=run_id,
                    generation_hours_linear_estimate=estimate, excludes='metrics and full-length memory growth',
                    warning='short smoke extrapolation is approximate; inspect camera previews, frame map and video'))
                print(f'Approximate generation cost: {estimate:.2f} GPU-hours; full-length memory demand is untested', flush=True)
                exit_code = 0
            else:
                estimate = load(root / 'smoke' / 'estimate.json')
                if estimate['compatibility_sha256'] != run_id:
                    raise ValueError('Smoke configuration differs')
                smoke_video = root / 'smoke' / (rows[0]['output_prefix'] + 'custom.mp4')
                smoke_receipt = load(smoke_video.with_suffix('.receipt.json'))
                expected_smoke = fingerprint({'run_id': run_id, 'row': rows[0], 'frames': cfg['smoke_frames'],
                    'initial_sha256': sha(rows[0]['input_image']), 'pose_sha256': sha(rows[0]['pose_path'])})
                if not resume_valid(smoke_receipt, expected_smoke, smoke_video):
                    raise ValueError('Smoke output or inputs changed; rerun smoke in a compatible study')
                validate_video(smoke_video, rows[0], cfg['smoke_frames'])
                if not args.protocol_approved or not args.approved_hours or args.approved_hours <= 0:
                    raise ValueError('After reviewing smoke, pass --protocol-approved --approved-hours H')
                if estimate['generation_hours_linear_estimate'] > args.approved_hours:
                    raise ValueError('Estimated generation alone exceeds approved hours')
                deadline = time.monotonic() + args.approved_hours * 3600
                save(root / 'approval.json', {'hours': args.approved_hours, 'protocol_approved': True,
                                             'compatibility_sha256': run_id, 'unix_time': time.time()})
                receipts = []
                for i, row in enumerate(rows):
                    print(f'[{i+1}/{len(rows)}] {row["scene"]}; remaining {len(rows)-i}', flush=True)
                    try:
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Approved time exhausted before next video')
                        receipt = generate(cfg, row, i, row['num_frames'], run_id, calibration, root, args.gpu)
                        receipts.append(receipt)
                        coverage[i].update(status='complete', frames=receipt['frames'])
                    except Exception as exc:
                        coverage[i].update(status='failed', error=str(exc))
                        traceback.print_exc()
                    export(root, rows, coverage, scores)
                if len(receipts) != 15:
                    raise RuntimeError('Incomplete generation; metrics require all 15 validated videos')
                if time.monotonic() >= deadline:
                    raise RuntimeError('Approved time exhausted before metrics')
                values = evaluate(cfg, rows, manifest, root, args.gpu)
                for item in coverage:
                    item['metrics_complete'] = True
                score = dict(model=RUN, checkpoint=cfg['checkpoint_provenance'], camera_input=True,
                             initial_observed_frames=1, nominal_duration=180, actual_duration=5397/30, N=15, **values)
                for key in ('startup_checkpoint_seconds', 'steady_generation_seconds', 'video_write_seconds'):
                    score[key + '_mean'] = sum(r['resources'][key] for r in receipts)/15
                for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'peak_process_rss_bytes'):
                    score[key + '_max'] = max(r['resources'][key] for r in receipts)
                scores.append(score)
                # Historical rows are never inferred from headline numbers.
                for existing in cfg.get('existing_results', []):
                    source_rows = [json.loads(line) for line in Path(existing['manifest']).read_text().splitlines() if line.strip()]
                    if identity(source_rows, existing['dataset_root']) != identity(rows, cfg['dataset_root']):
                        raise ValueError('Existing MemCam result has a different cohort')
                    vals = collect(cfg, manifest, existing['quality_dir'], existing['vbench_dir'], existing['run'],
                                   root / 'metrics' / (existing['run'] + '.json'), root, args.gpu)
                    scores.append(dict(model=existing['run'], checkpoint=existing['checkpoint'], camera_input=True,
                                       initial_observed_frames=1, nominal_duration=180, actual_duration=5397/30, N=15, **vals))
                    save(root / 'inputs' / ('source_' + existing['run'] + '.json'), {
                        'source': existing, 'artifact_sha256': {str(p): sha(p) for p in
                        [Path(existing['manifest']), Path(existing['quality_dir'])/'summary.json',
                         Path(existing['quality_dir'])/'metrics.jsonl', *Path(existing['vbench_dir']).glob('*_eval_results.json')]}})
                if cfg.get('vbench_long_command'):
                    command = cfg['vbench_long_command']
                    if not isinstance(command, list) or not all(isinstance(v, str) for v in command) or not Path(command[0]).is_absolute():
                        raise ValueError('vbench_long_command must be an argv list starting with an absolute executable')
                    long_receipt = root / 'metrics' / 'vbench_long_wrapper.json'
                    if not long_receipt.exists():
                        logged(command, root / 'logs' / 'vbench_long.log', args.gpu, cfg['memcam_repo'])
                        save(long_receipt, {'command': command, 'status': 'wrapper_exited_zero',
                                            'note': 'Separate result family; wrapper must validate its own cohort/config'})
                exit_code = 0
        except Exception as exc:
            save(root / 'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            print(f'FAILED: {exc}', file=sys.stderr)
            for item in coverage:
                if item['status'] == 'pending':
                    item.update(status='blocked', error=str(exc))
        finally:
            if args.stage != 'audit':
                if not scores:
                    scores.append(dict(model=RUN, checkpoint=cfg.get('checkpoint_provenance', ''),
                        camera_input=True, initial_observed_frames=1, nominal_duration=180,
                        actual_duration=5397/30, N=sum(r['status'] == 'complete' for r in coverage)))
                export(root, rows, coverage, scores)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
