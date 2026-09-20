"""Preserve and remap an existing cohort. Does not generate a new split."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

PATH_FIELDS = ('input_image', 'pose_path', 'gt_frames_dir', 'overlap_dir')
REQUIRED = ('scene', 'start_frame', 'duration_sec', 'fps', 'num_frames',
            'input_image', 'pose_path', 'gt_frames_dir', 'prompt',
            'output_prefix', 'split_id', 'split_seed')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(rows, root):
    """Retain relative file identities as well as every non-path field."""
    normalized = []
    for row in rows:
        item = dict(row)
        for key in PATH_FIELDS:
            if item.get(key) is not None:
                item[key] = str(PurePosixPath(item[key]).relative_to(root))
        normalized.append(item)
    return digest(json.dumps(normalized, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode())


def remap(rows, source_root, target_root, expected_rows=15, check_paths=True):
    if len(rows) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows, found {len(rows)}')
    result, names = [], set()
    for index, row in enumerate(rows):
        missing = set(REQUIRED) - row.keys()
        if missing:
            raise ValueError(f'Row {index}: missing {sorted(missing)}')
        for key in ('start_frame', 'num_frames'):
            if type(row[key]) is not int or row[key] < (1 if key == 'num_frames' else 0):
                raise ValueError(f'Row {index}: invalid {key}')
        for key in ('duration_sec', 'fps'):
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError(f'Row {index}: invalid {key}')
        name = row['output_prefix']
        if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in names:
            raise ValueError(f'Row {index}: unsafe or duplicate output prefix')
        names.add(name)
        item = dict(row)
        for key in PATH_FIELDS:
            if item.get(key) is None:
                continue
            path = PurePosixPath(item[key])
            if '..' in path.parts:
                raise ValueError(f'Row {index}: parent traversal in {key}')
            relative = path.relative_to(source_root)
            item[key] = str(Path(target_root) / str(relative))
            if check_paths:
                p = Path(item[key])
                valid = p.is_dir() if key.endswith('_dir') else p.is_file()
                if not valid:
                    raise ValueError(f'Row {index}: missing {key}: {p}')
        result.append(item)
    if identity(rows, source_root) != identity(result, target_root):
        raise ValueError('Cohort identity changed during remapping')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--target-root', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    original = args.manifest.read_bytes()
    rows = [json.loads(line) for line in original.splitlines() if line.strip()]
    mapped = remap(rows, args.source_root, args.target_root)
    # A new directory prevents accidental modification of a previous study.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'manifest.original.jsonl').write_bytes(original)
    serialized = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in mapped).encode()
    (args.output_dir / 'manifest.remapped.jsonl').write_bytes(serialized)
    provenance = dict(source_manifest=str(args.manifest.resolve()),
                      original_sha256=digest(original), remapped_sha256=digest(serialized),
                      cohort_sha256=identity(rows, args.source_root),
                      source_root=args.source_root, target_root=args.target_root,
                      rows=len(rows), status='inputs_only_not_generation_validated')
    (args.output_dir / 'manifest.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
