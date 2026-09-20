"""One-video adapter using upstream DFoT's prediction/interpolation procedure."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    start = time.monotonic()
    print('Importing Torch and upstream DFoT (no import timeout)', flush=True)
    sys.path.insert(0, job['dfot_repo'])
    import numpy as np
    import torch
    from initial_image import read_initial_image
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from utils.hydra_utils import unwrap_shortcuts
    from algorithms.dfot.dfot_video_pose import DFoTVideoPose
    from camera import convert_poses

    class ManifestDFoT(DFoTVideoPose):
        def _sample_sequence(self, *args, **kwargs):
            # Upstream rollouts may end with fewer than max_tokens poses.
            # Match upstream interpolation padding; image padding remains masked
            # with -1 upstream and never becomes an output or observation.
            if not self.use_causal_mask and kwargs.get('conditions') is not None:
                kwargs['conditions'] = self._pad_to_max_tokens(kwargs['conditions'])
            return super()._sample_sequence(*args, **kwargs)

    row, count = job['row'], job['frames']
    overrides = [
        'dataset=realestate10k', 'algorithm=dfot_video_pose',
        'experiment=video_generation', '@diffusion/continuous',
        'dataset.context_length=1', 'dataset.frame_skip=1',
        f'dataset.n_frames={count}', 'algorithm.compile=false',
        'algorithm.logging.metrics=[]', 'algorithm.checkpoint.strict=true',
        'algorithm.tasks.prediction.keyframe_density=0.0625',
        'algorithm.tasks.interpolation.max_batch_size=4',
        'algorithm.tasks.prediction.history_guidance.name=stabilized_vanilla',
        '+algorithm.tasks.prediction.history_guidance.guidance_scale=4.0',
        '+algorithm.tasks.prediction.history_guidance.stabilization_level=0.02',
        'algorithm.tasks.interpolation.history_guidance.name=vanilla',
        '+algorithm.tasks.interpolation.history_guidance.guidance_scale=1.5',
    ]
    config_dir = str(Path(job['dfot_repo']) / 'configurations')
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=unwrap_shortcuts(overrides, config_dir, 'config'))
    if cfg.algorithm.latent.enable:
        raise ValueError('This adapter supports the inspected pixel-space RE10K checkpoint only')
    if cfg.dataset.resolution != 256 or cfg.algorithm.context_frames != 1:
        raise ValueError('Unexpected checkpoint input contract')
    OmegaConf.save(cfg, job['resolved_config'])
    torch.manual_seed(job['seed'])
    np.random.seed(job['seed'])
    print('Loading checkpoint with strict upstream model-key validation', flush=True)
    model = ManifestDFoT(cfg.algorithm)
    checkpoint = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
    if not checkpoint.get('pretrained_ema', False):
        raise ValueError('Expected official released EMA-only RE10K checkpoint; audit other weights separately')
    model.should_validate_ema_weights = True
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model.eval().to('cuda')
    torch.cuda.synchronize()
    loaded = time.monotonic()
    # The only image opened by generation is the initial observed image.
    initial = torch.from_numpy(read_initial_image(row['input_image'])).permute(2, 0, 1).float() / 255
    poses = convert_poses(Path(row['pose_path']), row['start_frame'], count, job['calibration'],
                          allow_nominal=job.get('allow_nominal_calibration', False))
    np.save(job['camera_path'], poses)
    conditions = torch.from_numpy(poses)[None].to('cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    generation_start = time.monotonic()
    with torch.inference_mode():
        # Future slots are zeros, never GT. Upstream _predict_videos uses only
        # the first context frame and then generated keyframes/history.
        xs = torch.zeros((1, count, 3, 256, 256), device='cuda')
        xs[:, 0] = initial.to('cuda')
        predicted = model._predict_videos(model._normalize_x(xs), conditions)
        del xs
        torch.cuda.synchronize()
        generation_end = time.monotonic()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        # Stream frames to ffmpeg; no second full RGB copy on the GPU or CPU.
        writer = subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '256x256', '-r', str(row['fps']), '-i', '-',
            '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', job['output']], stdin=subprocess.PIPE)
        try:
            for k in range(count):
                frame = initial if k == 0 else model._unnormalize_x(predicted[:, k:k + 1])[0, 0].cpu()
                writer.stdin.write((frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype('uint8').tobytes())
        finally:
            writer.stdin.close()
        if writer.wait():
            raise RuntimeError('ffmpeg encoding failed')
    end = time.monotonic()
    metrics = dict(startup_checkpoint_seconds=loaded-start,
        steady_generation_seconds=generation_end-generation_start,
        video_write_seconds=end-generation_end, worker_wall_seconds=end-start,
        generated_frames=count-1, peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        dtype='float32', initial_observed_frames=1, prompt_supported=False,
        camera_conditioned=True, retrieval_latency=None,
        calibration_status=job['calibration'].get('status', 'verified'),
        context_representation='upstream sliding RGB tensor + generated keyframe interpolation',
        output_buffer_note='full prediction tensor retained by upstream; not a retrieval archive')
    Path(job['resources']).write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
