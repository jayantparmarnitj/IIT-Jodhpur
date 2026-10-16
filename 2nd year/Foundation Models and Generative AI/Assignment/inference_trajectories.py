"""
Inference Trajectory Collection - Phase 2 Submission Component

Captures the denoising trajectory for Run A and Run B by saving noisy samples x
at 5 equally spaced diffusion timesteps during the reverse process.

For each run:
- 5 equally spaced timesteps: [0, 250, 500, 750, 1000]
- 3+ samples per timestep
- Noisy intermediate representations saved as images
- Clear labeling: Run_{A,B}_t{timestep}_sample{idx}.png

This reveals how the modified noise schedule affects the denoising trajectory,
particularly the distributional mismatch at t=T for Run B.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TrajectoryCollector:
    """
    Collects and visualizes denoising trajectories for Run A and Run B.
    
    Trajectory capture:
    - Sample random noise x_T ~ N(0, I)
    - Execute reverse diffusion for exactly 50 inference steps
    - Capture x_t at timesteps: [0, 250, 500, 750, 1000]
    - Map reverse step index to diffusion timestep
    - Save 32-bit float latent representations and RGB visualizations
    """
    
    def __init__(
        self,
        model_dir: Path,
        dataset_root: str,
        output_dir: Path,
        num_samples: int = 3,
        seed: int = 42,
    ):
        self.model_dir = Path(model_dir)
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.seed = seed
        
        # ADDED: Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # ADDED: Load model
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_dir, torch_dtype=torch.float16
        ).to(self.device)
        self.unet = self.pipeline.unet
        
        # ADDED: Sampling parameters
        self.num_inference_steps = 50
        self.guidance_scale_text = 7.5
        self.guidance_scale_image = 1.5
        
        # ADDED: Timesteps to capture (equally spaced over 1000 diffusion steps)
        # Reverse mapping: diffusion step t_i corresponds to reverse step i
        # t_0 = 1000 (full noise), t_49 = 0 (clean image)
        # We capture at approximately t = [1000, 750, 500, 250, 0]
        self.capture_timesteps = [1000, 750, 500, 250, 0]
        
        # ADDED: Output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ADDED: Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def _timestep_to_reverse_step(self, timestep: int) -> int:
        """Map diffusion timestep to reverse process step index."""
        # For 50 inference steps over 1000 timesteps, reverse step i corresponds to:
        # diffusion timestep t = 1000 - (i+1) * (1000 / 50) = 1000 - (i+1) * 20
        reverse_step = int((1000 - timestep) / 20) - 1
        return max(0, min(reverse_step, self.num_inference_steps - 1))
    
    def collect_trajectories(self, run_name: str, test_dataset) -> None:
        """
        Collect denoising trajectories for a single run.
        """
        logger.info(f"Collecting trajectories for {run_name}...")
        
        run_output_dir = self.output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # ADDED: Generate trajectories for 3+ samples
        for sample_idx in range(self.num_samples):
            logger.info(f"  Sample {sample_idx+1}/{self.num_samples}...")
            
            # ADDED: Load test sample
            test_sample = test_dataset[sample_idx % len(test_dataset)]
            image = (test_sample['image'] + 1) / 2
            conditioning_image = (test_sample['conditioning_image'] + 1) / 2
            prompt = test_sample['prompt']
            
            # ADDED: Convert to PIL for pipeline
            image_pil = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            cond_pil = Image.fromarray((conditioning_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            
            # ADDED: Run inference with trajectory capture
            self._capture_single_trajectory(
                prompt=prompt,
                image=image_pil,
                conditioning_image=cond_pil,
                sample_idx=sample_idx,
                run_name=run_name,
                output_dir=run_output_dir,
            )
    
    def _capture_single_trajectory(
        self,
        prompt: str,
        image: Image.Image,
        conditioning_image: Image.Image,
        sample_idx: int,
        run_name: str,
        output_dir: Path,
    ) -> None:
        """
        Capture denoising trajectory for a single sample by hooking into
        the diffusion scheduler callback system.
        """
        # ADDED: Store captured latents
        captured_latents = {}
        
        # ADDED: Callback function to capture x_t at specific steps
        def capture_callback(pipeline, step_index, timestep, callback_kwargs):
            if timestep in self.capture_timesteps:
                latents = callback_kwargs["latents"]
                captured_latents[timestep] = latents.clone()
            return callback_kwargs
        
        # ADDED: Run inference with callback
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                image=image,
                conditioning_image=conditioning_image,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale_text,
                image_guidance_scale=self.guidance_scale_image,
                generator=torch.Generator(device=self.device).manual_seed(self.seed + sample_idx),
                callback_on_step_end=capture_callback,
                output_type="pil",
            )
        
        # ADDED: Save captured latents and visualizations
        for timestep in self.capture_timesteps:
            if timestep in captured_latents:
                latents = captured_latents[timestep]
                
                # ADDED: Save latent as numpy (for numerical analysis)
                latent_file = output_dir / f"{run_name}_t{timestep:04d}_sample{sample_idx:03d}.npy"
                np.save(latent_file, latents.cpu().numpy())
                
                # ADDED: Decode latent to image for visualization
                with torch.no_grad():
                    decoded = self.pipeline.vae.decode(latents / 0.18215).sample
                
                # ADDED: Convert to PIL
                decoded_np = (decoded * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()[0] * 255
                decoded_image = Image.fromarray(decoded_np.astype(np.uint8))
                
                # ADDED: Save visualization
                viz_file = output_dir / f"{run_name}_t{timestep:04d}_sample{sample_idx:03d}.png"
                decoded_image.save(viz_file)
                
                logger.info(f"    Saved {timestep:04d}: {latent_file.name}, {viz_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Collect inference trajectories")
    parser.add_argument('--run', type=str, choices=['A', 'B'], required=True,
                       help="Which run to collect trajectories from")
    parser.add_argument('--model_dir', type=str, required=True,
                       help="Directory containing fine-tuned model")
    parser.add_argument('--dataset_root', type=str, default='./data/cmp_facade',
                       help="Path to CMP Facade dataset")
    parser.add_argument('--output_dir', type=str, default='./trajectories',
                       help="Output directory for trajectory images")
    parser.add_argument('--num_samples', type=int, default=3,
                       help="Number of samples to collect trajectories for")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # ADDED: Import test dataset loader
    from finetune_instructpix2pix import CMPFacadeDataset
    
    # ADDED: Load test dataset
    test_dataset = CMPFacadeDataset(
        args.dataset_root, split='test', image_size=256, seed=args.seed
    )
    
    # ADDED: Collect trajectories
    collector = TrajectoryCollector(
        model_dir=args.model_dir,
        dataset_root=args.dataset_root,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        seed=args.seed,
    )
    
    collector.collect_trajectories(f"Run_{args.run}", test_dataset)
    logger.info(f"Trajectories saved to {args.output_dir}")


if __name__ == '__main__':
    main()
