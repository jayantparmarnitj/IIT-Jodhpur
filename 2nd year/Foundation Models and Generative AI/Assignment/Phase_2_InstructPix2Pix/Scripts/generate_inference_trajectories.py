"""
Generate Inference Trajectories for Phase 2 Run A and Run B

This script saves the noisy sample x at 5 equally spaced diffusion timesteps
during the reverse process for Run A and Run B final checkpoints.

Usage:
  python3 generate_inference_trajectories.py \
    --checkpoint_path_a Phase_2_InstructPix2Pix/Results/Run_A/checkpoint_16 \
    --checkpoint_path_b Phase_2_InstructPix2Pix/Results/Run_B/checkpoint_20 \
    --dataset_root ./data/cmp_facade_complete \
    --num_samples 3
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
from PIL import Image
import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing diffusers
try:
    from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
    from diffusers.utils import PIL_INTERPOLATION
except ImportError as e:
    logger.error(f"Failed to import from diffusers: {e}")
    sys.exit(1)


class InferenceTrajectoryGenerator:
    """Generate and save inference trajectories for diffusion models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "mps",
        precision: torch.dtype = torch.float16,
        run_name: str = "Run_A",
    ):
        """
        Initialize trajectory generator with model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use (mps, cuda, cpu)
            precision: Precision dtype (torch.float16 or torch.float32)
            run_name: Name of the run (Run_A or Run_B)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.precision = precision
        self.run_name = run_name
        
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        # Load pipeline
        try:
            self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix",
                torch_dtype=self.precision,
                safety_checker=None,
            )
            
            # Load fine-tuned UNet from checkpoint (local directory)
            # The checkpoint contains diffusion_pytorch_model.safetensors directly
            try:
                unet = UNet2DConditionModel.from_pretrained(
                    str(self.checkpoint_path),
                    torch_dtype=self.precision,
                )
                self.pipeline.unet = unet
                logger.info(f"✓ Loaded fine-tuned UNet from {self.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load fine-tuned UNet: {e}")
                logger.warning(f"Using pretrained UNet instead")
            
            self.pipeline = self.pipeline.to(self.device)
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
            
            logger.info(f"✓ Model loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_trajectory(
        self,
        input_image: Image.Image,
        label_image: Image.Image,
        prompt: str = "turn this semantic facade label map into a photorealistic building facade",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        seed: int = 42,
        num_timesteps: int = 5,
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Generate trajectory of noisy samples at equally spaced timesteps.
        
        Args:
            input_image: Input photorealistic image (for reference)
            label_image: Semantic label map (conditioning)
            prompt: Text prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Text guidance scale
            image_guidance_scale: Image guidance scale
            seed: Random seed
            num_timesteps: Number of equally spaced timesteps to save
        
        Returns:
            List of (noisy_sample, timestep_index) tuples
        """
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Prepare images - resize to 512x512
        input_image = input_image.resize((512, 512), Image.LANCZOS)
        label_image = label_image.resize((512, 512), Image.NEAREST)
        
        trajectory = []
        steps_to_save = np.linspace(0, num_inference_steps - 1, num_timesteps, dtype=int)
        
        logger.info(f"Generating trajectory with {num_timesteps} timesteps at steps: {steps_to_save}")
        
        # Use pipeline's scheduler to get timesteps
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps
        
        with torch.no_grad():
            # Prepare text embeddings
            text_input = self.pipeline.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.pipeline.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]
            
            # Unconditional embeddings for classifier-free guidance
            uncond_input = self.pipeline.tokenizer(
                [""],
                padding="max_length",
                max_length=text_input.input_ids.shape[-1],
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.pipeline.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
            
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            # Encode conditioning images
            input_latents = self.pipeline.vae.encode(
                self.pipeline.image_processor.preprocess(input_image).to(
                    self.device, dtype=self.precision
                )
            ).latent_dist.mean
            input_latents = input_latents * self.pipeline.vae.config.scaling_factor
            
            label_latents = self.pipeline.vae.encode(
                self.pipeline.image_processor.preprocess(label_image).to(
                    self.device, dtype=self.precision
                )
            ).latent_dist.mean
            label_latents = label_latents * self.pipeline.vae.config.scaling_factor
            
            # Initial noise
            shape = input_latents.shape
            noise = torch.randn(shape, dtype=self.precision, device=self.device)
            latents = noise
            
            # Diffusion loop with trajectory saving
            for step_idx, t in enumerate(tqdm.tqdm(timesteps, desc=f"Diffusion ({self.run_name})")):
                # Save latent at specified steps
                if step_idx in steps_to_save:
                    try:
                        decoded = self.pipeline.vae.decode(
                            latents / self.pipeline.vae.config.scaling_factor,
                            return_dict=False
                        )[0]
                        sample = self.pipeline.image_processor.postprocess(
                            decoded, output_type="np"
                        )[0]
                        trajectory.append((sample, int(t)))
                        logger.info(f"  ✓ Saved at step {step_idx}/{num_inference_steps}, t={int(t)}")
                    except Exception as e:
                        logger.warning(f"Could not save trajectory at step {step_idx}: {e}")
                
                # Prepare latent model input
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.pipeline.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                # Concatenate with conditioning
                latent_model_input = torch.cat([
                    latent_model_input,
                    torch.cat([label_latents] * 2)
                ], dim=1)
                
                # Predict noise
                noise_pred = self.pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )[0]
                
                # Classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                
                # Image guidance
                noise_pred = noise_pred + image_guidance_scale * (
                    noise_pred - torch.zeros_like(noise_pred)
                )
                
                # Scheduler step
                latents = self.pipeline.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
        
        return trajectory
    
    def save_trajectories(
        self,
        trajectories: List[Tuple[np.ndarray, int]],
        output_dir: Path,
        sample_idx: int,
    ):
        """Save trajectory samples as PNG images."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for step_idx, (sample, timestep) in enumerate(trajectories):
            # Convert to 0-255
            sample = (sample * 255).astype(np.uint8)
            if sample.shape[-1] == 3:  # RGB
                img = Image.fromarray(sample)
            else:  # Grayscale
                img = Image.fromarray(sample[:, :, 0])
            
            # Save with descriptive name
            filename = f"trajectory_{self.run_name}_sample{sample_idx:02d}_step{step_idx:02d}_t{timestep}.png"
            filepath = output_dir / filename
            img.save(filepath)
            logger.info(f"  ✓ Saved {filename}")


def load_dataset_samples(dataset_root: str, num_samples: int = 3) -> List[Tuple[Image.Image, Image.Image]]:
    """Load dataset samples for trajectory generation."""
    dataset_path = Path(dataset_root)
    
    # Load images from test set
    data_info_path = dataset_path / "data_info.json"
    
    if not data_info_path.exists():
        logger.warning(f"data_info.json not found, using first {num_samples} images")
        # Fallback: load first N images
        image_dir = dataset_path / "images"
        label_dir = dataset_path / "labels"
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        samples = []
        for i, img_file in enumerate(image_files[:num_samples]):
            label_file = label_dir / f"{img_file.stem}.png"
            if label_file.exists():
                img = Image.open(img_file).convert("RGB")
                label = Image.open(label_file).convert("RGB")
                samples.append((img, label))
        return samples
    
    # Load from data_info.json
    import json
    with open(data_info_path) as f:
        data_info = json.load(f)
    
    samples = []
    test_samples = data_info.get("test", [])[:num_samples]
    
    for sample_info in test_samples:
        img_path = dataset_path / sample_info["image"]
        label_path = dataset_path / sample_info["label"]
        
        if img_path.exists() and label_path.exists():
            img = Image.open(img_path).convert("RGB")
            label = Image.open(label_path).convert("RGB")
            samples.append((img, label))
    
    logger.info(f"Loaded {len(samples)} samples for trajectory generation")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate inference trajectories for Phase 2 Run A and Run B"
    )
    parser.add_argument(
        "--checkpoint_path_a",
        type=str,
        default="Phase_2_InstructPix2Pix/Results/Run_A/checkpoint_16",
        help="Path to Run A checkpoint",
    )
    parser.add_argument(
        "--checkpoint_path_b",
        type=str,
        default="Phase_2_InstructPix2Pix/Results/Run_B/checkpoint_20",
        help="Path to Run B checkpoint",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./data/cmp_facade_complete",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Phase_2_InstructPix2Pix/Results/Inference_Trajectories",
        help="Output directory for trajectories",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples per run",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=5,
        help="Number of equally spaced timesteps to save",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Text guidance scale",
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help="Image guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("GENERATING INFERENCE TRAJECTORIES FOR PHASE 2")
    logger.info("="*80)
    
    # Load dataset samples
    logger.info(f"\nLoading dataset from {args.dataset_root}")
    samples = load_dataset_samples(args.dataset_root, args.num_samples)
    
    if not samples:
        logger.error("No samples loaded!")
        return
    
    # Generate trajectories for Run A
    logger.info(f"\n{'='*80}")
    logger.info("RUN A - NATIVE SCHEDULE")
    logger.info(f"{'='*80}")
    
    gen_a = InferenceTrajectoryGenerator(
        args.checkpoint_path_a,
        run_name="Run_A",
    )
    
    for sample_idx, (input_img, label_img) in enumerate(samples):
        logger.info(f"\nGenerating trajectory for Run A, sample {sample_idx}")
        trajectory = gen_a.generate_trajectory(
            input_img,
            label_img,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            seed=args.seed,
            num_timesteps=args.num_timesteps,
        )
        gen_a.save_trajectories(trajectory, output_dir / "Run_A", sample_idx)
    
    # Generate trajectories for Run B
    logger.info(f"\n{'='*80}")
    logger.info("RUN B - ABLATED SCHEDULE (x=2.20)")
    logger.info(f"{'='*80}")
    
    gen_b = InferenceTrajectoryGenerator(
        args.checkpoint_path_b,
        run_name="Run_B",
    )
    
    for sample_idx, (input_img, label_img) in enumerate(samples):
        logger.info(f"\nGenerating trajectory for Run B, sample {sample_idx}")
        trajectory = gen_b.generate_trajectory(
            input_img,
            label_img,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            seed=args.seed,
            num_timesteps=args.num_timesteps,
        )
        gen_b.save_trajectories(trajectory, output_dir / "Run_B", sample_idx)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Inference trajectories saved to {output_dir}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
