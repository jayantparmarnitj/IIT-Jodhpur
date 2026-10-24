"""
Noise Schedule Ablation in Diffusion Models - Phase 2 Implementation
Group: Jash Lathiya, Jayant Parmar, Kumar Viplav, Mamata Das

This script implements Run 0 (zero-shot baseline), Run A (fine-tuning with native schedule),
and Run B (fine-tuning with power-transformed schedule) for the CMP Facade dataset.

Environment:
- Python 3.10+
- PyTorch 2.0+
- Diffusers 0.21.0+
- transformers 4.30.0+
- PIL, numpy, tqdm

Commands to reproduce:
  Run 0:  python finetune_instructpix2pix.py --run 0
  Run A:  python finetune_instructpix2pix.py --run A --epochs 20 --batch_size 4
  Run B:  python finetune_instructpix2pix.py --run B --epochs 20 --batch_size 4 --ablated_schedule
"""

import os
import sys
import argparse
import json
import random
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import cv2

from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import tqdm

# ADDED: Import exponent constant
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from EXPONENT import NOISE_SCHEDULE_EXPONENT

# MODIFIED: Suppress CUDA AMP warnings on MPS device
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch_dtype.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA is not available.*")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CMPFacadeDataset(Dataset):
    """
    CMP Facade Database loader.
    
    Dataset structure:
      data/
        ├── images/              # 606 rectified facade photographs
        └── labels/              # 12-class semantic segmentation masks
    
    Preprocessing:
    - All images resized to 256x256
    - Semantic labels resized via nearest-neighbor to prevent class blending
    - Fixed random seed (42) for 400 training, 100 validation, 106 test split
    """
    
    def __init__(
        self,
        dataset_root: str,
        split: str = 'train',
        image_size: int = 256,
        seed: int = 42,
        condition_prompt: str = "turn this semantic facade label map into a photorealistic building facade"
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.condition_prompt = condition_prompt
        
        # ADDED: Color palette for semantic labels (12 classes)
        # Maps class index to RGB value for visualization and nearest-neighbor preservation
        self.color_palette = {
            0: (0, 0, 0),           # background
            1: (128, 0, 0),         # facade
            2: (0, 128, 0),         # molding
            3: (0, 0, 128),         # cornice
            4: (128, 128, 0),       # pillar
            5: (128, 0, 128),       # window
            6: (0, 128, 128),       # door
            7: (192, 0, 0),         # sill
            8: (0, 192, 0),         # blind
            9: (0, 0, 192),         # balcony
            10: (192, 192, 0),      # shop
            11: (192, 0, 192),      # deco
            12: (0, 192, 192),      # background variant
        }
        
        # ADDED: Load and partition dataset
        self.image_paths = self._load_and_partition_dataset(seed)
    
    def _load_and_partition_dataset(self, seed: int) -> List[str]:
        """
        Load all images and partition using fixed random seed.
        Returns paths for the configured split.
        
        MODIFIED: Handle small datasets (< 606 images) by distributing proportionally.
        """
        image_dir = self.dataset_root / "images"
        
        if not image_dir.exists():
            raise ValueError(f"Dataset directory not found: {image_dir}")
        
        # ADDED: Collect all valid image paths
        all_images = sorted([
            str(p) for p in image_dir.glob("*.jpg")
            if (self.dataset_root / "labels" / (p.stem + ".png")).exists()
        ])
        
        if not all_images:
            raise ValueError(f"No valid image-label pairs found in {self.dataset_root}")
        
        # ADDED: Partition with fixed seed
        rng = random.Random(seed)
        rng.shuffle(all_images)
        
        # MODIFIED: Handle small datasets by scaling proportions
        total_images = len(all_images)
        
        if total_images >= 606:
            # Original split: 400 train, 100 validation, 106 test
            num_train = 400
            num_val = 100
        else:
            # Scale for small datasets: ~66% train, ~17% validation, ~17% test
            num_train = max(1, int(total_images * 0.66))
            num_val = max(1, int(total_images * 0.17))
        
        logger.info(f"Dataset split ({total_images} total images): "
                   f"train={num_train}, val={num_val}, "
                   f"test={max(0, total_images - num_train - num_val)}")
        
        if self.split == 'train':
            return all_images[:num_train]
        elif self.split == 'validation':
            return all_images[num_train:num_train + num_val]
        elif self.split == 'test':
            return all_images[num_train + num_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        image_path = self.image_paths[idx]
        label_path = str(self.dataset_root / "labels" / (Path(image_path).stem + ".png"))
        
        # ADDED: Load image and label
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        
        # ADDED: Resize to 256x256
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        # MODIFIED: Use nearest-neighbor for labels to prevent class blending
        label = label.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # ADDED: Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0  # [-1, 1]
        label = torch.from_numpy(np.array(label)).permute(2, 0, 1).float() / 127.5 - 1.0   # [-1, 1]
        
        return {
            'image': image,
            'conditioning_image': label,
            'prompt': self.condition_prompt,
        }


class NativeNoiseScheduler:
    """
    Stable Diffusion v1.5 native noise schedule.
    
    Native parameterization: θ_t = β_t (per-step variance)
    Bounded condition: β_t ∈ (0, 1) for all t ∈ {1, 2, ..., 1000}
    
    Native schedule parameters:
    - β_start = 0.00085
    - β_end = 0.0120
    - T = 1000 timesteps
    - Schedule: scaled linear in √β space
    """
    
    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        
        # ADDED: Native schedule bounds (Stable Diffusion v1.5)
        beta_start = 0.00085
        beta_end = 0.0120
        
        # ADDED: Compute native betas using scaled linear schedule
        betas = torch.linspace(
            np.sqrt(beta_start), np.sqrt(beta_end), num_train_timesteps, dtype=torch.float32
        ) ** 2
        
        # ADDED: Verify boundedness: β_t ∈ (0, 1) for all t
        assert betas.min() > 0 and betas.max() < 1, \
            f"Native schedule violates boundedness: [{betas.min()}, {betas.max()}]"
        
        # ADDED: Precompute schedule quantities (float32 for MPS compatibility)
        self.betas = betas.float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        logger.info(f"Native schedule: β_1={self.betas[0]:.6f}, β_1000={self.betas[-1]:.6f}")
        logger.info(f"Native SNR(1)={1.0 / (1.0 - self.alphas_cumprod[0]):.2f}, "
                   f"SNR(1000)={self.alphas_cumprod[-1] / (1.0 - self.alphas_cumprod[-1]):.6f}")


class AblatedNoiseScheduler:
    """
    Power-transformed noise schedule for ablation.
    
    Transform: θ_t^x = β_t^x where x = 2.20 ∈ (1.5, 3.0)
    
    Effect: For β_t ∈ (0, 1) and x > 1:
    - β_t^x < β_t (shrinks every variance value)
    - β_1^2.20 ≈ 1.76 × 10^-7
    - β_1000^2.20 ≈ 5.95 × 10^-5
    - α_bar,1000' ≈ 0.985 (retains ~98.5% signal variance)
    
    Mechanistic implication: Run B never encounters near-pure noise during fine-tuning,
    creating distributional mismatch at t=T during reverse sampling.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, exponent: float = NOISE_SCHEDULE_EXPONENT):
        self.num_train_timesteps = num_train_timesteps
        self.exponent = exponent
        
        # ADDED: Native schedule (same as NativeNoiseScheduler)
        beta_start = 0.00085
        beta_end = 0.0120
        betas = torch.linspace(
            np.sqrt(beta_start), np.sqrt(beta_end), num_train_timesteps, dtype=torch.float32
        ) ** 2
        
        # ADDED: Apply power transformation (float32 for MPS)
        self.betas_native = betas.float()
        self.betas_ablated = (self.betas_native ** exponent)
        
        # ADDED: Verify ablated schedule boundedness
        assert self.betas_ablated.min() > 0 and self.betas_ablated.max() < 1, \
            f"Ablated schedule violates boundedness: [{self.betas_ablated.min()}, {self.betas_ablated.max()}]"
        
        # ADDED: Precompute ablated schedule quantities
        self.alphas = 1.0 - self.betas_ablated
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # ADDED: Log schedule statistics
        logger.info(f"Ablated schedule (x={exponent}): β_1^x={self.betas_ablated[0]:.2e}, "
                   f"β_1000^x={self.betas_ablated[-1]:.2e}")
        logger.info(f"Ablated SNR(1)={1.0 / (1.0 - self.alphas_cumprod[0]):.2e}, "
                   f"SNR(1000)={self.alphas_cumprod[-1] / (1.0 - self.alphas_cumprod[-1]):.2f}")


class InstructPix2PixFineTuner:
    """
    Fine-tuner for InstructPix2Pix on CMP Facade dataset.
    
    Hyperparameters (identical for Run A and Run B):
    - Epochs: 20
    - Batch size: 4
    - Learning rate: 1e-5
    - Optimizer: AdamW with weight decay 1e-2
    - Training steps per epoch: 100
    - Total steps: 2000
    - Checkpoints: Every 4 epochs (5 checkpoints)
    - Precision: FP16 mixed precision
    - Random seed: 42
    """
    
    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        dataset_root: str = "./data/cmp_facade",
        output_dir: str = "./outputs",
        run_name: str = "Run_A",
        use_ablated_schedule: bool = False,
        seed: int = 42,
        batch_size: int = 4,
        max_steps: Optional[int] = None,
    ):
        self.model_id = model_id
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir) / run_name
        self.run_name = run_name
        self.use_ablated_schedule = use_ablated_schedule
        self.seed = seed
        self.max_steps = max_steps  # ADDED: Maximum steps (overrides epoch logic)
        
        # ADDED: Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # ADDED: Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MODIFIED: Device configuration - prefer MPS for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            device_type = "MPS (GPU)"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_type = "CUDA (GPU)"
        else:
            self.device = torch.device("cpu")
            device_type = "CPU"
        
        logger.info(f"=" * 60)
        logger.info(f"DEVICE SELECTION CONFIRMED")
        logger.info(f"  MPS Available: {torch.backends.mps.is_available()}")
        logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"  Selected Device: {self.device}")
        logger.info(f"  Device Type: {device_type}")
        logger.info(f"=" * 60)
        
        # FIXED: Set dtype BEFORE loading model (it's needed for model loading)
        # Use fp16 mixed precision for speed on MPS
        # MPS supports torch.float16 which gives 2-3x speedup with minimal accuracy loss
        self.dtype = torch.float16
        logger.info(f"Using precision: {self.dtype} (fp16 for UNet training on MPS)")
        
        # ADDED: Load pretrained pipeline
        logger.info(f"Loading pretrained model from {model_id}...")
        # FIXED: Load in fp32 to avoid dtype mismatches with input images
        # We'll only use fp16 for the UNet training, not for VAE/CLIP encoding
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        self.pipeline = self.pipeline.to(self.device)
        
        # ADDED: Verify model is on correct device
        model_device = next(self.pipeline.unet.parameters()).device
        logger.info(f"Model loaded on device: {model_device}")
        if str(model_device) != str(self.device):
            logger.warning(f"WARNING: Model device mismatch! Expected {self.device}, got {model_device}")
        
        # ADDED: Extract UNet for fine-tuning
        self.unet = self.pipeline.unet
        
        # ADDED: Hyperparameters
        self.num_epochs = 20
        self.batch_size = batch_size  # FIXED: Now uses parameter from __init__
        self.learning_rate = 1e-5
        self.weight_decay = 1e-2
        self.num_training_steps = 100  # steps per epoch
        self.total_steps = self.num_epochs * self.num_training_steps
        self.checkpoint_interval = 4  # epochs
        
        logger.info(f"Fine-tuning configuration: {self.num_epochs} epochs, "
                   f"{self.batch_size} batch size, {self.learning_rate} lr")
        logger.info(f"Use ablated schedule: {use_ablated_schedule}")
        if self.max_steps is not None:
            logger.info(f"QUICK VALIDATION MODE: max_steps={self.max_steps} (will stop after ~{self.max_steps * self.batch_size} images)")
    
    def load_datasets(self) -> Tuple[DataLoader, DataLoader]:
        """Load training and validation datasets."""
        logger.info(f"Loading datasets from {self.dataset_root}...")
        
        # ADDED: Load train and validation datasets
        train_dataset = CMPFacadeDataset(
            self.dataset_root, split='train', image_size=256, seed=self.seed
        )
        val_dataset = CMPFacadeDataset(
            self.dataset_root, split='validation', image_size=256, seed=self.seed
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        return train_loader, val_loader
    
    def finetune(self):
        """Execute fine-tuning for Run A or Run B."""
        train_loader, val_loader = self.load_datasets()
        
        # ADDED: Setup optimizer
        optimizer = AdamW(
            self.unet.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # FIXED: Build the noise scheduler ONCE, outside the training loop.
        if self.use_ablated_schedule:
            scheduler = AblatedNoiseScheduler()
        else:
            scheduler = NativeNoiseScheduler()
        alphas_cumprod = scheduler.alphas_cumprod.to(self.device).float()
        
        # ADDED: Training logs
        logs = {'epoch': [], 'train_loss': []}
        
        # FIXED: Global step counter for max_steps logic
        global_step = 0
        
        # ADDED: Fine-tuning loop
        for epoch in range(self.num_epochs):
            self.unet.train()
            epoch_loss = 0.0
            
            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # ADDED: Check if we've hit max_steps limit
                if self.max_steps is not None and global_step >= self.max_steps:
                    logger.info(f"Reached max_steps={self.max_steps}, stopping training early")
                    logger.info(f"✓ Quick validation complete after {global_step} steps and {epoch+1} epochs")
                    self._save_logs(logs)
                    return logs
                
                # ADDED: Move batch to device
                images = batch['image'].to(self.device)
                conditioning_images = batch['conditioning_image'].to(self.device)
                prompts = batch['prompt']
                
                # ADDED: Encode text prompts
                text_embeddings = self.pipeline.text_encoder(
                    self.pipeline.tokenizer(prompts, return_tensors="pt").input_ids.to(self.device)
                )[0]
                
                # ADDED: Encode images to latent space
                with torch.no_grad():
                    latents = self.pipeline.vae.encode(images).latent_dist.sample() * 0.18215
                    conditioning_latents = self.pipeline.vae.encode(conditioning_images).latent_dist.sample() * 0.18215
                
                # ADDED: Sample random timesteps
                timesteps = torch.randint(0, 1000, (images.shape[0],), device=self.device)
                
                # ADDED: Add noise (forward diffusion step)
                # NOTE: alphas_cumprod is precomputed once before the loop.
                noise = torch.randn_like(latents)
                
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
                
                noisy_latents = sqrt_alphas_cumprod * latents + sqrt_one_minus_alphas_cumprod * noise
                
                # ADDED: Concatenate conditioning latents
                latent_model_input = torch.cat([noisy_latents, conditioning_latents], dim=1)
                
                # ADDED: Forward pass (keep in fp32 for stability)
                noise_pred = self.unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # ADDED: Compute MSE loss
                loss = F.mse_loss(noise_pred, noise)
                
                # ADDED: Backward pass (MPS-compatible)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # FIXED: Periodically release cached MPS memory
                if str(self.device) == "mps" and (batch_idx + 1) % 20 == 0:
                    torch.mps.empty_cache()
                
                global_step += 1
            
            # ADDED: Log epoch loss
            avg_epoch_loss = epoch_loss / len(train_loader)
            logs['epoch'].append(epoch + 1)
            logs['train_loss'].append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.6f}")
            
            # ADDED: Save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                checkpoint_dir = self.output_dir / f"checkpoint_{epoch+1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self.unet.save_pretrained(checkpoint_dir)
                    logger.info(f"✓ Saved checkpoint to {checkpoint_dir}")
                except Exception as e:
                    logger.error(f"ERROR SAVING CHECKPOINT: {e}")
                    raise  # Don't silently fail
        
        # ADDED: Save final model
        final_model_dir = self.output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.unet.save_pretrained(final_model_dir)
            logger.info(f"✓ Saved final model to {final_model_dir}")
        except Exception as e:
            logger.error(f"ERROR SAVING FINAL MODEL: {e}")
            raise  # Don't silently fail
        
        # ADDED: Save logs to CSV
        try:
            self._save_logs(logs)
        except Exception as e:
            logger.error(f"ERROR SAVING LOGS: {e}")
            raise  # Don't silently fail
        
        return logs
    
    def _save_logs(self, logs: Dict):
        """Save training logs to CSV."""
        import csv
        
        log_file = self.output_dir / "training_logs.csv"
        try:
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss'])
                for epoch, loss in zip(logs['epoch'], logs['train_loss']):
                    writer.writerow([epoch, loss])
            logger.info(f"✓ Saved training logs to {log_file}")
        except Exception as e:
            logger.error(f"ERROR SAVING LOGS: {e}")
            raise


class InferenceRunner:
    """
    Run inference for Run 0 (zero-shot baseline), Run A, and Run B.
    
    Sampling configuration (identical across all runs):
    - Inference steps: 50
    - Random seed: 42
    - Text guidance scale: 7.5
    - Image guidance scale: 1.5
    """
    
    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        dataset_root: str = "./data/cmp_facade",
        output_dir: str = "./outputs",
        seed: int = 42,
    ):
        self.model_id = model_id
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # ADDED: Device configuration
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # ADDED: Load pretrained pipeline for Run 0
        logger.info(f"Loading pretrained model for Run 0 baseline...")
        self.pipeline_run0 = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(self.device)
        
        # ADDED: Sampling parameters
        self.num_inference_steps = 50
        self.guidance_scale_text = 7.5
        self.guidance_scale_image = 1.5
        self.seed = seed
    
    def run_zero_shot_baseline(self):
        """Run 0: Apply pretrained model without fine-tuning."""
        logger.info("Running zero-shot baseline (Run 0)...")
        
        output_dir = self.output_dir / "Run_0"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # MODIFIED: Use available data (train split if small dataset)
        try:
            # Try loading test set
            dataset = CMPFacadeDataset(
                self.dataset_root, split='test', image_size=256, seed=self.seed
            )
        except (ValueError, IndexError):
            # Fallback to train set if test set empty (small synthetic dataset)
            logger.warning("Test split empty or unavailable, using train split instead")
            dataset = CMPFacadeDataset(
                self.dataset_root, split='train', image_size=256, seed=self.seed
            )
        
        # ADDED: Generate samples from available data
        num_samples = min(10, len(dataset))  # Generate 10 samples (or less if dataset smaller)
        logger.info(f"Generating {num_samples} zero-shot samples...")
        
        for idx in range(num_samples):
            try:
                sample = dataset[idx]
                image = (sample['image'] + 1) / 2  # Denormalize to [0, 1]
                conditioning_image = (sample['conditioning_image'] + 1) / 2
                
                logger.info(f"Running inference on sample {idx}...")
                
                # ADDED: Run inference
                result = self.pipeline_run0(
                    prompt=sample['prompt'],
                    image=Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)),
                    conditioning_image=Image.fromarray((conditioning_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)),
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale_text,
                    image_guidance_scale=self.guidance_scale_image,
                    generator=torch.Generator(device=self.device).manual_seed(self.seed),
                )
                
                # ADDED: Save output
                output_path = output_dir / f"sample_{idx:03d}.png"
                result.images[0].save(output_path)
                logger.info(f"✓ Saved Run 0 sample {idx} to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
        
        logger.info(f"✓ Run 0 outputs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Noise Schedule Ablation in Diffusion Models")
    parser.add_argument('--run', type=str, choices=['0', 'A', 'B'], required=True,
                       help="Which run to execute: 0 (baseline), A (native schedule), B (ablated schedule)")
    parser.add_argument('--dataset_root', type=str, default='./data/cmp_facade',
                       help="Path to CMP Facade dataset")
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help="Output directory for results")
    parser.add_argument('--model_id', type=str, default='timbrooks/instruct-pix2pix',
                       help="Hugging Face model ID")
    parser.add_argument('--epochs', type=int, default=20, help="Number of fine-tuning epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--max_steps', type=int, default=None, 
                       help="QUICK VALIDATION: Maximum training steps (overrides epochs). "
                            "E.g., --max_steps 10 trains for 10 batches (~20-50 min instead of hours)")
    
    args = parser.parse_args()
    
    if args.run == '0':
        # ADDED: Run 0 - Zero-shot baseline
        logger.info("="*60)
        logger.info("RUN 0: PRE-FINE-TUNING BASELINE (ZERO-SHOT)")
        logger.info("="*60)
        runner = InferenceRunner(
            model_id=args.model_id,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        runner.run_zero_shot_baseline()
    
    elif args.run == 'A':
        # ADDED: Run A - Fine-tuning with native schedule
        logger.info("="*60)
        logger.info("RUN A: FINE-TUNING WITH NATIVE SCHEDULE")
        logger.info("="*60)
        finetuner = InstructPix2PixFineTuner(
            model_id=args.model_id,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            run_name='Run_A',
            use_ablated_schedule=False,
            seed=args.seed,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
        )
        finetuner.finetune()
    
    elif args.run == 'B':
        # MODIFIED: Run B - Fine-tuning with ablated schedule
        logger.info("="*60)
        logger.info("RUN B: FINE-TUNING WITH ABLATED SCHEDULE (x=2.20)")
        logger.info("="*60)
        finetuner = InstructPix2PixFineTuner(
            model_id=args.model_id,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            run_name='Run_B',
            use_ablated_schedule=True,
            seed=args.seed,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
        )
        finetuner.finetune()


if __name__ == '__main__':
    main()
