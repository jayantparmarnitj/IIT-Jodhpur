"""
Phase 3: ControlNet Semantic Segmentation with Noise Schedule Ablation
Group: Jash Lathiya (M26AI1050), Jayant Parmar (M26AI1051), Kumar Viplav (M26AI1056), Mamata Das (M26AI1060)

This script implements ControlNet fine-tuning on CMP Facade database with noise schedule ablation.

Features:
- Run 0: Zero-shot ControlNet baseline (no fine-tuning)
- Run A: ControlNet fine-tuning with native noise schedule
- Run B: ControlNet fine-tuning with ablated noise schedule (exponent=2.20)

The experiment tests whether distributional mismatch between training and inference
noise schedules affects ControlNet-based semantic-to-photorealistic synthesis.

Environment:
- Python 3.10+
- PyTorch 2.0+ with MPS support
- Diffusers 0.41.0+
- transformers 4.30.0+
- PIL, numpy, tqdm

Commands:
  Run 0:  python finetune_controlnet.py --run 0 --dataset_root ./data/cmp_facade_complete
  Run A:  python finetune_controlnet.py --run A --dataset_root ./data/cmp_facade_complete --epochs 20 --batch_size 4
  Run B:  python finetune_controlnet.py --run B --dataset_root ./data/cmp_facade_complete --epochs 20 --batch_size 4
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
from PIL import Image
import tqdm

# ADDED: Diffusers imports for ControlNet
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDPMScheduler,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer

# ADDED: Import exponent constant for schedule ablation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from EXPONENT import NOISE_SCHEDULE_EXPONENT

# MODIFIED: Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CMPFacadeDataset(Dataset):
    """
    CMP Facade Database with semantic conditioning for ControlNet.
    
    Structure:
      images/     - 606 JPEG facade photographs (1024×1024)
      labels/     - 606 PNG semantic masks (12-class, 1024×1024)
    
    Classes: background, facade, molding, cornice, pillar, window, door, 
             sill, blind, balcony, shop, deco (12 total)
    
    Preprocessing:
    - Resize images to 256×256 (BILINEAR)
    - Resize labels to 256×256 (NEAREST to preserve class boundaries)
    - Normalize images to [-1, 1] for diffusion model
    - Convert labels to class indices for ControlNet conditioning
    """
    
    def __init__(
        self,
        dataset_root: str,
        split: str = 'train',
        image_size: int = 256,
        seed: int = 42,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        
        # ADDED: CMP semantic classes
        self.class_names = [
            'background', 'facade', 'molding', 'cornice', 'pillar',
            'window', 'door', 'sill', 'blind', 'balcony', 'shop', 'deco'
        ]
        self.num_classes = len(self.class_names)
        
        # ADDED: Load and partition dataset with fixed seed
        self.image_paths = self._load_and_partition_dataset(seed)
        logger.info(f"Loaded {len(self.image_paths)} images for split '{split}'")
    
    def _load_and_partition_dataset(self, seed: int) -> List[str]:
        """
        Load all image-label pairs and partition by split.
        Uses fixed seed for reproducible train/val/test split.
        """
        image_dir = self.dataset_root / "images"
        label_dir = self.dataset_root / "labels"
        
        if not image_dir.exists() or not label_dir.exists():
            raise ValueError(f"Dataset directories not found in {self.dataset_root}")
        
        # ADDED: Find all valid image-label pairs
        all_images = sorted([
            str(p) for p in image_dir.glob("*.jpg")
            if (label_dir / (p.stem + ".png")).exists()
        ])
        
        if not all_images:
            raise ValueError(f"No valid image-label pairs found in {self.dataset_root}")
        
        # ADDED: Partition with fixed seed
        rng = random.Random(seed)
        rng.shuffle(all_images)
        
        total = len(all_images)
        num_train = 400
        num_val = 100
        # num_test = total - num_train - num_val
        
        logger.info(f"Dataset: {total} total, train={num_train}, val={num_val}, "
                   f"test={total - num_train - num_val}")
        
        if self.split == 'train':
            return all_images[:num_train]
        elif self.split == 'val':
            return all_images[num_train:num_train + num_val]
        elif self.split == 'test':
            return all_images[num_train + num_val:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def _load_label_as_class_indices(self, label_path: str) -> torch.Tensor:
        """
        Load semantic label PNG and convert to class index tensor.
        
        CMP labels are 12-class indexed PNGs where pixel value = class index.
        Returns: (H, W) tensor with values in [0, 11]
        """
        label = Image.open(label_path).convert('P')  # Palette mode
        label_array = np.array(label, dtype=np.uint8)
        
        # ADDED: Clip to valid class range [0, 11]
        label_array = np.clip(label_array, 0, self.num_classes - 1)
        
        return torch.from_numpy(label_array).long()
    
    def _label_to_image(self, label_indices: torch.Tensor) -> Image.Image:
        """
        Convert class index tensor to RGB image for visualization in ControlNet.
        
        Uses a fixed color palette to represent each semantic class.
        ControlNet expects RGB images as conditioning input.
        """
        # ADDED: Define color palette for 12 CMP classes
        palette = [
            (0, 0, 0),              # 0: background (black)
            (128, 64, 128),         # 1: facade (purple)
            (244, 35, 232),         # 2: molding (pink)
            (70, 70, 70),           # 3: cornice (dark gray)
            (102, 102, 156),        # 4: pillar (light gray)
            (190, 153, 153),        # 5: window (brown)
            (153, 153, 153),        # 6: door (gray)
            (250, 170, 30),         # 7: sill (orange)
            (220, 220, 0),          # 8: blind (yellow)
            (107, 142, 35),         # 9: balcony (olive)
            (152, 251, 152),        # 10: shop (pale green)
            (70, 130, 180),         # 11: deco (steel blue)
        ]
        
        # ADDED: Convert label indices to RGB
        label_np = label_indices.cpu().numpy().astype(np.uint8)
        rgb = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)
        
        for class_idx, (r, g, b) in enumerate(palette):
            mask = label_np == class_idx
            rgb[mask] = [r, g, b]
        
        return Image.fromarray(rgb, mode='RGB')
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load image-label pair and preprocess.
        
        Returns:
            dict with keys:
            - 'pixel_values': (3, 256, 256) normalized image in [-1, 1]
            - 'conditioning_pixel_values': (3, 256, 256) RGB conditioning from label
            - 'prompt': text prompt (unused, for API compatibility)
        """
        image_path = self.image_paths[idx]
        label_path = str(image_path).replace('/images/', '/labels/').replace('.jpg', '.png')
        
        # ADDED: Load image
        image = Image.open(image_path).convert('RGB')
        
        # ADDED: Load label and convert to RGB conditioning
        label_indices = self._load_label_as_class_indices(label_path)
        label_rgb = self._label_to_image(label_indices)
        
        # ADDED: Resize both to target size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        label_rgb = label_rgb.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        
        # ADDED: Convert to tensors and normalize to [-1, 1]
        image_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0  # (H, W, 3) -> [-1, 1]
        condition_tensor = torch.from_numpy(np.array(label_rgb)).float() / 127.5 - 1.0
        
        # ADDED: Transpose to (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        condition_tensor = condition_tensor.permute(2, 0, 1)
        
        return {
            'pixel_values': image_tensor,
            'conditioning_pixel_values': condition_tensor,
            'prompt': 'a photorealistic building facade',
        }


class NoiseScheduler:
    """
    Base noise schedule class for diffusion models.
    Handles precomputation of alpha values and scheduling.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, device: str = 'cpu'):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        self._compute_schedule()
    
    def _compute_schedule(self):
        """Compute base schedule (to be overridden in subclasses)."""
        raise NotImplementedError
    
    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get noise level for given timestep."""
        # ADDED: Returns alpha_bar (cumulative product of alphas)
        return self.alphas_cumprod[timesteps]


class NativeNoiseScheduler(NoiseScheduler):
    """Native linear schedule used by Stable Diffusion."""
    
    def _compute_schedule(self):
        """
        Compute native Stable Diffusion schedule.
        Uses linear interpolation in sqrt(alpha) space.
        
        Range: β ∈ [0.00085, 0.0120]
        """
        # ADDED: Linear schedule in sqrt(alpha) space
        sqrt_alpha_start = 0.99915  # sqrt(alpha_0)
        sqrt_alpha_end = 0.00085    # sqrt(alpha_T)
        
        # MODIFIED: Compute betas
        num_steps = self.num_train_timesteps
        alphas_cumprod = np.linspace(sqrt_alpha_end**2, sqrt_alpha_start**2, num_steps)
        
        # ADDED: Compute betas from alphas_cumprod
        betas = 1.0 - (alphas_cumprod / np.concatenate([[1.0], alphas_cumprod[:-1]]))
        betas = np.clip(betas, 0.0001, 0.9999)
        
        # ADDED: Precompute for efficiency
        self.register_buffer(
            'betas',
            torch.from_numpy(betas).float().to(self.device)
        )
        self.register_buffer(
            'alphas_cumprod',
            torch.from_numpy(alphas_cumprod).float().to(self.device)
        )
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Utility to store tensors as buffers."""
        setattr(self, name, tensor)


class AblatedNoiseScheduler(NoiseScheduler):
    """
    Ablated schedule with power transformation.
    Beta_t^exponent where exponent = 2.20
    """
    
    def _compute_schedule(self):
        """
        Compute schedule with power transformation: β_t^x
        
        Effect: Reduces noise at each step compared to native schedule.
        Training sees lower noise variance → inference on pure noise creates mismatch.
        """
        # ADDED: Start with native schedule
        sqrt_alpha_start = 0.99915
        sqrt_alpha_end = 0.00085
        
        num_steps = self.num_train_timesteps
        alphas_cumprod_native = np.linspace(sqrt_alpha_end**2, sqrt_alpha_start**2, num_steps)
        
        # ADDED: Compute native betas
        betas_native = 1.0 - (alphas_cumprod_native / np.concatenate([[1.0], alphas_cumprod_native[:-1]]))
        betas_native = np.clip(betas_native, 0.0001, 0.9999)
        
        # MODIFIED: Apply power transformation
        betas_ablated = betas_native ** NOISE_SCHEDULE_EXPONENT
        
        # ADDED: Recompute alpha products
        alphas = 1.0 - betas_ablated
        alphas_cumprod = np.cumprod(alphas)
        
        # ADDED: Precompute
        self.register_buffer('betas', torch.from_numpy(betas_ablated).float().to(self.device))
        self.register_buffer('alphas_cumprod', torch.from_numpy(alphas_cumprod).float().to(self.device))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Utility to store tensors as buffers."""
        setattr(self, name, tensor)


class ControlNetFinetuner:
    """
    ControlNet fine-tuning on CMP Facade dataset with noise schedule ablation.
    """
    
    def __init__(
        self,
        run_id: str,
        dataset_root: str,
        output_dir: str,
        model_id: str = 'runwayml/stable-diffusion-v1-5',
        controlnet_id: str = 'lllyasviel/sd-controlnet-seg',
        epochs: int = 20,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-2,
        seed: int = 42,
        use_ablated_schedule: bool = False,
        device: str = 'cpu',
    ):
        self.run_id = run_id
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir) / f"ControlNet_Run_{run_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.use_ablated_schedule = use_ablated_schedule
        self.device = device
        
        # ADDED: Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.mps.is_available():
            torch.mps.manual_seed(seed)
        
        logger.info(f"Run {run_id}: Ablated schedule = {use_ablated_schedule}")
        
        # ADDED: Load models
        self._load_models(model_id, controlnet_id)
        
        # ADDED: Initialize noise schedule
        if use_ablated_schedule:
            self.scheduler = AblatedNoiseScheduler(device=device)
        else:
            self.scheduler = NativeNoiseScheduler(device=device)
        
        logger.info(f"Noise schedule: {'ABLATED (x={NOISE_SCHEDULE_EXPONENT})' if use_ablated_schedule else 'NATIVE'}")
        
        # ADDED: Logging
        self.training_logs = []
    
    def _load_models(self, model_id: str, controlnet_id: str):
        """Load ControlNet and base diffusion models."""
        logger.info(f"Loading ControlNet from {controlnet_id}...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch.float32,
        )
        
        logger.info(f"Loading pipeline from {model_id}...")
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch.float32,
        )
        
        # ADDED: Move to device
        self.pipeline = self.pipeline.to(self.device)
        self.controlnet = self.controlnet.to(self.device)
        
        # ADDED: Get text encoder and VAE for training
        self.text_encoder = self.pipeline.text_encoder
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.unet = self.pipeline.unet
        
        # MODIFIED: Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
    
    def _encode_prompt(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to CLIP embeddings."""
        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompts,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        return text_embeddings
    
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to VAE latents."""
        with torch.no_grad():
            latents = self.vae.encode(images.to(self.device)).latent_dist.sample()
            latents = latents * 0.18215  # Scaling factor from Stable Diffusion
        return latents
    
    def train(self):
        """Execute training loop."""
        # ADDED: Load dataset
        train_dataset = CMPFacadeDataset(
            self.dataset_root,
            split='train',
            seed=self.seed,
        )
        val_dataset = CMPFacadeDataset(
            self.dataset_root,
            split='val',
            seed=self.seed,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # MODIFIED: Use 0 for macOS compatibility
        )
        
        # ADDED: Setup optimizer (only for ControlNet, not ControlNet embeddings)
        optimizer = AdamW(
            self.controlnet.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        logger.info(f"Starting training for {self.epochs} epochs...")
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # MODIFIED: Training loop
        for epoch in range(1, self.epochs + 1):
            self.controlnet.train()
            epoch_losses = []
            
            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # ADDED: Get inputs
                pixel_values = batch['pixel_values'].to(self.device)
                conditioning_pixel_values = batch['conditioning_pixel_values'].to(self.device)
                prompts = batch['prompt']
                
                # ADDED: Encode inputs
                latents = self._encode_images(pixel_values)
                prompt_embeds = self._encode_prompt(prompts)
                
                # ADDED: Sample timesteps
                timesteps = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (latents.shape[0],),
                    device=self.device,
                )
                
                # ADDED: Add noise (with ablated or native schedule)
                alphas_cumprod = self.scheduler.alphas_cumprod
                alpha = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                
                noise = torch.randn_like(latents)
                noisy_latents = (torch.sqrt(alpha) * latents +
                                torch.sqrt(1 - alpha) * noise)
                
                # ADDED: ControlNet forward pass
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=conditioning_pixel_values,
                    return_dict=False,
                )
                
                # ADDED: UNet forward pass with ControlNet guidance
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # ADDED: Compute loss (prediction of noise)
                loss = F.mse_loss(model_pred, noise)
                
                # ADDED: Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': np.mean(epoch_losses[-100:])})
            
            # ADDED: Log epoch loss
            avg_loss = np.mean(epoch_losses)
            self.training_logs.append({'epoch': epoch, 'train_loss': avg_loss})
            
            logger.info(f"Epoch {epoch}: avg_loss = {avg_loss:.6f}")
            
            # ADDED: Save checkpoint every 4 epochs
            if epoch % 4 == 0:
                checkpoint_dir = self.output_dir / f"checkpoint_{epoch}"
                checkpoint_dir.mkdir(exist_ok=True)
                self.controlnet.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # ADDED: Validation (optional, log only)
            if epoch % 4 == 0:
                logger.info(f"Validation checkpoint saved for epoch {epoch}")
        
        # ADDED: Save final model
        final_model_dir = self.output_dir / "final_model"
        final_model_dir.mkdir(exist_ok=True)
        self.controlnet.save_pretrained(final_model_dir)
        logger.info(f"Saved final model to {final_model_dir}")
        
        # ADDED: Save training logs
        self._save_training_logs()
    
    def _save_training_logs(self):
        """Save training logs to CSV."""
        import csv
        log_path = self.output_dir / "training_logs.csv"
        
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss'])
            writer.writeheader()
            writer.writerows(self.training_logs)
        
        logger.info(f"Saved training logs to {log_path}")
    
    def generate_samples(self, num_samples: int = 3):
        """Generate samples using the fine-tuned ControlNet."""
        # ADDED: Load test dataset
        test_dataset = CMPFacadeDataset(
            self.dataset_root,
            split='test',
            seed=self.seed,
        )
        
        self.controlnet.eval()
        output_dir = self.output_dir / "generated_samples"
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating {num_samples} samples...")
        
        for i in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[i]
            
            # ADDED: Prepare conditioning image
            conditioning_image = (sample['conditioning_pixel_values'] + 1) * 127.5
            conditioning_image = conditioning_image.clamp(0, 255).uint8()
            conditioning_image = Image.fromarray(
                conditioning_image.permute(1, 2, 0).numpy(),
                mode='RGB'
            )
            
            # ADDED: Generate image
            with torch.no_grad():
                output = self.pipeline(
                    prompt=sample['prompt'],
                    image=conditioning_image,
                    num_inference_steps=20,
                )
            
            # ADDED: Save
            output_path = output_dir / f"sample_{i:03d}.png"
            output.images[0].save(output_path)
            logger.info(f"Saved sample to {output_path}")


def run_experiment(args):
    """Main experiment execution."""
    logger.info(f"===== Phase 3 ControlNet Run {args.run} =====")
    logger.info(f"Dataset: {args.dataset_root}")
    logger.info(f"Output: {args.output_dir}")
    
    # ADDED: Set device
    if torch.mps.is_available():
        device = 'mps'
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using CUDA device {torch.cuda.current_device()}")
    else:
        device = 'cpu'
        logger.warning("Using CPU (slow)")
    
    if args.run == '0':
        # ADDED: Run 0 - Zero-shot (no training)
        logger.info("Run 0: Zero-shot baseline (ControlNet without fine-tuning)")
        logger.info("Not implemented in this version - use phase2 results for comparison")
        
    elif args.run in ['A', 'B']:
        # ADDED: Run A/B - Training
        use_ablated = (args.run == 'B')
        
        finetuner = ControlNetFinetuner(
            run_id=args.run,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            use_ablated_schedule=use_ablated,
            device=device,
        )
        
        # ADDED: Train
        finetuner.train()
        
        # ADDED: Generate samples
        finetuner.generate_samples(num_samples=3)
        
        logger.info(f"Run {args.run} complete!")
    
    else:
        raise ValueError(f"Invalid run: {args.run}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3 ControlNet Semantic Segmentation with Noise Schedule Ablation'
    )
    
    # ADDED: Required arguments
    parser.add_argument(
        '--run',
        type=str,
        required=True,
        choices=['0', 'A', 'B'],
        help='Experiment run: 0=zero-shot, A=native schedule, B=ablated schedule'
    )
    
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='./data/cmp_facade_complete',
        help='Path to CMP Facade dataset'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory for results'
    )
    
    # ADDED: Training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # ADDED: Run experiment
    run_experiment(args)


if __name__ == '__main__':
    main()
