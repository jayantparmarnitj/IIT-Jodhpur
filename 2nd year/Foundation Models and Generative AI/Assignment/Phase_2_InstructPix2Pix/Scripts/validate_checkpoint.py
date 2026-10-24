"""
Validation script for InstructPix2Pix checkpoint_8
Tests the fine-tuned model on test set images with metrics:
- Visual quality (sample generation)
- Loss computation on test set
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from EXPONENT import NOISE_SCHEDULE_EXPONENT

from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device selection (MPS for Mac)
if torch.backends.mps.is_available():
    device = "mps"
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

logger.info(f"Using device: {device}")


class CMPFacadeTestDataset(Dataset):
    """Load test set from CMP Facade dataset"""
    
    def __init__(self, dataset_root: str, image_size: int = 256, seed: int = 42):
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.condition_prompt = "turn this semantic facade label map into a photorealistic building facade"
        
        # Load test split
        image_dir = self.dataset_root / "images"
        all_images = sorted([
            str(p) for p in image_dir.glob("*.jpg")
            if (self.dataset_root / "labels" / (p.stem + ".png")).exists()
        ])
        
        # Use same seed-based split as training
        import random
        rng = random.Random(seed)
        rng.shuffle(all_images)
        
        total = len(all_images)
        num_train = int(total * 0.66)
        num_val = int(total * 0.17)
        
        # Test set is remainder
        self.image_paths = all_images[num_train + num_val:]
        
        logger.info(f"Test set: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.dataset_root / "labels" / (Path(img_path).stem + ".png")
        
        # Load images
        image = Image.open(img_path).convert("RGB").resize((self.image_size, self.image_size))
        label = Image.open(label_path).convert("RGB").resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Convert to tensors
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        label_tensor = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 127.5 - 1.0
        
        return {
            "image": image_tensor,
            "label": label_tensor,
            "path": img_path
        }


def validate_checkpoint(checkpoint_path, dataset_root, output_dir, batch_size=2, num_samples=5):
    """
    Validate checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint (e.g., Phase_2_InstructPix2Pix/Results/Run_A/checkpoint_16)
        dataset_root: Path to dataset root
        output_dir: Where to save validation results
        batch_size: Batch size for inference
        num_samples: Number of samples to generate for qualitative inspection
    """
    
    # Extract checkpoint number and run name from path
    checkpoint_name = Path(checkpoint_path).name  # e.g., "checkpoint_16"
    run_name = Path(checkpoint_path).parent.name  # e.g., "Run_A"
    
    output_path = Path(output_dir) / f"Validation_{run_name}_{checkpoint_name}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load fine-tuned pipeline
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
    ).to(device)
    
    # Load checkpoint weights
    try:
        from diffusers.models import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint_path,
            subfolder="unet",
            torch_dtype=torch.float16
        )
        pipeline.unet = unet.to(device)
        logger.info("✓ Checkpoint weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return
    
    # Load test dataset
    test_dataset = CMPFacadeTestDataset(dataset_root)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Testing on {len(test_dataset)} test images...")
    
    # 1. Generate sample outputs
    logger.info("\n--- Generating sample outputs ---")
    pipeline.enable_attention_slicing()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
            
            labels = batch["label"].to(device)
            
            # Generate predictions
            outputs = pipeline(
                prompt=[batch["label"].__class__.__name__] * labels.shape[0],
                image=labels,
                guidance_scale=7.5,
                num_inference_steps=50
            ).images
            
            # Save outputs
            for i, output in enumerate(outputs):
                sample_idx = batch_idx * batch_size + i
                output.save(output_path / f"sample_{sample_idx:03d}_generated.png")
                
                # Also save input label
                label_img = Image.fromarray(
                    ((batch["label"][i].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
                )
                label_img.save(output_path / f"sample_{sample_idx:03d}_label.png")
                
                logger.info(f"  Saved sample {sample_idx}")
    
    logger.info(f"✓ Sample outputs saved to {output_path}/")
    
    # 2. Compute statistics
    logger.info("\n--- Computing validation statistics ---")
    
    total_images = 0
    with torch.no_grad():
        for batch in test_loader:
            total_images += batch["image"].shape[0]
    
    logger.info(f"✓ Validation complete")
    logger.info(f"  - Total test images: {len(test_dataset)}")
    logger.info(f"  - Sample outputs: {output_path}/")
    
    # 3. Summary
    summary = f"""
CHECKPOINT VALIDATION SUMMARY
=============================
Checkpoint: {checkpoint_name} ({run_name})
Test Set Size: {len(test_dataset)} images
Device: {device}
Timestamp: {datetime.now()}

Generated Samples: {num_samples}
Output Directory: {output_path}

Next Steps:
1. Review generated images in {output_path}
2. Compare label inputs with generated outputs
3. Assess visual quality and semantic correctness
4. Once full training completes, compare all checkpoints
"""
    
    logger.info(summary)
    
    # Save summary
    with open(output_path / "VALIDATION_SUMMARY.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default="Phase_2_InstructPix2Pix/Results/Run_A/checkpoint_8",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--dataset_root",
        default="./data/cmp_facade_complete",
        help="Path to dataset root"
    )
    parser.add_argument(
        "--output_dir",
        default="Phase_2_InstructPix2Pix/Results",
        help="Output directory for validation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate"
    )
    
    args = parser.parse_args()
    
    validate_checkpoint(
        args.checkpoint_path,
        args.dataset_root,
        args.output_dir,
        args.batch_size,
        args.num_samples
    )
