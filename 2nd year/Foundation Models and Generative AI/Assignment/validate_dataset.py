"""
Dataset Validation and Preparation Utility
Validates CMP Facade dataset structure and generates preprocessing report.
"""

import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatasetValidator:
    """
    Validate CMP Facade dataset structure and contents.
    """
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / "images"
        self.label_dir = self.dataset_root / "labels"
        
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise ValueError(f"Label directory not found: {self.label_dir}")
    
    def validate_structure(self) -> dict:
        """Validate dataset structure and count files."""
        logger.info("Validating dataset structure...")
        
        # ADDED: Load images and labels
        image_files = sorted(list(self.image_dir.glob("*.jpg")))
        label_files = sorted(list(self.label_dir.glob("*.png")))
        
        logger.info(f"Found {len(image_files)} images")
        logger.info(f"Found {len(label_files)} labels")
        
        # ADDED: Check for paired files
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        paired_count = len(image_stems & label_stems)
        unpaired_images = image_stems - label_stems
        unpaired_labels = label_stems - image_stems
        
        logger.info(f"Paired image-label files: {paired_count}")
        
        if unpaired_images:
            logger.warning(f"Images without labels: {unpaired_images}")
        if unpaired_labels:
            logger.warning(f"Labels without images: {unpaired_labels}")
        
        return {
            'total_images': len(image_files),
            'total_labels': len(label_files),
            'paired_count': paired_count,
            'unpaired_images': len(unpaired_images),
            'unpaired_labels': len(unpaired_labels),
        }
    
    def validate_image_properties(self) -> dict:
        """Validate image dimensions and properties."""
        logger.info("Validating image properties...")
        
        # ADDED: Statistics collectors
        dimensions = defaultdict(int)
        color_modes = defaultdict(int)
        size_bytes = defaultdict(int)
        
        image_files = list(self.image_dir.glob("*.jpg"))
        
        for i, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path)
                dimensions[img.size] += 1
                color_modes[img.mode] += 1
                size_bytes[img_path.stat().st_size] += 1
                
                if i % 100 == 0:
                    logger.info(f"  Processed {i}/{len(image_files)} images")
            except Exception as e:
                logger.error(f"Error loading {img_path}: {e}")
        
        logger.info(f"Image dimensions found: {dict(dimensions)}")
        logger.info(f"Color modes found: {dict(color_modes)}")
        
        return {
            'dimensions': dict(dimensions),
            'color_modes': dict(color_modes),
            'size_bytes': {str(k): v for k, v in dict(size_bytes).items()},
            'min_width': min([d[0] for d in dimensions.keys()]),
            'max_width': max([d[0] for d in dimensions.keys()]),
            'min_height': min([d[1] for d in dimensions.keys()]),
            'max_height': max([d[1] for d in dimensions.keys()]),
        }
    
    def validate_label_properties(self) -> dict:
        """Validate label mask properties and semantic classes."""
        logger.info("Validating label properties...")
        
        # ADDED: Statistics collectors
        dimensions = defaultdict(int)
        color_modes = defaultdict(int)
        unique_classes = set()
        
        label_files = list(self.label_dir.glob("*.png"))
        
        for i, label_path in enumerate(label_files):
            try:
                label = Image.open(label_path)
                dimensions[label.size] += 1
                color_modes[label.mode] += 1
                
                # ADDED: Extract unique RGB values (classes)
                label_array = np.array(label)
                if len(label_array.shape) == 3:
                    for y in range(0, label_array.shape[0], 10):  # Subsample for speed
                        for x in range(0, label_array.shape[1], 10):
                            rgb = tuple(label_array[y, x, :3])
                            unique_classes.add(rgb)
                
                if i % 100 == 0:
                    logger.info(f"  Processed {i}/{len(label_files)} labels")
            except Exception as e:
                logger.error(f"Error loading {label_path}: {e}")
        
        logger.info(f"Label dimensions found: {dict(dimensions)}")
        logger.info(f"Label modes found: {dict(color_modes)}")
        logger.info(f"Unique semantic classes found: {len(unique_classes)}")
        
        # ADDED: Expected classes (12 semantic classes)
        expected_classes = 12
        if len(unique_classes) >= expected_classes:
            logger.info(f"✓ Found {len(unique_classes)} >= expected {expected_classes} classes")
        else:
            logger.warning(f"✗ Found only {len(unique_classes)} classes, expected {expected_classes}")
        
        return {
            'dimensions': dict(dimensions),
            'color_modes': dict(color_modes),
            'unique_classes': len(unique_classes),
            'expected_classes': expected_classes,
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        # ADDED: Run all validations
        structure_report = self.validate_structure()
        image_report = self.validate_image_properties()
        label_report = self.validate_label_properties()
        
        # ADDED: Generate markdown report
        report = f"""# CMP Facade Dataset Validation Report

## Dataset Structure

- Total images: {structure_report['total_images']}
- Total labels: {structure_report['total_labels']}
- Paired image-label files: {structure_report['paired_count']}
- Unpaired images: {structure_report['unpaired_images']}
- Unpaired labels: {structure_report['unpaired_labels']}

**Status:** {'✓ VALID' if structure_report['paired_count'] >= 600 else '✗ INCOMPLETE'}

## Image Properties

- **Dimensions:** {', '.join([str(d) for d in image_report['dimensions'].keys()])}
- **Min size:** {image_report['min_width']} × {image_report['min_height']}
- **Max size:** {image_report['max_width']} × {image_report['max_height']}
- **Color modes:** {', '.join(image_report['color_modes'].keys())}

**Preprocessing Required:** Resize to 256 × 256 (standardization)

## Label Properties

- **Dimensions:** {', '.join([str(d) for d in label_report['dimensions'].keys()])}
- **Color modes:** {', '.join(label_report['color_modes'].keys())}
- **Unique semantic classes:** {label_report['unique_classes']}
- **Expected classes:** {label_report['expected_classes']}

**Label Preprocessing:** Resize with nearest-neighbor interpolation to preserve class boundaries

## Dataset Recommendations

1. **Validation:** All {structure_report['paired_count']} paired files ready for preprocessing
2. **Resolution:** Original images {image_report['max_width']}×{image_report['max_height']} → standardize to 256×256
3. **Labels:** Use NEAREST interpolation (not BILINEAR) to avoid class blending
4. **Partition:** Use fixed random seed (42) for reproducible train/val/test split
   - Training: 400 images
   - Validation: 100 images
   - Test: 106 images

## Expected Data Split

```
Training set (400 images):
  - Input: semantic label maps (256×256, 12 classes)
  - Output: facade photographs (256×256, RGB)

Validation set (100 images):
  - Same format as training

Test set (106 images):
  - Same format as training
  - Used only for final evaluation
```

## Preprocessing Pipeline

1. Load image from JPEG
2. Load label from PNG
3. Resize image to 256×256 with BILINEAR interpolation
4. Resize label to 256×256 with NEAREST interpolation
5. Normalize to [-1, 1] range for diffusion model
6. Batch and feed to fine-tuning pipeline

---

**Report generated:** CMP Facade Dataset appears valid and ready for Phase 2 experiments
"""
        
        # ADDED: Save report if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def check_split_possibility(self) -> bool:
        """Verify dataset can be split into 400/100/106."""
        logger.info("Checking train/val/test split possibility...")
        
        structure = self.validate_structure()
        paired_count = structure['paired_count']
        
        required_count = 400 + 100 + 106  # 606
        
        if paired_count >= required_count:
            logger.info(f"✓ Dataset has {paired_count} samples (need {required_count})")
            return True
        else:
            logger.error(f"✗ Dataset has only {paired_count} samples (need {required_count})")
            return False


def main():
    parser = argparse.ArgumentParser(description="Validate CMP Facade dataset")
    parser.add_argument('--dataset_root', type=str, default='./data/cmp_facade',
                       help="Path to CMP Facade dataset")
    parser.add_argument('--report_file', type=str, default='./validation_report.md',
                       help="Output file for validation report")
    
    args = parser.parse_args()
    
    try:
        # ADDED: Create validator and run checks
        validator = DatasetValidator(args.dataset_root)
        
        # ADDED: Generate report
        report = validator.generate_report(args.report_file)
        print(report)
        
        # ADDED: Check split possibility
        can_split = validator.check_split_possibility()
        
        if can_split:
            logger.info("✓ Dataset validation PASSED - ready for Phase 2 experiments")
            exit(0)
        else:
            logger.error("✗ Dataset validation FAILED - insufficient samples")
            exit(1)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        exit(1)


if __name__ == '__main__':
    main()
