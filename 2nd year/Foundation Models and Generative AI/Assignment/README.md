# Phase 2: Noise Schedule Ablation in Diffusion Models

**Course:** Foundation Models and Generative AI  
**Group:** Jash Lathiya (M26AI1050), Jayant Parmar (M26AI1051), Kumar Viplav (M26AI1056), Mamata Das (M26AI1060)  
**Task:** Semantic Facade Label Map to Photorealistic Building Facade Synthesis

## Overview

This implementation executes three experimental runs to test the hypothesis that fine-tuning a pretrained image-to-image diffusion model (InstructPix2Pix) under a power-transformed noise schedule creates distributional mismatch during inference, degrading output quality.

**Key Experiment:**
- **Run 0:** Zero-shot baseline (pretrained InstructPix2Pix on unseen label-map modality)
- **Run A:** Fine-tuning with native schedule {β_t}
- **Run B:** Fine-tuning with ablated schedule {β_t^2.20} where 2.20 ∈ (1.5, 3.0)

## Environment Setup

### System Requirements

- **Python:** 3.10+
- **CUDA:** 11.8+ (strongly recommended for GPU acceleration)
- **Memory:** 16GB+ GPU VRAM (12GB minimum, may cause OOM on smaller devices)

### Installation

```bash
# Clone InstructPix2Pix official repository (optional, for reference)
git clone https://github.com/timothybrooks/instruct-pix2pix.git

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers Pillow numpy tqdm scikit-image scipy

# Optional: Install lpips for perceptual metrics
pip install lpips

# Optional: For evaluation metrics
pip install opencv-python scikit-metrics
```

### Software Versions

| Package | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.10, 3.11 |
| PyTorch | 2.0+ | With CUDA 11.8 support |
| diffusers | 0.21.0+ | Stable Diffusion pipeline |
| transformers | 4.30.0+ | CLIP text encoder |
| Pillow | 10.0+ | Image processing |
| numpy | 1.24+ | Numerical computation |
| tqdm | 4.65+ | Progress bars |
| scikit-image | 0.21+ | SSIM metric |
| scipy | 1.11+ | Scientific computing |

## Dataset Setup

### Download CMP Facade Database

```bash
# Create data directory
mkdir -p data/cmp_facade/{images,labels}

# Download from official source
cd data/cmp_facade
wget https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_Facade_DB_base_new.zip
wget https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_Facade_DB_extended.zip

# Extract
unzip CMP_Facade_DB_base_new.zip
unzip CMP_Facade_DB_extended.zip

# Organize: move all .jpg and .png files to appropriate directories
# Final structure should be:
# data/cmp_facade/
#   images/
#     00000001_0.8_1.0_0.jpg
#     00000002_0.8_1.0_0.jpg
#     ... (606 total)
#   labels/
#     00000001_0.8_1.0_0.png
#     00000002_0.8_1.0_0.png
#     ... (606 total, 12-class semantic masks)
```

**Dataset Statistics:**
- Total images: 606 (378 base, 228 extended)
- Train/Val/Test split: 400/100/106 (fixed seed 42)
- Image size: variable → standardized to 256×256
- Semantic classes: 12 (facade, molding, cornice, pillar, window, door, sill, blind, balcony, shop, deco, background)

## Execution Commands

### Run 0: Zero-Shot Baseline

Apply pretrained InstructPix2Pix directly to CMP Facade test inputs without fine-tuning.

```bash
python finetune_instructpix2pix.py \
  --run 0 \
  --dataset_root ./data/cmp_facade \
  --output_dir ./outputs \
  --model_id timbrooks/instruct-pix2pix \
  --seed 42
```

**Output:**
- Directory: `./outputs/Run_0/`
- Files: `sample_000.png`, `sample_001.png`, `sample_002.png` (3 generated samples)
- Expected: Poor quality (validates out-of-distribution nature)

**Expected Runtime:** ~2-3 minutes (GPU)

### Run A: Fine-Tuning with Native Schedule

Fine-tune full UNet on CMP Facade using native noise schedule {β_t}.

```bash
python finetune_instructpix2pix.py \
  --run A \
  --dataset_root ./data/cmp_facade \
  --output_dir ./outputs \
  --model_id timbrooks/instruct-pix2pix \
  --epochs 20 \
  --batch_size 4 \
  --seed 42
```

**Configuration:**
- Optimizer: AdamW
- Learning rate: 1e-5
- Weight decay: 1e-2
- Batch size: 4
- Steps per epoch: 100
- Total steps: 2000
- Checkpoint interval: 4 epochs (5 checkpoints saved)
- FP16 mixed precision enabled

**Output:**
- Directory: `./outputs/Run_A/`
- Subdirectories:
  - `checkpoint_4/`, `checkpoint_8/`, `checkpoint_12/`, `checkpoint_16/`, `checkpoint_20/` (fine-tuned UNet weights)
  - `final_model/` (trained model after epoch 20)
- Files: `training_logs.csv` (epoch, train_loss)

**Expected Runtime:** ~45-60 minutes (GPU with batch size 4)

### Run B: Fine-Tuning with Ablated Schedule

Fine-tune full UNet on CMP Facade using ablated noise schedule {β_t^2.20}.

```bash
python finetune_instructpix2pix.py \
  --run B \
  --dataset_root ./data/cmp_facade \
  --output_dir ./outputs \
  --model_id timbrooks/instruct-pix2pix \
  --epochs 20 \
  --batch_size 4 \
  --seed 42
```

**Key Difference from Run A:** Uses power-transformed schedule with exponent 2.20

**Output:**
- Directory: `./outputs/Run_B/`
- Same structure as Run A
- Schedule-specific behavior:
  - Native: α_bar,1000 ≈ 0.0047 (pure noise)
  - Ablated: α_bar,1000' ≈ 0.985 (98.5% signal retention)
  - Distributional mismatch expected during inference

**Expected Runtime:** ~45-60 minutes (GPU with batch size 4)

## Generating Inference Trajectories

After fine-tuning is complete, collect denoising trajectories showing intermediate noisy samples at 5 equally spaced timesteps.

### Run A Trajectories

```bash
python inference_trajectories.py \
  --run A \
  --model_dir ./outputs/Run_A/final_model \
  --dataset_root ./data/cmp_facade \
  --output_dir ./trajectories \
  --num_samples 3 \
  --seed 42
```

**Output:**
- Directory: `./trajectories/Run_A/`
- Files per sample (for each of 5 timesteps: 0, 250, 500, 750, 1000):
  - `Run_A_t0000_sample000.npy` (float32 latent representation)
  - `Run_A_t0000_sample000.png` (RGB visualization)
  - ... (5 timesteps × 3 samples = 15 .npy and 15 .png files)

### Run B Trajectories

```bash
python inference_trajectories.py \
  --run B \
  --model_dir ./outputs/Run_B/final_model \
  --dataset_root ./data/cmp_facade \
  --output_dir ./trajectories \
  --num_samples 3 \
  --seed 42
```

**Output:** Same structure as Run A in `./trajectories/Run_B/`

**Expected Runtime:** ~5-10 minutes per run (GPU)

## Generating Report

After all runs complete, generate the comprehensive Phase 2 report.

```bash
python evaluate_and_report.py \
  --run_0_dir ./outputs/Run_0 \
  --run_a_dir ./outputs/Run_A \
  --run_b_dir ./outputs/Run_B \
  --dataset_root ./data/cmp_facade \
  --output_dir ./report \
  --exponent 2.20
```

**Output:**
- File: `./report/Report.md` (comprehensive markdown report)
- Includes:
  - Schedule analysis with comparison tables
  - Experimental setup and hyperparameters
  - Results with metrics (SSIM, LPIPS, MSE)
  - Mechanistic analysis grounded in schedule mathematics
  - Takeaways and implications

**Expected Runtime:** ~5 minutes

## Complete Reproduction Workflow

To reproduce the entire experiment from scratch:

```bash
#!/bin/bash

# Set dataset root
DATASET_ROOT="./data/cmp_facade"
OUTPUT_DIR="./outputs"
REPORT_DIR="./report"
TRAJECTORIES_DIR="./trajectories"
SEED=42

echo "Starting Phase 2 Noise Schedule Ablation Experiment..."

# Run 0: Zero-shot baseline
echo "=== Run 0: Zero-Shot Baseline ==="
python finetune_instructpix2pix.py --run 0 \
  --dataset_root $DATASET_ROOT \
  --output_dir $OUTPUT_DIR \
  --seed $SEED

# Run A: Fine-tuning with native schedule
echo "=== Run A: Fine-Tuning (Native Schedule) ==="
python finetune_instructpix2pix.py --run A \
  --dataset_root $DATASET_ROOT \
  --output_dir $OUTPUT_DIR \
  --epochs 20 \
  --batch_size 4 \
  --seed $SEED

# Run B: Fine-tuning with ablated schedule
echo "=== Run B: Fine-Tuning (Ablated Schedule x=2.20) ==="
python finetune_instructpix2pix.py --run B \
  --dataset_root $DATASET_ROOT \
  --output_dir $OUTPUT_DIR \
  --epochs 20 \
  --batch_size 4 \
  --seed $SEED

# Collect inference trajectories
echo "=== Collecting Inference Trajectories ==="
python inference_trajectories.py --run A \
  --model_dir $OUTPUT_DIR/Run_A/final_model \
  --dataset_root $DATASET_ROOT \
  --output_dir $TRAJECTORIES_DIR \
  --num_samples 3 \
  --seed $SEED

python inference_trajectories.py --run B \
  --model_dir $OUTPUT_DIR/Run_B/final_model \
  --dataset_root $DATASET_ROOT \
  --output_dir $TRAJECTORIES_DIR \
  --num_samples 3 \
  --seed $SEED

# Generate report
echo "=== Generating Report ==="
python evaluate_and_report.py \
  --run_0_dir $OUTPUT_DIR/Run_0 \
  --run_a_dir $OUTPUT_DIR/Run_A \
  --run_b_dir $OUTPUT_DIR/Run_B \
  --dataset_root $DATASET_ROOT \
  --output_dir $REPORT_DIR \
  --exponent 2.20

echo "=== Experiment Complete ==="
echo "Outputs:"
echo "  - Training logs: $OUTPUT_DIR/{Run_A,Run_B}/training_logs.csv"
echo "  - Trajectories: $TRAJECTORIES_DIR/"
echo "  - Report: $REPORT_DIR/Report.md"
```

## Noise Schedule Exponent

**Exponent Value:** 2.20 (declared in `EXPONENT.py`)

**Verification:**
- Range required: x ∈ (1.5, 3.0)
- Declared value: 2.20 ✓
- Effect: β_t^2.20 < β_t for all t ∈ (0, 1)
- β_1^2.20 ≈ 1.76 × 10^-7
- β_1000^2.20 ≈ 5.95 × 10^-5

## Key Implementation Details

### Noise Schedules

**Native Schedule (Run A):**
```python
β_t ∈ (0.00085, 0.0120) via scaled linear in √β space
α_t = 1 - β_t
ᾱ_t = ∏ α_s
```

**Ablated Schedule (Run B):**
```python
β_t^x where x = 2.20
ᾱ_t' = ∏ (1 - β_t^2.20)
```

### Distributional Mismatch Mechanism

**Forward Process:**
- Run A: Noise level increases from α_bar=0.99915 at t=1 to α_bar≈0.0047 at t=1000
- Run B: Noise level increases from α_bar=0.9999998 at t=1 to α_bar≈0.985 at t=1000

**Reverse Process (Both Runs):**
- Initialize from x_T ~ N(0, I) (pure noise, α_bar≈0)
- Run A: Familiar regime (trained to denoise from near-pure-noise)
- Run B: Acute mismatch (trained to preserve 98.5% signal, now must denoise from pure noise)

### Schedule Precomputation

All schedule quantities precomputed in float64 to prevent precision loss, then cast to float32:
```python
self.betas = betas.double()
self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
```

## Submission Components

### 1. Code (`finetune_instructpix2pix.py`, `EXPONENT.py`)
- ✓ Every modified line marked `# MODIFIED`
- ✓ Every added line marked `# ADDED`
- ✓ Exponent declared as named constant at top
- ✓ README with exact reproduction commands

### 2. Fine-Tuning Logs
- ✓ `training_logs.csv` per run: `epoch, train_loss`
- ✓ 5 equally spaced checkpoints with 3+ generated samples per checkpoint
- ✓ Clearly labeled: `Run_{A,B}_epoch{n}_sample{i}.png`

### 3. Inference Trajectories
- ✓ Final model from Run A and Run B
- ✓ 5 equally spaced timesteps: [0, 250, 500, 750, 1000]
- ✓ 3+ samples per timestep
- ✓ Labeled: `Run_{A,B}_t{timestep:04d}_sample{idx:03d}.{npy,png}`

### 4. Report
- ✓ Comprehensive markdown report with all required sections
- ✓ Task and Model definition
- ✓ Dataset verification with pretraining domain separation
- ✓ Schedule analysis with mathematics and comparison tables
- ✓ Experimental setup with hyperparameters
- ✓ Implementation details with before/after code
- ✓ Results with metrics and visualizations
- ✓ Analysis with mechanistic explanation grounded in mathematics
- ✓ Takeaway with findings and implications

## Troubleshooting

### Out-of-Memory Errors

**Issue:** CUDA out of memory during fine-tuning

**Solutions:**
- Reduce batch size: `--batch_size 2` or `--batch_size 1`
- Reduce epochs: `--epochs 10` (adjusts total steps proportionally)
- Use CPU (slow): Remove CUDA check or set `CUDA_VISIBLE_DEVICES=""`

### Model Download Failures

**Issue:** Hugging Face model download hangs

**Solution:**
```bash
# Set cache directory and retry
export HF_HOME="./cache"
python finetune_instructpix2pix.py --run A ...
```

### Dataset Organization Issues

**Issue:** "No valid image-label pairs found"

**Verify:**
```bash
ls data/cmp_facade/images/ | wc -l  # Should be 606
ls data/cmp_facade/labels/ | wc -l  # Should be 606
```

Files must have matching names (only extension differs: .jpg vs. .png).

## References

- **InstructPix2Pix:** Brooks et al., "InstructPix2Pix: Learning to Follow Image Editing Instructions" (CVPR 2023, arXiv:2211.09800v2)
- **Stable Diffusion:** Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022, arXiv:2112.10752)
- **CMP Facade Database:** https://cmp.felk.cvut.cz/~tylecr1/facade/

## Contact

For questions about this implementation:
- **Jash Lathiya (M26AI1050)** - Project coordination
- **Jayant Parmar (M26AI1051)** - Implementation and evaluation
- **Kumar Viplav (M26AI1056)** - Experimental design
- **Mamata Das (M26AI1060)** - Analysis and reporting

---

**Last Updated:** October 2026  
**Status:** Ready for Phase 2 Submission
