# Complete Project: Noise Schedule Ablation in Diffusion Models

**Course:** Foundation Models and Generative AI  
**Group:** Jash Lathiya (M26AI1050), Jayant Parmar (M26AI1051), Kumar Viplav (M26AI1056), Mamata Das (M26AI1060)  
**Date:** October 16, 2026

---

## 📚 Project Overview

This project explores noise schedule ablation in diffusion models across three distinct phases:

- **Phase 1:** Audit of existing InstructPix2Pix implementation (baseline)
- **Phase 2:** InstructPix2Pix with noise schedule ablation
- **Phase 3:** ControlNet with noise schedule ablation
- **Analysis:** Comparative study of both architectures

**Hypothesis:** Power-transformed noise schedules (β_t^x) create distributional mismatch during inference, affecting output quality.

---

## 📁 Project Structure

```
Assignment/
│
├── Phase_1_Audit/                              ← Phase 1 (Reference)
│   └── Audit_Reports/                          ← Audit documents only
│
├── Phase_2_InstructPix2Pix/                    ← Phase 2 (Separate)
│   ├── Scripts/
│   │   └── finetune_instructpix2pix.py
│   ├── Results/
│   │   ├── Run_0/
│   │   ├── Run_A/
│   │   └── Run_B/
│   └── README_PHASE2.md
│
├── Phase_3_ControlNet_Implementation/          ← Phase 3 (Separate)
│   ├── Scripts/
│   │   ├── finetune_controlnet.py
│   │   └── EXPONENT.py
│   ├── Documentation/
│   │   └── *.md
│   ├── Results/
│   │   ├── ControlNet_Run_0/
│   │   ├── ControlNet_Run_A/
│   │   └── ControlNet_Run_B/
│   └── README_PHASE3.md
│
├── Comparative_Analysis/                       ← Analysis (After Both)
│   ├── Training_Logs_Analysis/
│   ├── Visual_Quality_Comparison/
│   ├── Noise_Schedule_Analysis/
│   ├── Phase_2_vs_Phase_3_Comparison.md
│   └── ANALYSIS_GUIDE.md
│
├── data/
│   └── cmp_facade_complete/
│       ├── images/    (606 JPEG files)
│       └── labels/    (606 PNG files)
│
├── RUN_PHASE2.sh                              ← Execute Phase 2
├── RUN_PHASE3.sh                              ← Execute Phase 3
├── RUN_ANALYSIS.sh                            ← Run Analysis
├── PROJECT_STRUCTURE.md
└── MASTER_README.md                           ← This file
```

---

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Verify Python version
python3 --version                              # Should be 3.10+

# Verify GPU/device
python3 -c "import torch; print(torch.mps.is_available())"  # True for Mac

# Verify dataset
ls data/cmp_facade_complete/images | wc -l     # Should be 606
```

### Phase 2: Run InstructPix2Pix Experiments

```bash
bash RUN_PHASE2.sh
```

**What it does:**
- Run 0: Zero-shot baseline (no fine-tuning)
- Run A: Fine-tuning with native schedule
- Run B: Fine-tuning with ablated schedule (x=2.20)

**Results saved to:**
- `Phase_2_InstructPix2Pix/Results/Run_{0,A,B}/`

**Expected time:** ~2-3 hours GPU (or skip if using existing results)

---

### Phase 3: Run ControlNet Experiments

```bash
bash RUN_PHASE3.sh
```

**What it does:**
- Run 0: Zero-shot baseline (no fine-tuning)
- Run A: Fine-tuning with native schedule
- Run B: Fine-tuning with ablated schedule (x=2.20)

**Results saved to:**
- `Phase_3_ControlNet_Implementation/Results/ControlNet_Run_{0,A,B}/`

**Expected time:** ~2-3 hours GPU

---

### Run Analysis (After Both Phases)

```bash
bash RUN_ANALYSIS.sh
```

**What it does:**
- Organizes Phase 2 & 3 results
- Compares training logs
- Groups sample images
- Generates analysis structure

**Results saved to:**
- `Comparative_Analysis/`

**Expected time:** ~5 minutes

---

## 📊 Execution Options

### Option 1: Run Everything Sequentially (Full Study)

```bash
# Step 1: Phase 2 experiments
bash RUN_PHASE2.sh          # Choose: All runs / Quick test / Single run

# Step 2: Phase 3 experiments
bash RUN_PHASE3.sh          # Choose: All runs / Quick test / Single run

# Step 3: Analysis
bash RUN_ANALYSIS.sh        # Auto-collects and organizes results
```

**Total time:** ~5-8 hours GPU + 30 min analysis

---

### Option 2: Phase 3 Only (Using Phase 2 Results)

If Phase 2 results already exist:

```bash
# Just run Phase 3
bash RUN_PHASE3.sh          # Runs all experiments

# Then analyze
bash RUN_ANALYSIS.sh        # Compares with existing Phase 2 results
```

**Total time:** ~2-3 hours GPU + 30 min analysis

---

### Option 3: Individual Runs

Run specific experiments without full automation:

```bash
# Phase 2: Run A only
python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
  --run A \
  --dataset_root ./data/cmp_facade_complete \
  --output_dir ./Phase_2_InstructPix2Pix/Results \
  --epochs 20 --batch_size 4 --seed 42

# Phase 3: Run A only
python3 Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py \
  --run A \
  --dataset_root ./data/cmp_facade_complete \
  --output_dir ./Phase_3_ControlNet_Implementation/Results \
  --epochs 20 --batch_size 4 --seed 42
```

---

## 📋 What Each Phase Does

### Phase 1: Audit (Reference Only)

Located in: `Phase_1_Audit/Audit_Reports/`

**Contents:**
- Code audit reports
- Technical verification
- Implementation validation

**Action:** Read for reference, no execution needed

**Files:**
- CODE_AUDIT_REPORT.md
- AUDIT_SUMMARY_AND_NEXT_STEPS.md
- VERIFIED_FINDINGS.md

---

### Phase 2: InstructPix2Pix

Located in: `Phase_2_InstructPix2Pix/`

**Implementation:** Text-conditioned image-to-image diffusion model

**Three Experimental Runs:**

1. **Run 0: Zero-shot Baseline**
   - No fine-tuning
   - Pretrained model on unseen modality
   - Expected: Poor quality (out-of-distribution)

2. **Run A: Native Schedule**
   - Fine-tuning with standard Stable Diffusion schedule
   - Aligned: Training and inference use same schedule
   - Expected: Good quality

3. **Run B: Ablated Schedule (x=2.20)**
   - Fine-tuning with power-transformed schedule: β_t^2.20
   - Misaligned: Inference uses native, training used ablated
   - Expected: Degraded quality

**Outputs:**
- Training logs: `training_logs.csv`
- Checkpoints: `checkpoint_{4,8,12,16,20}/`
- Final model: `final_model/`
- Generated samples: `generated_samples/`

---

### Phase 3: ControlNet

Located in: `Phase_3_ControlNet_Implementation/`

**Implementation:** Semantic-conditioned image-to-image diffusion model

**Three Experimental Runs:**

1. **Run 0: Zero-shot Baseline**
   - No fine-tuning
   - Pretrained ControlNet on unseen semantic modality
   - Expected: Poor quality

2. **Run A: Native Schedule**
   - Fine-tuning with standard Stable Diffusion schedule
   - Expected: Good quality

3. **Run B: Ablated Schedule (x=2.20)**
   - Fine-tuning with power-transformed schedule: β_t^2.20
   - Expected: Degraded quality (same as Phase 2)

**Outputs:** Same structure as Phase 2

---

## 🔍 Analysis

Located in: `Comparative_Analysis/`

**Purpose:** Compare Phase 2 and Phase 3 results to understand:
- How architecture affects schedule sensitivity
- Whether the hypothesis holds in both contexts
- Qualitative differences in failure modes

**Analysis Folders:**

1. **Training_Logs_Analysis/**
   - CSV files from both phases
   - Auto-generated loss comparison
   - Statistical analysis

2. **Visual_Quality_Comparison/**
   - Sample images from all runs
   - Phase 2 vs Phase 3 side-by-side
   - Manual quality assessment

3. **Noise_Schedule_Analysis/**
   - Technical schedule comparison
   - Run A vs Run B impact analysis
   - Distributional mismatch mechanism

**Main Reports:**
- `Phase_2_vs_Phase_3_Comparison.md` - Main comparison
- `ANALYSIS_GUIDE.md` - How to analyze
- `Findings_and_Conclusions.md` - Summary (to be created)

---

## 📖 Key Files Reference

### Execution Scripts
- `RUN_PHASE2.sh` - Execute Phase 2 (interactive, choose runs)
- `RUN_PHASE3.sh` - Execute Phase 3 (interactive, choose runs)
- `RUN_ANALYSIS.sh` - Analyze both phases (auto-collects results)

### Configuration
- `EXPONENT.py` - Noise schedule exponent: 2.20
- `PROJECT_STRUCTURE.md` - Detailed folder organization
- `MASTER_README.md` - This file

### Implementation
- `Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py`
- `Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py`

### Documentation
- `Phase_2_InstructPix2Pix/README_PHASE2.md` - Phase 2 details
- `Phase_3_ControlNet_Implementation/Documentation/README_PHASE3.md` - Phase 3 details
- `Comparative_Analysis/ANALYSIS_GUIDE.md` - Analysis instructions

### Data
- `data/cmp_facade_complete/images/` - 606 facade photographs
- `data/cmp_facade_complete/labels/` - 606 semantic masks

---

## ⏱️ Timeline & Effort

### Phase 2: InstructPix2Pix
- Setup: 5 minutes
- Run 0: 10 minutes
- Run A: 1 hour GPU
- Run B: 1 hour GPU
- **Total:** ~2-3 hours (can skip if using existing)

### Phase 3: ControlNet
- Setup: 5 minutes
- Run 0: 10 minutes
- Run A: 1 hour GPU
- Run B: 1 hour GPU
- **Total:** ~2-3 hours GPU

### Analysis
- Organization: 5 minutes
- Review: 30-60 minutes
- Documentation: 1-2 hours
- **Total:** ~2-3 hours

### Grand Total
- **With both phases:** 5-8 hours GPU + 2-3 hours analysis
- **Phase 3 only (using Phase 2):** 2-3 hours GPU + 2-3 hours analysis

---

## 🎯 What to Look For

### In Training Logs

**Phase 2 & 3 should show:**
- Run A: Smooth loss curve, convergence to lower loss
- Run B: May show different loss pattern
- Run B vs Run A: Loss difference indicates schedule impact

### In Generated Samples

**Expected patterns:**
- Run 0: Poor quality (baseline)
- Run A: Good quality (in-distribution)
- Run B: Variable quality (distributional mismatch)
- Run B artifacts: Might show distortions from schedule mismatch

### Across Architectures

**Compare Phase 2 vs Phase 3:**
- Same noise schedule effect?
- Different sensitivity levels?
- Which architecture more robust?

---

## ✅ Completion Checklist

### Phase 2 (Optional if Results Exist)
- [ ] Run Phase 2: `bash RUN_PHASE2.sh`
- [ ] Check results: `ls Phase_2_InstructPix2Pix/Results/Run_A/`
- [ ] Verify logs: `cat Phase_2_InstructPix2Pix/Results/Run_A/training_logs.csv`

### Phase 3
- [ ] Run Phase 3: `bash RUN_PHASE3.sh`
- [ ] Check results: `ls Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A/`
- [ ] Verify logs: `cat Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A/training_logs.csv`

### Analysis
- [ ] Run analysis: `bash RUN_ANALYSIS.sh`
- [ ] Check structure: `ls Comparative_Analysis/`
- [ ] Review logs: `cat Comparative_Analysis/Training_Logs_Analysis/loss_comparison_analysis.md`
- [ ] Compare samples: `ls Comparative_Analysis/Visual_Quality_Comparison/`

### Documentation
- [ ] Read: `Comparative_Analysis/ANALYSIS_GUIDE.md`
- [ ] Create: `visual_quality_report.md`
- [ ] Create: `schedule_comparison.md`
- [ ] Create: `Findings_and_Conclusions.md`

---

## 🆘 Troubleshooting

### Dataset Issues
```bash
# Verify dataset
ls data/cmp_facade_complete/images | wc -l  # Should be 606
```

### GPU/Device Issues
```bash
# Check device
python3 -c "import torch; print('MPS:', torch.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
```

### Memory Issues
```bash
# Use smaller batch size
python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
  --run A \
  --dataset_root ./data/cmp_facade_complete \
  --output_dir ./Phase_2_InstructPix2Pix/Results \
  --batch_size 2  # Reduced from 4
```

### Script Permissions
```bash
# Make scripts executable
chmod +x RUN_PHASE2.sh RUN_PHASE3.sh RUN_ANALYSIS.sh
```

---

## 📞 Support

### For Phase 2 Questions
See: `Phase_2_InstructPix2Pix/README_PHASE2.md`

### For Phase 3 Questions
See: `Phase_3_ControlNet_Implementation/Documentation/README_PHASE3.md`

### For Analysis Help
See: `Comparative_Analysis/ANALYSIS_GUIDE.md`

---

## 🎓 Learning Outcomes

By completing this project, you will:

✅ Understand noise schedule design in diffusion models  
✅ Implement custom noise schedulers (native and ablated)  
✅ Train diffusion models with different conditioning approaches  
✅ Compare two state-of-the-art architectures (InstructPix2Pix, ControlNet)  
✅ Analyze the impact of schedule ablation on model performance  
✅ Create reproducible ML experiments  
✅ Perform comparative analysis of complex systems  

---

## 📝 References

- **Stable Diffusion:** Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
- **InstructPix2Pix:** Brooks et al., "InstructPix2Pix: Learning to Follow Image Editing Instructions" (CVPR 2023)
- **ControlNet:** Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models" (arXiv:2302.05543)
- **CMP Facade Database:** https://cmp.felk.cvut.cz/~tylecr1/facade/

---

## 📄 Document Index

| Document | Purpose |
|----------|---------|
| `PROJECT_STRUCTURE.md` | Detailed folder organization |
| `MASTER_README.md` | This file - full project overview |
| `Phase_2_InstructPix2Pix/README_PHASE2.md` | Phase 2 execution guide |
| `Phase_3_ControlNet_Implementation/Documentation/README_PHASE3.md` | Phase 3 execution guide |
| `Comparative_Analysis/ANALYSIS_GUIDE.md` | How to analyze results |
| `Phase_1_Audit/Audit_Reports/*.md` | Reference audit documents |

---

## 🚀 Ready to Begin?

**Start here:**

```bash
# Option 1: Run Phase 2
bash RUN_PHASE2.sh

# Option 2: Run Phase 3  
bash RUN_PHASE3.sh

# Option 3: Run both then analyze
bash RUN_PHASE2.sh && bash RUN_PHASE3.sh && bash RUN_ANALYSIS.sh
```

**Or for more control:**

```bash
# See available options
bash RUN_PHASE2.sh      # Choose specific runs
bash RUN_PHASE3.sh      # Choose specific runs
bash RUN_ANALYSIS.sh    # Auto-collects results
```

---

**Status:** ✅ Fully organized and ready for execution  
**Last Updated:** October 16, 2026  
**Next Step:** Choose a starting option above and execute!

