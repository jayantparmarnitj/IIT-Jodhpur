# Project Structure: Phases 1-3 with Comparative Analysis

**Date:** October 16, 2026  
**Status:** 🚀 Reorganized and Ready for Execution

---

## 📁 Project Folder Organization

```
Assignment/
│
├── 📁 Phase_1_Audit/                           ← Phase 1: Audit Documents
│   └── Audit_Reports/
│       ├── CODE_AUDIT_REPORT.md
│       ├── AUDIT_SUMMARY_AND_NEXT_STEPS.md
│       ├── VERIFIED_FINDINGS.md
│       ├── AUDIT_EXECUTIVE_SUMMARY.txt
│       ├── ANALYSIS_INDEX.md
│       └── PERFORMANCE_ANALYSIS_REPORT.md
│
├── 📁 Phase_2_InstructPix2Pix/                 ← Phase 2: InstructPix2Pix
│   ├── Scripts/
│   │   └── finetune_instructpix2pix.py
│   ├── Results/
│   │   ├── Run_0/                             ← Zero-shot baseline
│   │   ├── Run_A/                             ← Native schedule
│   │   └── Run_B/                             ← Ablated schedule
│   └── README_PHASE2.md
│
├── 📁 Phase_3_ControlNet_Implementation/       ← Phase 3: ControlNet
│   ├── Scripts/
│   │   ├── finetune_controlnet.py
│   │   └── EXPONENT.py
│   ├── Documentation/
│   │   ├── START_HERE.md
│   │   ├── README_PHASE3.md
│   │   ├── PHASE3_READY_TO_RUN.md
│   │   ├── DATASET_VERIFICATION_COMPLETE.md
│   │   └── (other documentation)
│   ├── Results/
│   │   ├── Run_0/                             ← Zero-shot baseline
│   │   ├── Run_A/                             ← Native schedule
│   │   └── Run_B/                             ← Ablated schedule
│   └── README_PHASE3.md
│
├── 📁 Comparative_Analysis/                    ← Analysis & Comparison
│   ├── Phase_2_vs_Phase_3_Comparison.md        ← Main comparison report
│   ├── Training_Logs_Analysis/
│   │   ├── Phase2_vs_Phase3_Loss_Curves.png
│   │   ├── Phase2_vs_Phase3_Training_Stats.csv
│   │   └── loss_comparison_analysis.md
│   ├── Visual_Quality_Comparison/
│   │   ├── Phase2_samples/
│   │   │   ├── Run_0_samples/
│   │   │   ├── Run_A_samples/
│   │   │   └── Run_B_samples/
│   │   ├── Phase3_samples/
│   │   │   ├── Run_0_samples/
│   │   │   ├── Run_A_samples/
│   │   │   └── Run_B_samples/
│   │   └── visual_quality_report.md
│   ├── Noise_Schedule_Analysis/
│   │   ├── schedule_comparison.md
│   │   ├── Run_A_vs_Run_B_impact.md
│   │   └── distributional_mismatch_analysis.md
│   └── Findings_and_Conclusions.md
│
├── 📁 data/
│   └── cmp_facade_complete/
│       ├── images/     (606 JPEG files)
│       └── labels/     (606 PNG files)
│
└── 📄 Project_Execution_Plan.md                ← This document
```

---

## 🚀 Execution Plan: Phase 2 & Phase 3 Runs

### Phase 2: InstructPix2Pix (Existing + Rerun if Needed)

**Status:** Previous runs exist with results  
**Location:** `Phase_2_InstructPix2Pix/Results/`

#### Option 1: Skip (Use Existing Results)
- Current Run_0, Run_A, Run_B results already exist
- Use for comparison with Phase 3

#### Option 2: Rerun All (Fresh Data)
```bash
# Run 0: Zero-shot baseline
python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py --run 0 \
  --dataset_root ./data/cmp_facade_complete \
  --output_dir ./Phase_2_InstructPix2Pix/Results \
  --seed 42

# Run A: Native schedule
python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py --run A \
  --dataset_root ./data/cmp_facade_complete \
  --output_dir ./Phase_2_InstructPix2Pix/Results \
  --epochs 20 --batch_size 4 --seed 42

# Run B: Ablated schedule
python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py --run B \
  --dataset_root ./data/cmp_facade_complete \
  --output_dir ./Phase_2_InstructPix2Pix/Results \
  --epochs 20 --batch_size 4 --seed 42
```

**Output:** Will appear in `Phase_2_InstructPix2Pix/Results/Run_{0,A,B}/`

---

### Phase 3: ControlNet (New Implementation)

**Status:** Ready to run  
**Location:** `Phase_3_ControlNet_Implementation/`

```bash
# Run 0: Zero-shot baseline
python3 Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py --run 0 \
  --dataset_root ./data/cmp_facade_complete \
  --output_dir ./Phase_3_ControlNet_Implementation/Results \
  --seed 42

# Run A: Native schedule
python3 Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py --run A \
  --dataset_root ./data/cmp_facade_complete \
  --output_dir ./Phase_3_ControlNet_Implementation/Results \
  --epochs 20 --batch_size 4 --seed 42

# Run B: Ablated schedule
python3 Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py --run B \
  --dataset_root ./data/cmp_facade_complete \
  --output_dir ./Phase_3_ControlNet_Implementation/Results \
  --epochs 20 --batch_size 4 --seed 42
```

**Output:** Will appear in `Phase_3_ControlNet_Implementation/Results/ControlNet_Run_{0,A,B}/`

---

## 📊 Comparative Analysis Structure

### 1. Training Logs Analysis
**Location:** `Comparative_Analysis/Training_Logs_Analysis/`

Compare training curves:
```
Phase 2 (InstructPix2Pix):
  - Phase_2_InstructPix2Pix/Results/Run_A/training_logs.csv
  - Phase_2_InstructPix2Pix/Results/Run_B/training_logs.csv

Phase 3 (ControlNet):
  - Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A/training_logs.csv
  - Phase_3_ControlNet_Implementation/Results/ControlNet_Run_B/training_logs.csv
```

Analyze:
- Loss curves: Run A vs Run B (within each phase)
- Convergence rate comparison
- Schedule impact (ablated vs native)

### 2. Visual Quality Comparison
**Location:** `Comparative_Analysis/Visual_Quality_Comparison/`

Compare generated samples:
```
Phase 2 Samples:
  - Run_0: Zero-shot quality
  - Run_A: Native schedule (expected: good)
  - Run_B: Ablated schedule (expected: degraded)

Phase 3 Samples:
  - Run_0: Zero-shot quality
  - Run_A: Native schedule (expected: good)
  - Run_B: Ablated schedule (expected: degraded)
```

Metrics to evaluate:
- Visual realism (subjective)
- Semantic accuracy (label-to-output correspondence)
- Artifact presence
- Detail preservation

### 3. Noise Schedule Analysis
**Location:** `Comparative_Analysis/Noise_Schedule_Analysis/`

Document:
- How schedule impacts training per architecture
- Distributional mismatch effects (Run B)
- Differences between InstructPix2Pix and ControlNet
- Hypotheses and findings

### 4. Main Comparison Report
**Location:** `Comparative_Analysis/Phase_2_vs_Phase_3_Comparison.md`

Cover:
- Architecture differences
- Conditioning approach differences
- Training results comparison
- Inference quality comparison
- Schedule ablation effectiveness in each model
- Overall findings and conclusions

---

## 📋 Quick Execution Checklist

### Setup
- [ ] Dataset verified: 606 images + labels in `data/cmp_facade_complete/`
- [ ] Phase 2 script ready: `Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py`
- [ ] Phase 3 script ready: `Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py`
- [ ] GPU/MPS device ready: `python3 -c "import torch; print(torch.mps.is_available())"`

### Phase 2 Runs (if rerunning)
- [ ] Run 0: Zero-shot baseline (~10 min)
- [ ] Run A: Native schedule (~1 hour GPU)
- [ ] Run B: Ablated schedule (~1 hour GPU)
- [ ] Results saved to: `Phase_2_InstructPix2Pix/Results/`

### Phase 3 Runs
- [ ] Run 0: Zero-shot baseline (~10 min)
- [ ] Run A: Native schedule (~1 hour GPU)
- [ ] Run B: Ablated schedule (~1 hour GPU)
- [ ] Results saved to: `Phase_3_ControlNet_Implementation/Results/`

### Analysis
- [ ] Collect training logs from both phases
- [ ] Organize sample images by run and phase
- [ ] Generate loss curve comparisons
- [ ] Write comparative analysis report
- [ ] Document findings in `Comparative_Analysis/`

---

## 🎯 Analysis Tasks

### 1. Training Dynamics Comparison

**Questions to Answer:**
- How do loss curves differ between Phase 2 and Phase 3?
- Does Run B (ablated schedule) show different loss patterns than Run A in each phase?
- Which architecture converges faster?
- Are there any unexpected behaviors?

**Output:**
- `Comparative_Analysis/Training_Logs_Analysis/loss_comparison_analysis.md`
- Visualization: Loss curves overlaid for both phases

### 2. Visual Quality Assessment

**Questions to Answer:**
- Does Phase 2 produce better quality outputs than Phase 3 or vice versa?
- How does the schedule ablation (Run B) affect output quality visually in each architecture?
- Which architecture handles the distributional mismatch better?
- Are there qualitative differences in failure modes?

**Output:**
- `Comparative_Analysis/Visual_Quality_Comparison/visual_quality_report.md`
- Sample images organized by phase and run

### 3. Schedule Impact Analysis

**Questions to Answer:**
- How significant is the distributional mismatch effect (Run A vs Run B) in each phase?
- Do both architectures show similar schedule sensitivity?
- What is the mechanism of impact (loss-wise, quality-wise)?

**Output:**
- `Comparative_Analysis/Noise_Schedule_Analysis/Run_A_vs_Run_B_impact.md`
- Detailed breakdown of schedule effects

### 4. Overall Findings

**Summary Report:**
- `Comparative_Analysis/Phase_2_vs_Phase_3_Comparison.md`
- `Comparative_Analysis/Findings_and_Conclusions.md`

Cover:
- Key insights from Phase 2 vs Phase 3 comparison
- Noise schedule ablation effectiveness
- Architecture-specific behaviors
- Recommendations for future work

---

## 📂 File Organization Reference

### Phase 1 (Audit Only - No Rerun Needed)
```
Phase_1_Audit/
└── Audit_Reports/          ← Read-only, for reference
```

### Phase 2 (Can Use Existing or Rerun)
```
Phase_2_InstructPix2Pix/
├── Scripts/                ← Training script
├── Results/                ← Run 0, A, B outputs
└── README_PHASE2.md        ← Documentation
```

### Phase 3 (New - Ready to Run)
```
Phase_3_ControlNet_Implementation/
├── Scripts/                ← Training script
├── Documentation/          ← Guides and references
├── Results/                ← Run 0, A, B outputs (will be created)
└── README_PHASE3.md        ← Documentation
```

### Analysis (New - To Be Filled)
```
Comparative_Analysis/
├── Phase_2_vs_Phase_3_Comparison.md    ← Main report
├── Training_Logs_Analysis/             ← Loss curves & stats
├── Visual_Quality_Comparison/          ← Sample images & eval
├── Noise_Schedule_Analysis/            ← Schedule impact
└── Findings_and_Conclusions.md         ← Summary
```

---

## 🎓 Analysis Workflow

### Step 1: Collect Results (After All Runs)
```
Copy/organize outputs:
Phase_2_InstructPix2Pix/Results/Run_{0,A,B}/
  - training_logs.csv
  - final_model/
  - generated_samples/

Phase_3_ControlNet_Implementation/Results/ControlNet_Run_{0,A,B}/
  - training_logs.csv
  - final_model/
  - generated_samples/
```

### Step 2: Analyze Training Logs
```python
# Compare loss curves
import pandas as pd

phase2_a = pd.read_csv('Phase_2_InstructPix2Pix/Results/Run_A/training_logs.csv')
phase2_b = pd.read_csv('Phase_2_InstructPix2Pix/Results/Run_B/training_logs.csv')

phase3_a = pd.read_csv('Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A/training_logs.csv')
phase3_b = pd.read_csv('Phase_3_ControlNet_Implementation/Results/ControlNet_Run_B/training_logs.csv')

# Generate comparison plots
# Write analysis document
```

### Step 3: Compare Visual Quality
```
Organize samples:
Comparative_Analysis/Visual_Quality_Comparison/
├── Phase2_samples/
│   ├── Run_0_samples/
│   ├── Run_A_samples/
│   └── Run_B_samples/
├── Phase3_samples/
│   ├── Run_0_samples/
│   ├── Run_A_samples/
│   └── Run_B_samples/
```

### Step 4: Write Comparative Report
```
Output: Comparative_Analysis/Phase_2_vs_Phase_3_Comparison.md

Sections:
- Architecture Overview
- Training Results
- Visual Quality Analysis
- Schedule Impact Comparison
- Key Findings
- Conclusions
```

---

## 📊 Expected Timeline

### Phase 2 Rerun (Optional)
- Run 0: ~10 minutes
- Run A: ~1 hour
- Run B: ~1 hour
- **Total:** ~2-3 hours GPU time (if rerunning)

### Phase 3 First Run (New)
- Run 0: ~10 minutes
- Run A: ~1 hour
- Run B: ~1 hour
- **Total:** ~2-3 hours GPU time

### Analysis
- Organize files: ~30 min
- Generate visualizations: ~30 min
- Write reports: ~2-3 hours
- **Total:** ~3-4 hours

### Grand Total
- **If using existing Phase 2 results:** ~2-3 hours (Phase 3 runs + analysis)
- **If rerunning all:** ~8-10 hours (all runs + analysis)

---

## ✅ Folder Structure Summary

```
Assignment/
├── Phase_1_Audit/                    ← Audit documents (reference)
├── Phase_2_InstructPix2Pix/          ← InstructPix2Pix (scripts + results)
├── Phase_3_ControlNet_Implementation/ ← ControlNet (scripts + results)
├── Comparative_Analysis/             ← Analysis & comparisons (to be filled)
├── data/
│   └── cmp_facade_complete/          ← Dataset (606 images)
└── Project_Execution_Plan.md         ← This document
```

---

## 🚀 Ready to Begin?

1. **Check dataset:** `ls data/cmp_facade_complete/images | wc -l` (should be 606)
2. **Choose Phase 2 approach:** Use existing results or rerun
3. **Run Phase 3:** Execute the three scripts (Run 0, A, B)
4. **Analyze:** Collect results and generate comparative report
5. **Document:** Create final analysis in `Comparative_Analysis/`

You're all set! 🎉

