#!/usr/bin/env python3
"""
Comparative Analysis: Phase 2 (InstructPix2Pix) vs Phase 3 (ControlNet)

This script analyzes results from both phases and generates comprehensive
comparison reports including:
1. Training loss curve comparison
2. Schedule impact analysis
3. Visual quality assessment
4. Summary findings
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

# Color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

def print_header(title):
    """Print a formatted header."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{title.center(70)}{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(msg):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_warning(msg):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_error(msg):
    """Print error message."""
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def load_training_logs(phase_path):
    """Load training logs from CSV."""
    try:
        df = pd.read_csv(phase_path)
        return df
    except Exception as e:
        return None

def analyze_training_logs():
    """Analyze and compare training logs from both phases."""
    print_header("TRAINING LOGS ANALYSIS")
    
    analysis_dir = Path("Comparative_Analysis/Training_Logs_Analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load logs
    phase2_a = load_training_logs("Phase_2_InstructPix2Pix/Results/Run_A/training_logs.csv")
    phase2_b = load_training_logs("Phase_2_InstructPix2Pix/Results/Run_B/training_logs.csv")
    phase3_a = load_training_logs("Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A/training_logs.csv")
    phase3_b = load_training_logs("Phase_3_ControlNet_Implementation/Results/ControlNet_Run_B/training_logs.csv")
    
    # Check which logs exist
    logs_available = {
        'Phase 2 Run A': phase2_a is not None,
        'Phase 2 Run B': phase2_b is not None,
        'Phase 3 Run A': phase3_a is not None,
        'Phase 3 Run B': phase3_b is not None,
    }
    
    print("Available logs:")
    for name, available in logs_available.items():
        status = f"{Colors.GREEN}✓ Found{Colors.END}" if available else f"{Colors.RED}✗ Missing{Colors.END}"
        print(f"  {name}: {status}")
    
    # Generate statistics
    stats = {}
    
    if phase2_a is not None:
        stats['Phase2_RunA'] = {
            'final_loss': phase2_a['train_loss'].iloc[-1],
            'min_loss': phase2_a['train_loss'].min(),
            'max_loss': phase2_a['train_loss'].max(),
            'avg_loss': phase2_a['train_loss'].mean(),
            'epochs': len(phase2_a)
        }
        print_success(f"Phase 2 Run A: Final loss = {stats['Phase2_RunA']['final_loss']:.6f}")
    
    if phase2_b is not None:
        stats['Phase2_RunB'] = {
            'final_loss': phase2_b['train_loss'].iloc[-1],
            'min_loss': phase2_b['train_loss'].min(),
            'max_loss': phase2_b['train_loss'].max(),
            'avg_loss': phase2_b['train_loss'].mean(),
            'epochs': len(phase2_b)
        }
        print_success(f"Phase 2 Run B: Final loss = {stats['Phase2_RunB']['final_loss']:.6f}")
    
    if phase3_a is not None:
        stats['Phase3_RunA'] = {
            'final_loss': phase3_a['train_loss'].iloc[-1],
            'min_loss': phase3_a['train_loss'].min(),
            'max_loss': phase3_a['train_loss'].max(),
            'avg_loss': phase3_a['train_loss'].mean(),
            'epochs': len(phase3_a)
        }
        print_success(f"Phase 3 Run A: Final loss = {stats['Phase3_RunA']['final_loss']:.6f}")
    
    if phase3_b is not None:
        stats['Phase3_RunB'] = {
            'final_loss': phase3_b['train_loss'].iloc[-1],
            'min_loss': phase3_b['train_loss'].min(),
            'max_loss': phase3_b['train_loss'].max(),
            'avg_loss': phase3_b['train_loss'].mean(),
            'epochs': len(phase3_b)
        }
        print_success(f"Phase 3 Run B: Final loss = {stats['Phase3_RunB']['final_loss']:.6f}")
    
    # Save statistics
    stats_path = analysis_dir / "Phase2_vs_Phase3_Training_Stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print_success(f"Statistics saved to {stats_path}")
    
    return stats

def create_loss_comparison_analysis():
    """Create detailed loss comparison analysis."""
    print_header("LOSS COMPARISON ANALYSIS")
    
    analysis_dir = Path("Comparative_Analysis/Training_Logs_Analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    report = """# Loss Comparison Analysis: Phase 2 vs Phase 3

## Executive Summary

This report compares training loss curves between Phase 2 (InstructPix2Pix) and Phase 3 (ControlNet) 
experiments, focusing on:
1. Native schedule (Run A) - In-distribution training
2. Ablated schedule (Run B) - Out-of-distribution training
3. Impact of noise schedule ablation (Run A vs Run B)

## Phase 2: InstructPix2Pix Results

### Run A (Native Schedule)
- Architecture: InstructPix2Pix
- Noise Schedule: β_t (standard)
- Expected: In-distribution training
- Status: Check Run_A/training_logs.csv for details

### Run B (Ablated Schedule)  
- Architecture: InstructPix2Pix
- Noise Schedule: β_t^2.20 (power-transformed)
- Expected: Out-of-distribution training
- Status: Check Run_B/training_logs.csv for details

## Phase 3: ControlNet Results

### Run A (Native Schedule)
- Architecture: ControlNet + Stable Diffusion
- Noise Schedule: β_t (standard)
- Expected: In-distribution training
- Status: Check Run_A/training_logs.csv for details

### Run B (Ablated Schedule)
- Architecture: ControlNet + Stable Diffusion
- Noise Schedule: β_t^2.20 (power-transformed)
- Expected: Out-of-distribution training
- Status: Check Run_B/training_logs.csv for details

## Key Findings

### Within-Phase Comparison (Run A vs Run B)

**Phase 2 (InstructPix2Pix):**
- Schedule impact on training dynamics
- Difference in convergence rates
- Final loss comparison

**Phase 3 (ControlNet):**
- Schedule impact on training dynamics
- Difference in convergence rates
- Final loss comparison

### Cross-Phase Comparison (Phase 2 vs Phase 3)

**Native Schedule (Run A):**
- Which architecture converges faster?
- Which achieves lower final loss?
- Training stability comparison

**Ablated Schedule (Run B):**
- Which architecture is affected more by schedule mismatch?
- Which shows signs of distributional divergence?
- Recovery capability comparison

## Schedule Ablation Hypothesis

**Expected:**
- Run A (Native): Familiar noise distribution during training → Better convergence
- Run B (Ablated): Mismatched noise distribution → Potential convergence issues

**Observed:**
- [Results to be filled after experiments complete]

## Conclusions

[Conclusions to be added after full analysis]

---

Generated: {timestamp}
"""
    
    report_path = analysis_dir / "loss_comparison_analysis.md"
    with open(report_path, 'w') as f:
        f.write(report.format(timestamp=datetime.now().isoformat()))
    
    print_success(f"Loss comparison report created: {report_path}")

def organize_sample_images():
    """Organize generated sample images for comparison."""
    print_header("ORGANIZING SAMPLE IMAGES")
    
    comparison_dir = Path("Comparative_Analysis/Visual_Quality_Comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 2 samples
    phase2_dir = comparison_dir / "Phase2_samples"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    
    for run in ['Run_0', 'Run_A', 'Run_B']:
        src = Path(f"Phase_2_InstructPix2Pix/Results/{run}/")
        if src.exists() and src.is_dir():
            dst = phase2_dir / run / "generated_samples"
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy sample images
            sample_src = src / "generated_samples"
            if sample_src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(sample_src, dst)
                print_success(f"Phase 2 {run} samples copied")
    
    # Phase 3 samples
    phase3_dir = comparison_dir / "Phase3_samples"
    phase3_dir.mkdir(parents=True, exist_ok=True)
    
    for run in ['Run_0', 'ControlNet_Run_A', 'ControlNet_Run_B']:
        src = Path(f"Phase_3_ControlNet_Implementation/Results/{run}/")
        if src.exists() and src.is_dir():
            run_name = run.replace('ControlNet_', '')
            dst = phase3_dir / run_name / "generated_samples"
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy sample images
            sample_src = src / "generated_samples"
            if sample_src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(sample_src, dst)
                print_success(f"Phase 3 {run_name} samples copied")

def create_visual_quality_report():
    """Create visual quality comparison report."""
    print_header("VISUAL QUALITY ANALYSIS")
    
    analysis_dir = Path("Comparative_Analysis/Visual_Quality_Comparison")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    report = """# Visual Quality Comparison: Phase 2 vs Phase 3

## Overview

This report compares visual quality of generated samples from:
- Phase 2: InstructPix2Pix (text-conditioned)
- Phase 3: ControlNet (semantic-conditioned)

## Sample Organization

### Phase 2 Samples (InstructPix2Pix)
Location: `Phase2_samples/`
- Run_0: Zero-shot baseline (no fine-tuning)
- Run_A: Fine-tuned with native schedule
- Run_B: Fine-tuned with ablated schedule (x=2.20)

### Phase 3 Samples (ControlNet)
Location: `Phase3_samples/`
- Run_0: Zero-shot baseline (no fine-tuning)
- Run_A: Fine-tuned with native schedule
- Run_B: Fine-tuned with ablated schedule (x=2.20)

## Quality Metrics

### Evaluation Criteria

1. **Visual Realism**
   - How photorealistic are the generated facades?
   - Texture quality and color accuracy
   - Overall naturalness

2. **Semantic Accuracy**
   - How well does output match input semantic mask?
   - Correct element placement (windows, doors, etc.)
   - Semantic consistency

3. **Detail Preservation**
   - Level of architectural detail
   - Texture complexity
   - Boundary sharpness

4. **Artifact Presence**
   - Blurriness or distortion
   - Color bleeding
   - Structural inconsistencies

## Within-Phase Analysis

### Phase 2 (InstructPix2Pix)

**Run 0 (Zero-shot):**
- Quality: [To be assessed]
- Observations: [To be filled]

**Run A (Native Schedule):**
- Quality: [Expected: Good]
- Observations: [To be filled]

**Run B (Ablated Schedule):**
- Quality: [Expected: Degraded vs Run A]
- Observations: [To be filled]

**Run A vs Run B Difference:**
- Quality degradation: [To be assessed]
- Specific artifacts in Run B: [To be filled]

### Phase 3 (ControlNet)

**Run 0 (Zero-shot):**
- Quality: [To be assessed]
- Observations: [To be filled]

**Run A (Native Schedule):**
- Quality: [Expected: Good]
- Observations: [To be filled]

**Run B (Ablated Schedule):**
- Quality: [Expected: Degraded vs Run A]
- Observations: [To be filled]

**Run A vs Run B Difference:**
- Quality degradation: [To be assessed]
- Specific artifacts in Run B: [To be filled]

## Cross-Phase Comparison

### Architecture Comparison (Run A: Native Schedule)

**Phase 2 (InstructPix2Pix):**
- Strengths: [To be filled]
- Weaknesses: [To be filled]

**Phase 3 (ControlNet):**
- Strengths: [To be filled]
- Weaknesses: [To be filled]

**Head-to-Head Winner:** [To be determined after visual inspection]

### Schedule Sensitivity Comparison (Run A vs Run B)

**Phase 2 - Sensitivity to Ablation:**
- Impact severity: [High / Medium / Low]
- Artifacts introduced: [To be listed]

**Phase 3 - Sensitivity to Ablation:**
- Impact severity: [High / Medium / Low]
- Artifacts introduced: [To be listed]

**Which is more robust to schedule mismatch?** [To be determined]

## Key Observations

[To be filled after visual comparison]

## Conclusions

[To be added after comprehensive analysis]

---

Generated: {timestamp}
"""
    
    report_path = analysis_dir / "visual_quality_report.md"
    with open(report_path, 'w') as f:
        f.write(report.format(timestamp=datetime.now().isoformat()))
    
    print_success(f"Visual quality report created: {report_path}")

def create_main_comparison_report():
    """Create main Phase 2 vs Phase 3 comparison report."""
    print_header("GENERATING MAIN COMPARISON REPORT")
    
    analysis_dir = Path("Comparative_Analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    report = """# Phase 2 vs Phase 3: Comprehensive Comparison Report

**Date:** {timestamp}
**Comparison:** InstructPix2Pix (Phase 2) vs ControlNet (Phase 3)
**Noise Schedule Ablation:** Exponent = 2.20

---

## Executive Summary

This report compares two different architectural approaches to semantic-conditioned image-to-image synthesis:
- **Phase 2:** InstructPix2Pix (text-conditioned instruction following)
- **Phase 3:** ControlNet (semantic mask-conditioned spatial control)

Both phases test the hypothesis that noise schedule distributional mismatch affects output quality.

---

## 1. Architecture Overview

### Phase 2: InstructPix2Pix

**Model:** InstructPix2Pix
**Conditioning Type:** Text instructions + images
**Fine-tuning Target:** Full UNet
**Base Model:** timbrooks/instruct-pix2pix

**Conditioning Process:**
1. Encode text instruction with CLIP
2. Encode image with VAE
3. Feed both to UNet for noise prediction

### Phase 3: ControlNet

**Model:** ControlNet + Stable Diffusion
**Conditioning Type:** Semantic masks (RGB images) + text prompts
**Fine-tuning Target:** ControlNet module only
**Base Models:** 
- ControlNet: lllyasviel/sd-controlnet-seg
- Diffusion: runwayml/stable-diffusion-v1-5

**Conditioning Process:**
1. Encode semantic mask to RGB conditioning image
2. Encode text prompt with CLIP
3. ControlNet learns spatial mapping
4. UNet generates image conditioned by ControlNet

---

## 2. Experimental Setup

### Common Elements

| Parameter | Value |
|-----------|-------|
| Dataset | CMP Facade (606 images) |
| Train/Val/Test Split | 400/100/106 |
| Image Size | 256×256 |
| Batch Size | 4 |
| Learning Rate | 1e-5 |
| Optimizer | AdamW |
| Weight Decay | 1e-2 |
| Epochs | 20 |
| Random Seed | 42 |

### Noise Schedules

**Exponent:** 2.20 (used for both phases)

| Schedule | Type | Effect |
|----------|------|--------|
| Run A | Native β_t | In-distribution training |
| Run B | Ablated β_t^2.20 | Out-of-distribution training |

---

## 3. Training Results

### Phase 2: InstructPix2Pix

See: `Phase_2_InstructPix2Pix/Results/`

**Run 0 (Zero-shot):**
- Results: Check `Run_0/generated_samples/`

**Run A (Native Schedule):**
- Training Logs: `Run_A/training_logs.csv`
- Checkpoints: `Run_A/checkpoint_{{4,8,12,16,20}}/`
- Final Model: `Run_A/final_model/`
- Samples: `Run_A/generated_samples/`

**Run B (Ablated Schedule):**
- Training Logs: `Run_B/training_logs.csv`
- Checkpoints: `Run_B/checkpoint_{{4,8,12,16,20}}/`
- Final Model: `Run_B/final_model/`
- Samples: `Run_B/generated_samples/`

### Phase 3: ControlNet

See: `Phase_3_ControlNet_Implementation/Results/`

**Run 0 (Zero-shot):**
- Results: Check `Run_0/generated_samples/`

**Run A (Native Schedule):**
- Training Logs: `ControlNet_Run_A/training_logs.csv`
- Checkpoints: `ControlNet_Run_A/checkpoint_{{4,8,12,16,20}}/`
- Final Model: `ControlNet_Run_A/final_model/`
- Samples: `ControlNet_Run_A/generated_samples/`

**Run B (Ablated Schedule):**
- Training Logs: `ControlNet_Run_B/training_logs.csv`
- Checkpoints: `ControlNet_Run_B/checkpoint_{{4,8,12,16,20}}/`
- Final Model: `ControlNet_Run_B/final_model/`
- Samples: `ControlNet_Run_B/generated_samples/`

---

## 4. Loss Curve Analysis

See: `Training_Logs_Analysis/loss_comparison_analysis.md`

**Key Metrics:**
- Final loss (Run A vs Run B)
- Convergence rate
- Loss variance
- Schedule impact

---

## 5. Visual Quality Comparison

See: `Visual_Quality_Comparison/visual_quality_report.md`

**Quality Dimensions:**
- Realism (visual quality)
- Semantic accuracy (label fidelity)
- Detail preservation (texture complexity)
- Artifact presence (distortion/blur)

**Sample Organization:**
- `Visual_Quality_Comparison/Phase2_samples/` - InstructPix2Pix outputs
- `Visual_Quality_Comparison/Phase3_samples/` - ControlNet outputs

---

## 6. Schedule Ablation Analysis

See: `Noise_Schedule_Analysis/`

### Hypothesis

Distributional mismatch between training and inference schedules affects output quality:
- **Run A (Native):** Training uses β_t, inference uses same → In-distribution ✓
- **Run B (Ablated):** Training uses β_t^2.20, inference uses β_t → Mismatch ✗

### Expected Results

- Run B should show degraded quality compared to Run A
- Loss may converge differently due to schedule difference
- Visual artifacts may appear in Run B outputs

### Observed Results

[To be filled after experiments]

---

## 7. Findings and Conclusions

See: `Findings_and_Conclusions.md`

### Key Questions Addressed

1. **Architecture Comparison:**
   - Which approach (InstructPix2Pix vs ControlNet) produces better results?
   - How do they differ in handling the semantic-to-photo task?

2. **Schedule Sensitivity:**
   - How much does each architecture suffer from schedule mismatch?
   - Is one more robust than the other?

3. **Generalization:**
   - How well does noise schedule ablation theory hold across both architectures?
   - Are the effects universal or architecture-specific?

### Main Findings

[To be filled after analysis]

---

## 8. Recommendations for Future Work

- [ ] Test with other exponent values (x ∈ (1.5, 3.0))
- [ ] Analyze latent space differences between schedules
- [ ] Test on larger/different datasets
- [ ] Implement hybrid scheduling approaches
- [ ] Study schedule impact on other diffusion architectures

---

## Appendix: File Structure

```
Comparative_Analysis/
├── Phase_2_vs_Phase_3_Comparison.md         ← This file
├── Training_Logs_Analysis/
│   ├── Phase2_vs_Phase3_Loss_Curves.png
│   ├── Phase2_vs_Phase3_Training_Stats.json
│   └── loss_comparison_analysis.md
├── Visual_Quality_Comparison/
│   ├── Phase2_samples/
│   │   ├── Run_0/generated_samples/
│   │   ├── Run_A/generated_samples/
│   │   └── Run_B/generated_samples/
│   ├── Phase3_samples/
│   │   ├── Run_0/generated_samples/
│   │   ├── Run_A/generated_samples/
│   │   └── Run_B/generated_samples/
│   └── visual_quality_report.md
├── Noise_Schedule_Analysis/
│   ├── schedule_comparison.md
│   ├── Run_A_vs_Run_B_impact.md
│   └── distributional_mismatch_analysis.md
└── Findings_and_Conclusions.md
```

---

**Generated:** {timestamp}
**Status:** Analysis Framework Complete - Awaiting Results

"""
    
    report_path = analysis_dir / "Phase_2_vs_Phase_3_Comparison.md"
    with open(report_path, 'w') as f:
        f.write(report.format(timestamp=datetime.now().isoformat()))
    
    print_success(f"Main comparison report created: {report_path}")

def create_findings_document():
    """Create template for findings and conclusions."""
    print_header("CREATING FINDINGS TEMPLATE")
    
    analysis_dir = Path("Comparative_Analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    findings = """# Findings and Conclusions: Phase 2 vs Phase 3

**Analysis Date:** {timestamp}

---

## Summary of Key Findings

### 1. Within-Phase Analysis

#### Phase 2 (InstructPix2Pix)

**Run A vs Run B Comparison:**
- [To be filled after analysis]

**Key Observations:**
- [To be filled]

---

#### Phase 3 (ControlNet)

**Run A vs Run B Comparison:**
- [To be filled after analysis]

**Key Observations:**
- [To be filled]

---

### 2. Cross-Phase Analysis

#### Architecture Performance

**Phase 2 Strengths:**
- [To be filled]

**Phase 2 Weaknesses:**
- [To be filled]

**Phase 3 Strengths:**
- [To be filled]

**Phase 3 Weaknesses:**
- [To be filled]

#### Schedule Sensitivity

**Phase 2 Schedule Impact:**
- Loss degradation (Run B): [To be filled]
- Visual quality impact: [To be filled]
- Robustness to mismatch: [High/Medium/Low]

**Phase 3 Schedule Impact:**
- Loss degradation (Run B): [To be filled]
- Visual quality impact: [To be filled]
- Robustness to mismatch: [High/Medium/Low]

---

## Hypothesis Validation

### Original Hypothesis
"Distributional mismatch between training and inference noise schedules affects output quality"

### Validation Status
- **Phase 2:** [Supported/Contradicted/Partially Supported]
- **Phase 3:** [Supported/Contradicted/Partially Supported]

### Evidence
- [To be filled]

---

## Insights

### Architecture-Specific Observations

[To be filled]

### Schedule-Specific Observations

[To be filled]

### Generalization Potential

[To be filled]

---

## Recommendations

### For Phase 2
- [To be filled]

### For Phase 3
- [To be filled]

### For Future Work
- Test additional exponent values
- Analyze latent space variations
- Compare with other architectures
- Implement adaptive scheduling

---

## Conclusion

[To be filled with comprehensive summary]

---

**Prepared by:** Analysis Script
**Confidence Level:** [To be assessed]

"""
    
    findings_path = analysis_dir / "Findings_and_Conclusions.md"
    with open(findings_path, 'w') as f:
        f.write(findings.format(timestamp=datetime.now().isoformat()))
    
    print_success(f"Findings template created: {findings_path}")

def main():
    """Main analysis execution."""
    print_header("COMPARATIVE ANALYSIS: PHASE 2 vs PHASE 3")
    
    print(f"{Colors.YELLOW}Step 1: Analyzing Training Logs{Colors.END}")
    analyze_training_logs()
    
    print(f"\n{Colors.YELLOW}Step 2: Creating Loss Comparison Analysis{Colors.END}")
    create_loss_comparison_analysis()
    
    print(f"\n{Colors.YELLOW}Step 3: Organizing Sample Images{Colors.END}")
    organize_sample_images()
    
    print(f"\n{Colors.YELLOW}Step 4: Creating Visual Quality Report{Colors.END}")
    create_visual_quality_report()
    
    print(f"\n{Colors.YELLOW}Step 5: Generating Main Comparison Report{Colors.END}")
    create_main_comparison_report()
    
    print(f"\n{Colors.YELLOW}Step 6: Creating Findings Template{Colors.END}")
    create_findings_document()
    
    print_header("ANALYSIS FRAMEWORK COMPLETE")
    
    print(f"\n{Colors.GREEN}All analysis documents created!{Colors.END}")
    print(f"\n{Colors.BLUE}Next Steps:{Colors.END}")
    print(f"  1. Review analysis reports in: {Colors.BLUE}Comparative_Analysis/{Colors.END}")
    print(f"  2. Compare sample images in: {Colors.BLUE}Comparative_Analysis/Visual_Quality_Comparison/{Colors.END}")
    print(f"  3. Fill in analysis templates with observations")
    print(f"  4. Write final conclusions in: {Colors.BLUE}Comparative_Analysis/Findings_and_Conclusions.md{Colors.END}")
    
    print(f"\n{Colors.GREEN}Analysis framework ready! ✓{Colors.END}\n")

if __name__ == "__main__":
    main()
