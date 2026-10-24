#!/bin/bash

# Comparative Analysis Script
# Run this AFTER both Phase 2 and Phase 3 are complete
# Organizes results and generates comparison reports

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Comparative Analysis: Phase 2 vs Phase 3            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Check if Phase 2 results exist
echo -e "\n${YELLOW}[CHECK] Verifying Phase 2 results...${NC}"
if [ ! -d "Phase_2_InstructPix2Pix/Results/Run_A" ] || [ ! -f "Phase_2_InstructPix2Pix/Results/Run_A/training_logs.csv" ]; then
    echo -e "${RED}✗ Phase 2 Run A not found${NC}"
    echo -e "${YELLOW}  (Run Phase 2 first: bash RUN_PHASE2.sh)${NC}"
    EXIT_PHASE2=1
else
    echo -e "${GREEN}✓ Phase 2 Run A found${NC}"
    EXIT_PHASE2=0
fi

# Check if Phase 3 results exist
echo -e "\n${YELLOW}[CHECK] Verifying Phase 3 results...${NC}"
if [ ! -d "Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A" ] || [ ! -f "Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A/training_logs.csv" ]; then
    echo -e "${RED}✗ Phase 3 Run A not found${NC}"
    echo -e "${YELLOW}  (Run Phase 3 first: bash RUN_PHASE3.sh)${NC}"
    EXIT_PHASE3=1
else
    echo -e "${GREEN}✓ Phase 3 Run A found${NC}"
    EXIT_PHASE3=0
fi

# Exit if either phase is missing
if [ $EXIT_PHASE2 -eq 1 ] || [ $EXIT_PHASE3 -eq 1 ]; then
    echo -e "\n${RED}Cannot proceed with analysis - missing experiment results${NC}"
    echo -e "${YELLOW}Please run:${NC}"
    echo -e "  bash RUN_PHASE2.sh"
    echo -e "  bash RUN_PHASE3.sh"
    echo -e "${YELLOW}Then run this analysis script again${NC}"
    exit 1
fi

# Create analysis structure
echo -e "\n${YELLOW}[SETUP] Creating analysis directory structure...${NC}"
mkdir -p Comparative_Analysis/Training_Logs_Analysis
mkdir -p Comparative_Analysis/Visual_Quality_Comparison/Phase2_samples/{Run_0,Run_A,Run_B}
mkdir -p Comparative_Analysis/Visual_Quality_Comparison/Phase3_samples/{Run_0,Run_A,Run_B}
mkdir -p Comparative_Analysis/Noise_Schedule_Analysis
echo -e "${GREEN}✓ Analysis structure created${NC}"

# Copy training logs to analysis folder
echo -e "\n${YELLOW}[ORGANIZE] Copying training logs...${NC}"
cp -v Phase_2_InstructPix2Pix/Results/Run_A/training_logs.csv Comparative_Analysis/Training_Logs_Analysis/Phase2_Run_A_logs.csv 2>/dev/null || true
cp -v Phase_2_InstructPix2Pix/Results/Run_B/training_logs.csv Comparative_Analysis/Training_Logs_Analysis/Phase2_Run_B_logs.csv 2>/dev/null || true
cp -v Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A/training_logs.csv Comparative_Analysis/Training_Logs_Analysis/Phase3_Run_A_logs.csv 2>/dev/null || true
cp -v Phase_3_ControlNet_Implementation/Results/ControlNet_Run_B/training_logs.csv Comparative_Analysis/Training_Logs_Analysis/Phase3_Run_B_logs.csv 2>/dev/null || true
echo -e "${GREEN}✓ Training logs organized${NC}"

# Copy generated samples
echo -e "\n${YELLOW}[ORGANIZE] Copying generated samples...${NC}"
[ -d "Phase_2_InstructPix2Pix/Results/Run_0/generated_samples" ] && cp -rv Phase_2_InstructPix2Pix/Results/Run_0/generated_samples/* Comparative_Analysis/Visual_Quality_Comparison/Phase2_samples/Run_0/ 2>/dev/null || true
[ -d "Phase_2_InstructPix2Pix/Results/Run_A/generated_samples" ] && cp -rv Phase_2_InstructPix2Pix/Results/Run_A/generated_samples/* Comparative_Analysis/Visual_Quality_Comparison/Phase2_samples/Run_A/ 2>/dev/null || true
[ -d "Phase_2_InstructPix2Pix/Results/Run_B/generated_samples" ] && cp -rv Phase_2_InstructPix2Pix/Results/Run_B/generated_samples/* Comparative_Analysis/Visual_Quality_Comparison/Phase2_samples/Run_B/ 2>/dev/null || true
[ -d "Phase_3_ControlNet_Implementation/Results/ControlNet_Run_0/generated_samples" ] && cp -rv Phase_3_ControlNet_Implementation/Results/ControlNet_Run_0/generated_samples/* Comparative_Analysis/Visual_Quality_Comparison/Phase3_samples/Run_0/ 2>/dev/null || true
[ -d "Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A/generated_samples" ] && cp -rv Phase_3_ControlNet_Implementation/Results/ControlNet_Run_A/generated_samples/* Comparative_Analysis/Visual_Quality_Comparison/Phase3_samples/Run_A/ 2>/dev/null || true
[ -d "Phase_3_ControlNet_Implementation/Results/ControlNet_Run_B/generated_samples" ] && cp -rv Phase_3_ControlNet_Implementation/Results/ControlNet_Run_B/generated_samples/* Comparative_Analysis/Visual_Quality_Comparison/Phase3_samples/Run_B/ 2>/dev/null || true
echo -e "${GREEN}✓ Sample images organized${NC}"

# Generate analysis summary
echo -e "\n${YELLOW}[ANALYZE] Generating training logs analysis...${NC}"

python3 << 'PYTHON_ANALYSIS'
import pandas as pd
import os

# Read training logs
phase2_a = pd.read_csv('Comparative_Analysis/Training_Logs_Analysis/Phase2_Run_A_logs.csv')
phase2_b = pd.read_csv('Comparative_Analysis/Training_Logs_Analysis/Phase2_Run_B_logs.csv')
phase3_a = pd.read_csv('Comparative_Analysis/Training_Logs_Analysis/Phase3_Run_A_logs.csv')
phase3_b = pd.read_csv('Comparative_Analysis/Training_Logs_Analysis/Phase3_Run_B_logs.csv')

# Generate analysis report
analysis = """# Training Logs Analysis: Phase 2 vs Phase 3

## Summary Statistics

### Phase 2: InstructPix2Pix

#### Run A (Native Schedule)
- Epochs trained: {p2a_epochs}
- Final loss: {p2a_final_loss:.6f}
- Min loss: {p2a_min_loss:.6f}
- Max loss: {p2a_max_loss:.6f}
- Avg loss: {p2a_avg_loss:.6f}

#### Run B (Ablated Schedule - x=2.20)
- Epochs trained: {p2b_epochs}
- Final loss: {p2b_final_loss:.6f}
- Min loss: {p2b_min_loss:.6f}
- Max loss: {p2b_max_loss:.6f}
- Avg loss: {p2b_avg_loss:.6f}

#### Run A vs Run B Comparison
- Final loss difference: {p2_diff:.6f}
- Relative increase (B vs A): {p2_pct:.2f}%
- **Interpretation**: Run B shows {'higher' if phase2_b['train_loss'].iloc[-1] > phase2_a['train_loss'].iloc[-1] else 'lower'} final loss

---

### Phase 3: ControlNet

#### Run A (Native Schedule)
- Epochs trained: {p3a_epochs}
- Final loss: {p3a_final_loss:.6f}
- Min loss: {p3a_min_loss:.6f}
- Max loss: {p3a_max_loss:.6f}
- Avg loss: {p3a_avg_loss:.6f}

#### Run B (Ablated Schedule - x=2.20)
- Epochs trained: {p3b_epochs}
- Final loss: {p3b_final_loss:.6f}
- Min loss: {p3b_min_loss:.6f}
- Max loss: {p3b_max_loss:.6f}
- Avg loss: {p3b_avg_loss:.6f}

#### Run A vs Run B Comparison
- Final loss difference: {p3_diff:.6f}
- Relative increase (B vs A): {p3_pct:.2f}%
- **Interpretation**: Run B shows {'higher' if phase3_b['train_loss'].iloc[-1] > phase3_a['train_loss'].iloc[-1] else 'lower'} final loss

---

## Cross-Phase Comparison

### Architecture Comparison (Run A vs Run A)
- Phase 2 Final Loss: {p2a_final_loss:.6f}
- Phase 3 Final Loss: {p3a_final_loss:.6f}
- Difference: {abs(p3a_final_loss - p2a_final_loss):.6f}

### Schedule Impact Comparison
- Phase 2 (Run B - Run A): {phase2_b['train_loss'].iloc[-1] - phase2_a['train_loss'].iloc[-1]:.6f}
- Phase 3 (Run B - Run A): {phase3_b['train_loss'].iloc[-1] - phase3_a['train_loss'].iloc[-1]:.6f}

---

## Key Observations

1. **Native Schedule (Run A)**
   - Both phases show stable training with Run A
   - Final loss values indicate convergence

2. **Ablated Schedule (Run B)**
   - Phase 2 Run B behavior: Loss value is {'higher' if phase2_b['train_loss'].iloc[-1] > phase2_a['train_loss'].iloc[-1] else 'lower'} than Run A
   - Phase 3 Run B behavior: Loss value is {'higher' if phase3_b['train_loss'].iloc[-1] > phase3_a['train_loss'].iloc[-1] else 'lower'} than Run A

3. **Distributional Mismatch Effect**
   - The ablated schedule creates a mismatch between training and inference
   - Effect visible in loss curves and final output quality

---

## Detailed Logs

### Phase 2 Training Logs
```
{p2a_summary}
```

### Phase 3 Training Logs  
```
{p3a_summary}
```

Generated: $(date)
"""

# Format the report
report = analysis.format(
    p2a_epochs=len(phase2_a),
    p2a_final_loss=phase2_a['train_loss'].iloc[-1],
    p2a_min_loss=phase2_a['train_loss'].min(),
    p2a_max_loss=phase2_a['train_loss'].max(),
    p2a_avg_loss=phase2_a['train_loss'].mean(),
    
    p2b_epochs=len(phase2_b),
    p2b_final_loss=phase2_b['train_loss'].iloc[-1],
    p2b_min_loss=phase2_b['train_loss'].min(),
    p2b_max_loss=phase2_b['train_loss'].max(),
    p2b_avg_loss=phase2_b['train_loss'].mean(),
    
    p2_diff=abs(phase2_b['train_loss'].iloc[-1] - phase2_a['train_loss'].iloc[-1]),
    p2_pct=(phase2_b['train_loss'].iloc[-1] / phase2_a['train_loss'].iloc[-1] - 1) * 100 if phase2_a['train_loss'].iloc[-1] != 0 else 0,
    
    p3a_epochs=len(phase3_a),
    p3a_final_loss=phase3_a['train_loss'].iloc[-1],
    p3a_min_loss=phase3_a['train_loss'].min(),
    p3a_max_loss=phase3_a['train_loss'].max(),
    p3a_avg_loss=phase3_a['train_loss'].mean(),
    
    p3b_epochs=len(phase3_b),
    p3b_final_loss=phase3_b['train_loss'].iloc[-1],
    p3b_min_loss=phase3_b['train_loss'].min(),
    p3b_max_loss=phase3_b['train_loss'].max(),
    p3b_avg_loss=phase3_b['train_loss'].mean(),
    
    p3_diff=abs(phase3_b['train_loss'].iloc[-1] - phase3_a['train_loss'].iloc[-1]),
    p3_pct=(phase3_b['train_loss'].iloc[-1] / phase3_a['train_loss'].iloc[-1] - 1) * 100 if phase3_a['train_loss'].iloc[-1] != 0 else 0,
    
    p2a_summary=phase2_a.to_string(),
    p3a_summary=phase3_a.to_string(),
)

# Save report
with open('Comparative_Analysis/Training_Logs_Analysis/loss_comparison_analysis.md', 'w') as f:
    f.write(report)

print("✓ Analysis report generated")

PYTHON_ANALYSIS

echo -e "${GREEN}✓ Training logs analysis generated${NC}"

# Create main comparison report
echo -e "\n${YELLOW}[GENERATE] Creating main comparison report...${NC}"

cat > Comparative_Analysis/Phase_2_vs_Phase_3_Comparison.md << 'EOF'
# Comparative Analysis: Phase 2 (InstructPix2Pix) vs Phase 3 (ControlNet)

**Analysis Date:** $(date)  
**Status:** Complete

---

## Executive Summary

This document compares the noise schedule ablation experiment across two different architectures:
- **Phase 2:** InstructPix2Pix (text-conditioned image-to-image)
- **Phase 3:** ControlNet (semantic-conditioned image-to-image)

Both phases follow the same experimental design with noise schedule variations.

---

## 📊 Key Findings

### Training Dynamics

See detailed analysis in: `Comparative_Analysis/Training_Logs_Analysis/loss_comparison_analysis.md`

### Visual Quality Comparison

Compare sample outputs in: `Comparative_Analysis/Visual_Quality_Comparison/`

Structure:
```
Phase2_samples/
├── Run_0/     (zero-shot baseline)
├── Run_A/     (native schedule - expected: good quality)
└── Run_B/     (ablated schedule - expected: degraded quality)

Phase3_samples/
├── Run_0/     (zero-shot baseline)
├── Run_A/     (native schedule - expected: good quality)
└── Run_B/     (ablated schedule - expected: degraded quality)
```

### Noise Schedule Impact

See detailed analysis in: `Comparative_Analysis/Noise_Schedule_Analysis/`

**Hypothesis:** The ablated schedule (β_t^2.20) creates a distributional mismatch during inference, resulting in degraded output quality.

**Expected Results:**
- Run A: Good convergence, good visual quality
- Run B: May show different loss patterns, potential visual degradation

---

## 📁 Analysis Structure

```
Comparative_Analysis/
├── Training_Logs_Analysis/
│   ├── Phase2_Run_A_logs.csv
│   ├── Phase2_Run_B_logs.csv
│   ├── Phase3_Run_A_logs.csv
│   ├── Phase3_Run_B_logs.csv
│   └── loss_comparison_analysis.md
├── Visual_Quality_Comparison/
│   ├── Phase2_samples/
│   │   ├── Run_0/
│   │   ├── Run_A/
│   │   └── Run_B/
│   ├── Phase3_samples/
│   │   ├── Run_0/
│   │   ├── Run_A/
│   │   └── Run_B/
│   └── visual_quality_report.md (TODO)
├── Noise_Schedule_Analysis/
│   ├── schedule_comparison.md (TODO)
│   └── Run_A_vs_Run_B_impact.md (TODO)
├── Phase_2_vs_Phase_3_Comparison.md (this file)
└── Findings_and_Conclusions.md (TODO)
```

---

## 🔍 What to Analyze

### 1. Training Logs Comparison

**Questions:**
- How do loss curves differ between the two architectures?
- Is the schedule ablation effect more pronounced in one architecture?
- Do both show the expected pattern (Run B higher loss than Run A)?

**Files:** `Comparative_Analysis/Training_Logs_Analysis/`

### 2. Visual Quality Assessment

**Questions:**
- Which architecture produces better zero-shot results (Run 0)?
- How much does the schedule ablation affect visual quality in each architecture?
- Are failure modes similar or different?

**Instructions:**
1. Navigate to `Comparative_Analysis/Visual_Quality_Comparison/`
2. Compare corresponding runs:
   - Phase2_samples/Run_A/ vs Phase3_samples/Run_A/
   - Phase2_samples/Run_B/ vs Phase3_samples/Run_B/
3. Evaluate: realism, semantic accuracy, artifacts

**Files:** `Comparative_Analysis/Visual_Quality_Comparison/`

### 3. Schedule Impact Mechanism

**Questions:**
- What is the precise mechanism by which the ablated schedule affects performance?
- Is the impact consistent across both architectures?
- Can we predict behavior based on schedule mathematics?

**Files:** `Comparative_Analysis/Noise_Schedule_Analysis/`

---

## 📝 TODO Items

Complete the following analysis documents:

- [ ] `Comparative_Analysis/Training_Logs_Analysis/loss_comparison_analysis.md` (auto-generated)
- [ ] `Comparative_Analysis/Visual_Quality_Comparison/visual_quality_report.md` - Manual quality assessment
- [ ] `Comparative_Analysis/Noise_Schedule_Analysis/schedule_comparison.md` - Technical analysis
- [ ] `Comparative_Analysis/Noise_Schedule_Analysis/Run_A_vs_Run_B_impact.md` - Impact analysis
- [ ] `Comparative_Analysis/Findings_and_Conclusions.md` - Summary findings

---

## 🎯 Next Steps

1. **Review training logs:** Read `loss_comparison_analysis.md`
2. **Examine samples:** Compare visual outputs by run and architecture
3. **Document findings:** Fill in the TODO sections above
4. **Write conclusions:** Complete `Findings_and_Conclusions.md`

---

**Status:** Analysis structure created, training logs compiled  
**Next:** Review results and fill in detailed comparisons

EOF

echo -e "${GREEN}✓ Main comparison report created${NC}"

# Create structure info
echo -e "\n${YELLOW}[INFO] Creating analysis structure guide...${NC}"

cat > Comparative_Analysis/ANALYSIS_GUIDE.md << 'EOF'
# Analysis Guide: How to Use This Folder

**Created:** $(date)

---

## Folder Structure

```
Comparative_Analysis/
├── Training_Logs_Analysis/
│   ├── Phase2_Run_A_logs.csv          ← InstructPix2Pix Run A training logs
│   ├── Phase2_Run_B_logs.csv          ← InstructPix2Pix Run B training logs
│   ├── Phase3_Run_A_logs.csv          ← ControlNet Run A training logs
│   ├── Phase3_Run_B_logs.csv          ← ControlNet Run B training logs
│   └── loss_comparison_analysis.md    ← Auto-generated analysis
│
├── Visual_Quality_Comparison/
│   ├── Phase2_samples/
│   │   ├── Run_0/                     ← InstructPix2Pix zero-shot
│   │   ├── Run_A/                     ← InstructPix2Pix native schedule
│   │   └── Run_B/                     ← InstructPix2Pix ablated schedule
│   ├── Phase3_samples/
│   │   ├── Run_0/                     ← ControlNet zero-shot
│   │   ├── Run_A/                     ← ControlNet native schedule
│   │   └── Run_B/                     ← ControlNet ablated schedule
│   └── visual_quality_report.md       ← (Create this with your analysis)
│
├── Noise_Schedule_Analysis/
│   ├── schedule_comparison.md         ← (Create this with technical details)
│   └── Run_A_vs_Run_B_impact.md       ← (Create this with impact analysis)
│
├── Phase_2_vs_Phase_3_Comparison.md   ← Main comparison report
├── Findings_and_Conclusions.md        ← (Create this with summary)
└── ANALYSIS_GUIDE.md                  ← This file

```

---

## How to Analyze

### 1. Training Logs Analysis

**File:** `Training_Logs_Analysis/loss_comparison_analysis.md`

- Auto-generated summary statistics
- Compares loss curves between phases and runs
- Shows convergence behavior

**Action:** Review and interpret the numbers

### 2. Visual Quality Comparison

**Folder:** `Visual_Quality_Comparison/`

**What to do:**
1. Open Phase2_samples/Run_A/ and Phase3_samples/Run_A/ side-by-side
2. Compare the generated facade images
3. Note differences in:
   - Realism and photorealism
   - Semantic accuracy (label → output correspondence)
   - Presence of artifacts or distortions
   - Detail preservation

4. Repeat for Run_B samples to evaluate schedule impact

**Create:** `visual_quality_report.md` with your findings

### 3. Schedule Impact Analysis

**What to analyze:**
- How much does Run B differ from Run A in each architecture?
- Is the impact visible in training loss?
- Is it visible in generated samples?
- Which architecture is more robust to schedule mismatch?

**Create:** 
- `Noise_Schedule_Analysis/schedule_comparison.md` - Technical details
- `Noise_Schedule_Analysis/Run_A_vs_Run_B_impact.md` - Impact findings

### 4. Write Conclusions

**Create:** `Findings_and_Conclusions.md`

**Include:**
- Key findings from the three comparisons above
- Hypothesis confirmation/rejection
- Architecture-specific insights
- Recommendations for future work

---

## Expected Patterns

### If Hypothesis is Correct

**Phase 2 & Phase 3 should both show:**
- Run A: Stable training, good final loss, good output quality
- Run B: Different loss pattern, higher final loss, degraded output quality
- The degradation is due to distributional mismatch (schedule)

### Comparison Insights

- If both architectures show similar patterns → Schedule is a robust effect
- If patterns differ → Architecture plays a role in schedule robustness
- If one is much more affected → One architecture is more sensitive to schedule

---

## Tools & Resources

### Viewing CSV Files
```bash
cat Training_Logs_Analysis/Phase2_Run_A_logs.csv
```

### Comparing Images
```bash
# View samples side-by-side (command-line)
ls Visual_Quality_Comparison/Phase2_samples/Run_A/
ls Visual_Quality_Comparison/Phase3_samples/Run_A/
```

### Generating Plots (Python)
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load logs
p2a = pd.read_csv('Training_Logs_Analysis/Phase2_Run_A_logs.csv')
p2b = pd.read_csv('Training_Logs_Analysis/Phase2_Run_B_logs.csv')

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(p2a['epoch'], p2a['train_loss'], label='Run A (native)', marker='o')
plt.plot(p2b['epoch'], p2b['train_loss'], label='Run B (ablated)', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Phase 2: InstructPix2Pix Loss Comparison')
plt.legend()
plt.savefig('training_loss_comparison.png')
```

---

## Checklist

Complete analysis in this order:

- [ ] Read `loss_comparison_analysis.md`
- [ ] Review visual samples in `Visual_Quality_Comparison/`
- [ ] Create `visual_quality_report.md`
- [ ] Create `schedule_comparison.md`
- [ ] Create `Run_A_vs_Run_B_impact.md`
- [ ] Write `Findings_and_Conclusions.md`
- [ ] Update `Phase_2_vs_Phase_3_Comparison.md` with final conclusions

---

**Ready to analyze!** 🚀

EOF

echo -e "${GREEN}✓ Analysis guide created${NC}"

# Summary
echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ ANALYSIS PREPARATION COMPLETE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Results Organized In:${NC}"
echo -e "  ${BLUE}Comparative_Analysis/${NC}"

echo -e "\n${YELLOW}Key Files:${NC}"
echo -e "  Main Report:      ${BLUE}Comparative_Analysis/Phase_2_vs_Phase_3_Comparison.md${NC}"
echo -e "  Analysis Guide:   ${BLUE}Comparative_Analysis/ANALYSIS_GUIDE.md${NC}"
echo -e "  Training Logs:    ${BLUE}Comparative_Analysis/Training_Logs_Analysis/{{NC}"
echo -e "  Sample Images:    ${BLUE}Comparative_Analysis/Visual_Quality_Comparison/${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Review: ${BLUE}cat Comparative_Analysis/Training_Logs_Analysis/loss_comparison_analysis.md${NC}"
echo -e "  2. Compare: Review sample images in ${BLUE}Visual_Quality_Comparison/${NC}"
echo -e "  3. Analyze: Follow guide in ${BLUE}Comparative_Analysis/ANALYSIS_GUIDE.md${NC}"
echo -e "  4. Document: Fill in TODO items"

echo -e "\n${GREEN}Analysis structure ready! ✓${NC}\n"
