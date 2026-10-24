#!/bin/bash

# Comprehensive execution script for Phase 2 & Phase 3 experiments
# This script runs all experiments and organizes results

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET_ROOT="./data/cmp_facade_complete"
SEED=42
EPOCHS=20
BATCH_SIZE=4

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  PHASE 2 & 3 EXPERIMENT RUNNER             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Verify dataset
echo -e "\n${YELLOW}Verifying dataset...${NC}"
IMAGES_COUNT=$(ls $DATASET_ROOT/images/*.jpg 2>/dev/null | wc -l)
LABELS_COUNT=$(ls $DATASET_ROOT/labels/*.png 2>/dev/null | wc -l)

if [ "$IMAGES_COUNT" -ne 606 ] || [ "$LABELS_COUNT" -ne 606 ]; then
    echo -e "${RED}✗ Dataset incomplete: $IMAGES_COUNT images, $LABELS_COUNT labels (need 606)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dataset verified: 606 images + 606 labels${NC}"

# Verify scripts exist
echo -e "\n${YELLOW}Verifying scripts...${NC}"
if [ ! -f "Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py" ]; then
    echo -e "${RED}✗ Phase 2 script not found${NC}"
    exit 1
fi
if [ ! -f "Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py" ]; then
    echo -e "${RED}✗ Phase 3 script not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All scripts found${NC}"

# Ask user if they want to rerun Phase 2
echo -e "\n${YELLOW}Phase 2 Results Status:${NC}"
if [ -d "Phase_2_InstructPix2Pix/Results/Run_A" ]; then
    echo -e "${BLUE}  Previous Phase 2 results exist${NC}"
    read -p "  Rerun Phase 2? (y/n, default: n): " RERUN_PHASE2
    RERUN_PHASE2=${RERUN_PHASE2:-n}
else
    echo -e "${YELLOW}  No previous Phase 2 results found - will run${NC}"
    RERUN_PHASE2="y"
fi

# PHASE 2: InstructPix2Pix (Optional Rerun)
if [ "$RERUN_PHASE2" = "y" ]; then
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}PHASE 2: InstructPix2Pix - Starting${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    
    # Run 0: Zero-shot baseline
    echo -e "\n${YELLOW}Run 0: Zero-shot Baseline${NC}"
    python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
        --run 0 \
        --dataset_root $DATASET_ROOT \
        --output_dir ./Phase_2_InstructPix2Pix/Results \
        --seed $SEED
    echo -e "${GREEN}✓ Phase 2 Run 0 Complete${NC}"
    
    # Run A: Native schedule
    echo -e "\n${YELLOW}Run A: Fine-tuning with Native Schedule${NC}"
    python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
        --run A \
        --dataset_root $DATASET_ROOT \
        --output_dir ./Phase_2_InstructPix2Pix/Results \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seed $SEED
    echo -e "${GREEN}✓ Phase 2 Run A Complete${NC}"
    
    # Run B: Ablated schedule
    echo -e "\n${YELLOW}Run B: Fine-tuning with Ablated Schedule (x=2.20)${NC}"
    python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
        --run B \
        --dataset_root $DATASET_ROOT \
        --output_dir ./Phase_2_InstructPix2Pix/Results \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seed $SEED
    echo -e "${GREEN}✓ Phase 2 Run B Complete${NC}"
    
    echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ PHASE 2 COMPLETE${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
else
    echo -e "${BLUE}Skipping Phase 2 rerun (using existing results)${NC}"
fi

# PHASE 3: ControlNet (Always run new)
echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 3: ControlNet - Starting${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

# Run 0: Zero-shot baseline
echo -e "\n${YELLOW}Run 0: Zero-shot Baseline${NC}"
python3 Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py \
    --run 0 \
    --dataset_root $DATASET_ROOT \
    --output_dir ./Phase_3_ControlNet_Implementation/Results \
    --seed $SEED
echo -e "${GREEN}✓ Phase 3 Run 0 Complete${NC}"

# Run A: Native schedule
echo -e "\n${YELLOW}Run A: Fine-tuning with Native Schedule${NC}"
python3 Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py \
    --run A \
    --dataset_root $DATASET_ROOT \
    --output_dir ./Phase_3_ControlNet_Implementation/Results \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seed $SEED
echo -e "${GREEN}✓ Phase 3 Run A Complete${NC}"

# Run B: Ablated schedule
echo -e "\n${YELLOW}Run B: Fine-tuning with Ablated Schedule (x=2.20)${NC}"
python3 Phase_3_ControlNet_Implementation/Scripts/finetune_controlnet.py \
    --run B \
    --dataset_root $DATASET_ROOT \
    --output_dir ./Phase_3_ControlNet_Implementation/Results \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seed $SEED
echo -e "${GREEN}✓ Phase 3 Run B Complete${NC}"

echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ PHASE 3 COMPLETE${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"

# Summary
echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ ALL EXPERIMENTS COMPLETE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Results Location:${NC}"
echo -e "  Phase 2: ${BLUE}Phase_2_InstructPix2Pix/Results/${NC}"
echo -e "  Phase 3: ${BLUE}Phase_3_ControlNet_Implementation/Results/${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Run analysis script: bash RUN_ANALYSIS.sh"
echo -e "  2. Review results in: ${BLUE}Comparative_Analysis/${NC}"
echo -e "  3. Read reports: ${BLUE}Comparative_Analysis/Phase_2_vs_Phase_3_Comparison.md${NC}"

echo -e "\n${GREEN}All done! ✓${NC}\n"
