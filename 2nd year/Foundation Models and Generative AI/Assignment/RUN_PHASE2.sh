#!/bin/bash

# Phase 2: InstructPix2Pix Independent Execution Script
# Run this script separately to execute Phase 2 experiments

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DATASET_ROOT="./data/cmp_facade_complete"
SEED=42
EPOCHS=20
BATCH_SIZE=4

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            PHASE 2: InstructPix2Pix Experiments              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Verify dataset
echo -e "\n${YELLOW}[SETUP] Verifying dataset...${NC}"
IMAGES_COUNT=$(ls $DATASET_ROOT/images/*.jpg 2>/dev/null | wc -l)
LABELS_COUNT=$(ls $DATASET_ROOT/labels/*.png 2>/dev/null | wc -l)

if [ "$IMAGES_COUNT" -ne 606 ] || [ "$LABELS_COUNT" -ne 606 ]; then
    echo -e "${RED}✗ Dataset incomplete: $IMAGES_COUNT images, $LABELS_COUNT labels (need 606)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dataset verified: 606 images + 606 labels${NC}"

# Verify script exists
echo -e "\n${YELLOW}[SETUP] Verifying Phase 2 script...${NC}"
if [ ! -f "Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py" ]; then
    echo -e "${RED}✗ Phase 2 script not found at Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Phase 2 script found${NC}"

# Create output directory
mkdir -p Phase_2_InstructPix2Pix/Results

echo -e "\n${YELLOW}Configuration:${NC}"
echo -e "  Dataset: ${BLUE}$DATASET_ROOT${NC}"
echo -e "  Output:  ${BLUE}Phase_2_InstructPix2Pix/Results${NC}"
echo -e "  Epochs:  ${BLUE}$EPOCHS${NC}"
echo -e "  Batch Size: ${BLUE}$BATCH_SIZE${NC}"
echo -e "  Seed:    ${BLUE}$SEED${NC}"

# Ask which runs to execute
echo -e "\n${YELLOW}Select which runs to execute:${NC}"
echo -e "  1) All runs (Run 0, A, B) - Recommended"
echo -e "  2) Quick test (Run 0 only)"
echo -e "  3) Run A only (native schedule)"
echo -e "  4) Run B only (ablated schedule)"
read -p "  Enter choice (1-4, default: 1): " RUN_CHOICE
RUN_CHOICE=${RUN_CHOICE:-1}

# PHASE 2: InstructPix2Pix Experiments
echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 2: InstructPix2Pix - Executing${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

case $RUN_CHOICE in
    1|"")
        # All runs
        
        # Run 0: Zero-shot baseline
        echo -e "\n${YELLOW}[1/3] Run 0: Zero-shot Baseline${NC}"
        echo -e "      (no fine-tuning, just inference on pretrained model)"
        python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
            --run 0 \
            --dataset_root $DATASET_ROOT \
            --output_dir ./Phase_2_InstructPix2Pix/Results \
            --seed $SEED
        echo -e "${GREEN}✓ Run 0 Complete${NC}"
        
        # Run A: Native schedule
        echo -e "\n${YELLOW}[2/3] Run A: Fine-tuning with Native Schedule${NC}"
        echo -e "      (training with standard Stable Diffusion schedule)"
        echo -e "      Expected time: ~1 hour on GPU"
        python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
            --run A \
            --dataset_root $DATASET_ROOT \
            --output_dir ./Phase_2_InstructPix2Pix/Results \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --seed $SEED
        echo -e "${GREEN}✓ Run A Complete${NC}"
        
        # Run B: Ablated schedule
        echo -e "\n${YELLOW}[3/3] Run B: Fine-tuning with Ablated Schedule (x=2.20)${NC}"
        echo -e "      (training with power-transformed schedule: β_t^2.20)"
        echo -e "      Expected time: ~1 hour on GPU"
        python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
            --run B \
            --dataset_root $DATASET_ROOT \
            --output_dir ./Phase_2_InstructPix2Pix/Results \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --seed $SEED
        echo -e "${GREEN}✓ Run B Complete${NC}"
        ;;
    2)
        # Quick test
        echo -e "\n${YELLOW}Run 0: Zero-shot Baseline (Quick Test)${NC}"
        python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
            --run 0 \
            --dataset_root $DATASET_ROOT \
            --output_dir ./Phase_2_InstructPix2Pix/Results \
            --seed $SEED
        echo -e "${GREEN}✓ Quick Test Complete${NC}"
        ;;
    3)
        # Run A only
        echo -e "\n${YELLOW}Run A: Fine-tuning with Native Schedule${NC}"
        python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
            --run A \
            --dataset_root $DATASET_ROOT \
            --output_dir ./Phase_2_InstructPix2Pix/Results \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --seed $SEED
        echo -e "${GREEN}✓ Run A Complete${NC}"
        ;;
    4)
        # Run B only
        echo -e "\n${YELLOW}Run B: Fine-tuning with Ablated Schedule${NC}"
        python3 Phase_2_InstructPix2Pix/Scripts/finetune_instructpix2pix.py \
            --run B \
            --dataset_root $DATASET_ROOT \
            --output_dir ./Phase_2_InstructPix2Pix/Results \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --seed $SEED
        echo -e "${GREEN}✓ Run B Complete${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Summary
echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ PHASE 2 EXECUTION COMPLETE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Results Location:${NC}"
echo -e "  ${BLUE}Phase_2_InstructPix2Pix/Results/${NC}"

echo -e "\n${YELLOW}Output Files:${NC}"
echo -e "  Training logs:     ${BLUE}Run_A/training_logs.csv, Run_B/training_logs.csv${NC}"
echo -e "  Checkpoints:       ${BLUE}Run_A/checkpoint_{4,8,12,16,20}/, Run_B/checkpoint_{4,8,12,16,20}/${NC}"
echo -e "  Final models:      ${BLUE}Run_A/final_model/, Run_B/final_model/${NC}"
echo -e "  Generated samples: ${BLUE}Run_0/sample_*.png, Run_A/sample_*.png, Run_B/sample_*.png${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Review results in Phase_2_InstructPix2Pix/Results/"
echo -e "  2. When ready, run Phase 3: ${BLUE}bash RUN_PHASE3.sh${NC}"
echo -e "  3. After both phases, run analysis: ${BLUE}bash RUN_ANALYSIS.sh${NC}"

echo -e "\n${GREEN}Phase 2 Done! ✓${NC}\n"
