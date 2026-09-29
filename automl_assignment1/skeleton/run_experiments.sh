#!/bin/bash
# run_experiments.sh

SEEDS=(17 42 100)
DATASETS=("breast-w" "credit-g" "phoneme" "electricity" "covertype")
PROFILE="course" 

for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Starting $dataset with seed $seed..."
        
        # Run forest methods
        python experiment.py \
            --dataset "$dataset" \
            --profile "$PROFILE" \
            --methods default random smbo hyperband \
            --seed "$seed"
            
        # Run foundation model exclusively on the largest dataset
        if [ "$dataset" = "covertype" ]; then
            echo "Starting foundation model for $dataset with seed $seed..."
            python experiment.py \
                --dataset "$dataset" \
                --profile "$PROFILE" \
                --methods foundation \
                --seed "$seed"
        fi
    done
done
