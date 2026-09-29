$SEEDS = @(17, 42, 100)
$DATASETS = @("breast-w", "credit-g", "phoneme", "electricity", "covertype")
$PROFILE = "course"

foreach ($seed in $SEEDS) {
    foreach ($dataset in $DATASETS) {
        Write-Host "Starting $dataset with seed $seed..."
        
        # Run forest methods
        python experiment.py --dataset $dataset --profile $PROFILE --methods default random smbo hyperband --seed $seed
        
        # Run foundation model exclusively on the largest dataset
        if ($dataset -eq "covertype") {
            Write-Host "Starting foundation model for $dataset with seed $seed..."
            python experiment.py --dataset $dataset --profile $PROFILE --methods foundation --seed $seed
        }
    }
}