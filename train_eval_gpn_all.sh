# #!/bin/bash

# either ood or classification
setting=$1

# Array of your python scripts
datasets=("Cora" "CiteSeer" "PubMed" "AmazonPhotos" "AmazonComputers" "CoauthorCS" "CoauthorPhysics")

# Function to run a script on a specific GPU
run_script() {
    # CUDA_VISIBLE_DEVICES=$1 python $2 &
    # python3 "train_and_eval.py" with "configs/gpn/ood_loc_gpn_16.yaml" "data.dataset=$1" "run.gpu=$2" &
    if [ "$1" == "ood" ]; then
        python3 "train_and_eval.py" with "configs/gpn/ood_loc_gpn_16.yaml" "data.dataset=$2" "run.gpu=$3" &
    elif [ "$1" == "class" ]; then
        python3 "train_and_eval.py" with "configs/gpn/classification_gpn_16.yaml" "data.dataset=$2" "run.gpu=$3" &
    else
        echo "Wrong or missing parameter for running the script!"
        exit 1
    fi
}

# Max number of concurrent scripts
MAX_CONCURRENT=4

# Initialize GPU and script indices
gpu=0
script=0

# Array to keep track of running processes
pids=()

# Loop through all scripts
while [ $script -lt ${#datasets[@]} ]; do
    # Run script on GPU
    run_script $setting ${datasets[$script]} $gpu
    pids+=($!)  # Store PID of the process
    script=$((script+1))
    gpu=$((gpu+1))

    # Reset GPU index when it reaches 4
    if [ $gpu -eq $MAX_CONCURRENT ]; then
        gpu=0
    fi

    # Wait if max concurrent scripts are running
    if [ ${#pids[@]} -eq $MAX_CONCURRENT ]; then
        wait -n  # Wait for any process to finish
        # Remove finished process from the list
        pids=($(jobs -pr))
    fi
done

# Wait for all remaining processes to finish
wait


echo "All Experiments Have Completed."