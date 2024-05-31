# Remove old script
rm -f fastrcnn_finetune.py

# Convert notebook to regular python script
jupyter nbconvert --to script fastrcnn_finetune.ipynb

# Remove old log file
rm -f output.log

# Add environment variables
export CUDA_DEVICE="cuda:0"
export EXP_LEARNING_RATE="0.001"
export EXP_USE_COCO_PRETRAINED="1"
export EXP_USE_AGAR_PRETRAINING="1"
export EXP_NUM_EPOCHS_AGAR="20"
export EXP_NUM_EPOCHS_PHOTOBOX="200"
export EXP_USE_ROTATION_AUGMENTATION="1"
export EXP_USE_PATCH_AUGMENTATION="0"
export EXP_MAX_NORM="1"

# Start training script in background
nohup python -u fastrcnn_finetune.py > output.log 2>&1 &

sleep 2

# Follow the output log
tail -f output.log