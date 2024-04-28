# Remove old script
rm fastrcnn_finetune.py

# Convert notebook to regular python script
jupyter nbconvert --to script fastrcnn_finetune.ipynb

# Remove old log file
rm output.log

# Start training script in background
nohup python -u fastrcnn_finetune.py > output.log 2>&1 &

# Follow the output log
tail -f output.log