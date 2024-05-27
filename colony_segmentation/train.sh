# Remove old script
rm -f fastrcnn_finetune.py

# Convert notebook to regular python script
jupyter nbconvert --to script fastrcnn_finetune.ipynb

# Remove old log file
rm -f output.log

# Start training script in background
nohup python -u fastrcnn_finetune.py > output.log 2>&1 &

sleep 2

# Follow the output log
tail -f output.log