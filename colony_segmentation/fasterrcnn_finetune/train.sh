# Convert notebook to regular python script
jupyter nbconvert --to script fastrcnn_finetune.ipynb

nohup python -u fastrcnn_finetune.py > output.log 2>&1 &