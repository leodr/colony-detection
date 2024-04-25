# Convert notebook to regular python script
jupyter nbconvert --to script fastrcnn_finetune.ipynb

nohup python -u script.py > output.log 2>&1 &