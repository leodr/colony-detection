# Remove old script
rm -f patch_generation_pipeline.py

# Convert notebook to regular python script
jupyter nbconvert --to script generate_inpaint_patches.ipynb

# Remove old log file
rm -f output.log

# Start training script in background
nohup python -u generate_inpaint_patches.py > output.log 2>&1 &

# Follow the output log
tail -f output.log