# Remove old script
rm -f find_inpaint_circles.py

# Convert notebook to regular python script
jupyter nbconvert --to script find_inpaint_circles.ipynb

# Remove old log file
rm -f output.log

# Start training script in background
nohup python -u find_inpaint_circles.py > output.log 2>&1 &

# Follow the output log
tail -f output.log