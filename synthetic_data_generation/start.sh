# Remove old script
rm circle_finding.py

# Convert notebook to regular python script
jupyter nbconvert --to script circle_finding.ipynb

# Remove old log file
rm output.log

# Start training script in background
nohup python -u circle_finding.py > output.log 2>&1 &

# Follow the output log
tail -f output.log