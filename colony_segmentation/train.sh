# Start training script in background
nohup python -u train.py \
  --learning_rate 0.005 \
  --use_coco_pretrained 0 \
  --use_agar_pretraining 0 \
  --num_epochs_agar 20 \
  --num_epochs_photobox 1000 \
  --use_rotation_augmentation 0 \
  --use_patch_augmentation 0 \
  --max_norm -1 \
  > output.log 2>&1 &

sleep 2

# Follow the output log
tail -f output.log