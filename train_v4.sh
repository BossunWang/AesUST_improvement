CUDA_VISIBLE_DEVICES=1 python train.py \
  --content_dir "../../Photo_image_training_data" \
  --style_dir "../art_dataset_v2" \
  --vgg "pretrained_weights/vgg_normalised.pth" \
  --sample_path "samples_v4/" \
  --save_dir "weights_v4/" \
  --log_dir "log_v4/" \
  --checkpoints "checkpoint_v4" \
  --img_size 256 \
  --crop_size 256 \
  --lr 0.0001 \
  --lr_decay 0.00005 \
  --stage0_iter 50000 \
  --stage1_iter 100000 \
  --stage2_iter 250000 \
  --batch_size 1 \
  --n_threads 1 \
  --assigned_labels 6 \
  --print_interval 5000 \
  --save_model_interval 40000 \
  --style_weight 1.0 \
  --content_weight 1.0 \
  --gan_weight 1.0 \
  --identity_weight 50.0 \
  --AR1_weight 0.5 \
  --AR2_weight 500.0 \
  --patch_size 96 \
  --stride 48 \
  --top_k 8 \
  --use_patch
