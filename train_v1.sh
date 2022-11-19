CUDA_VISIBLE_DEVICES=0 python train.py \
  --content_dir "/media/glory/Transcend/Dataset/Scene/Photo_image_training_data" \
  --style_dir "../art_dataset_v2" \
  --vgg "pretrained_weights/vgg_normalised.pth" \
  --sample_path "samples_v1/" \
  --save_dir "weights_v1/" \
  --log_dir "log_v1/" \
  --checkpoints "checkpoint_v1" \
  --img_size 256 \
  --crop_size 256 \
  --lr 0.0001 \
  --lr_decay 0.00005 \
  --stage1_iter 80000 \
  --stage2_iter 80000 \
  --batch_size 1 \
  --n_threads 1 \
  --assigned_labels 6 \
  --print_interval 2000 \
  --save_model_interval 10000 \
  --style_weight 1.0 \
  --content_weight 1.0 \
  --gan_weight 5.0 \
  --identity_weight 50.0 \
  --AR1_weight 0.5 \
  --AR2_weight 500.0
