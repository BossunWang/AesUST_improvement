#python test.py \
#--content_dir /media/glory/Transcend/Dataset/Scene/DIV2K_valid_HR \
#--style_dir ../AesUST/style/ \
#--output DIV2K_valid_HR_outputs \
#--content_size 512 \
#--style_size 512 \
#--crop \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--decoder "train_log_v2/weights_v2/decoder_iter_360000.pth" \
#--transform "train_log_v2/weights_v2/transformer_iter_360000.pth" \
#--discriminator "train_log_v2/weights_v2/discriminator_iter_360000.pth"

#python test.py \
#--content_dir ../AesUST/inputs/select_content \
#--style_dir ../AesUST/inputs/select_style/ \
#--output train_log_v2/evaluation_outputs \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--decoder "train_log_v2/weights_v2/decoder_iter_360000.pth" \
#--transform "train_log_v2/weights_v2/transformer_iter_360000.pth" \
#--discriminator "train_log_v2/weights_v2/discriminator_iter_360000.pth"

#python test_v3.py \
#--content_dir /media/glory/Transcend/Dataset/Scene/DIV2K_valid_HR \
#--style_dir ../AesUST/style/ \
#--output DIV2K_valid_HR_outputs \
#--content_size 512 \
#--style_size 512 \
#--crop \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v3/weights_v3/content_encoder_iter_500000.pth" \
#--decoder "train_log_v3/weights_v3/decoder_iter_500000.pth" \
#--transform "train_log_v3/weights_v3/transformer_iter_500000.pth" \
#--discriminator "train_log_v3/weights_v3/discriminator_iter_500000.pth"

#python test_v3.py \
#--content_dir ../AesUST/inputs/select_content \
#--style_dir ../AesUST/inputs/select_style/ \
#--output train_log_v3/evaluation_outputs \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v3/weights_v3/content_encoder_iter_400000.pth" \
#--decoder "train_log_v3/weights_v3/decoder_iter_400000.pth" \
#--transform "train_log_v3/weights_v3/transformer_iter_400000.pth" \
#--discriminator "train_log_v3/weights_v3/discriminator_iter_400000.pth"

#python test_v3.py \
#--content_dir ../creativity-transfer/outputs_Louis_cat \
#--style_dir ../creativity-transfer/train_Louis \
#--output train_log_v3/creativity_transfer_Louis_cat \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v3/weights_v3/content_encoder_iter_400000.pth" \
#--decoder "train_log_v3/weights_v3/decoder_iter_400000.pth" \
#--transform "train_log_v3/weights_v3/transformer_iter_400000.pth" \
#--discriminator "train_log_v3/weights_v3/discriminator_iter_400000.pth"

#python test_v3.py \
#--content_dir ../AesUST/inputs/select_content \
#--style_dir ../AesUST/inputs/select_style/ \
#--output train_log_v4/evaluation_outputs \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v4/weights_v4/content_encoder_iter_400000.pth" \
#--decoder "train_log_v4/weights_v4/decoder_iter_400000.pth" \
#--transform "train_log_v4/weights_v4/transformer_iter_400000.pth" \
#--discriminator "train_log_v4/weights_v4/discriminator_iter_400000.pth"

#python test_v3.py \
#--content_dir ../creativity-transfer_develop/outputs_bob1_test_afhq_cat \
#--style_dir Louis_Wain_selected \
#--output /media/glory/Transcend/code/AesUST_improvement/train_log_v3/bob1_afhq_cat_Louis_Wain_selected_outputs \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v3/weights_v3/content_encoder_iter_400000.pth" \
#--decoder "train_log_v3/weights_v3/decoder_iter_400000.pth" \
#--transform "train_log_v3/weights_v3/transformer_iter_400000.pth" \
#--discriminator "train_log_v3/weights_v3/discriminator_iter_400000.pth"
#
#python test_v3.py \
#--content_dir ../creativity-transfer_develop/outputs_christmas_cat1_test_afhq_cat \
#--style_dir Louis_Wain_selected \
#--output /media/glory/Transcend/code/AesUST_improvement/train_log_v3/christmas_cat1_afhq_cat_Louis_Wain_selected_outputs \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v3/weights_v3/content_encoder_iter_400000.pth" \
#--decoder "train_log_v3/weights_v3/decoder_iter_400000.pth" \
#--transform "train_log_v3/weights_v3/transformer_iter_400000.pth" \
#--discriminator "train_log_v3/weights_v3/discriminator_iter_400000.pth"

#python test_v3.py \
#--content_dir ../creativity-transfer_develop/outputs_bob1_test_afhq_cat \
#--style_dir Louis_Wain_selected \
#--output /media/glory/Transcend/code/AesUST_improvement/train_log_v4/bob1_afhq_cat_Louis_Wain_selected_outputs \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v4/weights_v4/content_encoder_iter_400000.pth" \
#--decoder "train_log_v4/weights_v4/decoder_iter_400000.pth" \
#--transform "train_log_v4/weights_v4/transformer_iter_400000.pth" \
#--discriminator "train_log_v4/weights_v4/discriminator_iter_400000.pth"
#
#python test_v3.py \
#--content_dir ../creativity-transfer_develop/outputs_christmas_cat1_test_afhq_cat \
#--style_dir Louis_Wain_selected \
#--output /media/glory/Transcend/code/AesUST_improvement/train_log_v4/christmas_cat1_afhq_cat_Louis_Wain_selected_outputs \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v4/weights_v4/content_encoder_iter_400000.pth" \
#--decoder "train_log_v4/weights_v4/decoder_iter_400000.pth" \
#--transform "train_log_v4/weights_v4/transformer_iter_400000.pth" \
#--discriminator "train_log_v4/weights_v4/discriminator_iter_400000.pth"

#python test_v3.py \
#--content_dir ../AesUST/inputs/select_content \
#--style_dir ../AesUST/inputs/select_style/ \
#--output train_log_v5/evaluation_outputs \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v5/weights_v5/content_encoder_iter_400000.pth" \
#--decoder "train_log_v5/weights_v5/decoder_iter_400000.pth" \
#--transform "train_log_v5/weights_v5/transformer_iter_400000.pth" \
#--discriminator "train_log_v5/weights_v5/discriminator_iter_400000.pth"

#python test_v3.py \
#--content_dir ../creativity-transfer_develop/outputs_train_christmas_cat3_test_afhq_cat \
#--style_dir Louis_Wain_selected \
#--output /media/glory/Transcend/code/AesUST_improvement/train_log_v5/christmas_cat3_afhq_cat_Louis_Wain_selected_outputs \
#--content_size 512 \
#--style_size 512 \
#--crop \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v5/weights_v5/content_encoder_iter_400000.pth" \
#--decoder "train_log_v5/weights_v5/decoder_iter_400000.pth" \
#--transform "train_log_v5/weights_v5/transformer_iter_400000.pth" \
#--discriminator "train_log_v5/weights_v5/discriminator_iter_400000.pth"

#python test_v3.py \
#--content_dir "/media/glory/Transcend/code/DRB-GAN/train_log_v4/stylized_outputs_christmas_cat3_test_afhq_cat/LOUIS WAIN" \
#--style_dir Louis_Wain_selected \
#--output /media/glory/Transcend/code/AesUST_improvement/train_log_v5/christmas_cat3_afhq_cat_DRBGAN_Louis_Wain_selected_outputs \
#--content_size 512 \
#--style_size 512 \
#--crop \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--content_encoder "train_log_v5/weights_v5/content_encoder_iter_400000.pth" \
#--decoder "train_log_v5/weights_v5/decoder_iter_400000.pth" \
#--transform "train_log_v5/weights_v5/transformer_iter_400000.pth" \
#--discriminator "train_log_v5/weights_v5/discriminator_iter_400000.pth"

python test_v3.py \
--content_dir ../AesUST/inputs/select_content \
--style_dir ../AesUST/inputs/select_style/ \
--output train_log_v6/evaluation_outputs \
--vgg "pretrained_weights/vgg_normalised.pth" \
--content_encoder "train_log_v6/weights_v6/content_encoder_iter_400000.pth" \
--decoder "train_log_v6/weights_v6/decoder_iter_400000.pth" \
--transform "train_log_v6/weights_v6/transformer_iter_400000.pth" \
--discriminator "train_log_v6/weights_v6/discriminator_iter_400000.pth"