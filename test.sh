#python test.py \
#--content_dir /media/glory/Transcend/Dataset/Scene/DIV2K_valid_HR \
#--style_dir ../AesUST/style/ \
#--output DIV2K_valid_HR_outputs \
#--content_size 512 \
#--style_size 512 \
#--crop \
#--vgg "pretrained_weights/vgg_normalised.pth" \
#--decoder "train_log_v1/weights_v1/decoder_iter_160000.pth" \
#--transform "train_log_v1/weights_v1/transformer_iter_160000.pth" \
#--discriminator "train_log_v1/weights_v1/discriminator_iter_160000.pth"

python test.py \
--content_dir ../AesUST/inputs/select_content \
--style_dir ../AesUST/inputs/select_style/ \
--output evaluation_outputs \
--vgg "pretrained_weights/vgg_normalised.pth" \
--decoder "train_log_v2/weights_v2/decoder_iter_360000.pth" \
--transform "train_log_v2/weights_v2/transformer_iter_360000.pth" \
--discriminator "train_log_v2/weights_v2/discriminator_iter_360000.pth"