python quality_criteria.py \
--device gpu \
--content /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/AesUST/inputs/select_content \
--style /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/AesUST/inputs/select_style \
--stylized /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/AesUST/evaluation_outputs/ \
--output_log AesUST_pretrained_weights_evaluation_outputs.txt

python quality_criteria.py \
--device gpu \
--content /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/AesUST/inputs/select_content \
--style /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/AesUST/inputs/select_style \
--stylized /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/AesUST_improvement/train_log_v2/evaluation_outputs/ \
--output_log AesUST_v2_evaluation_outputs.txt