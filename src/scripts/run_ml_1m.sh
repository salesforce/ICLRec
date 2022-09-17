python3 main.py --data_name ml-1m --cf_weight 0.0 \
--model_idx 1 --gpu_id 0 \
--batch_size 256 --contrast_type IntentCL \
--num_intent_cluster 256 --seq_representation_type mean \
--warm_up_epoches 0 --intent_cf_weight 0.1 --num_hidden_layers 2 --max_seq_length 200 
