# after install https://github.com/huggingface/transformers

# cd examples/question-answering
# mkdir -p data

# wget -O data/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

# wget -O data/dev-v1.1.json  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

# python run_squad.py \
#   --model_type bert \
#   --model_name_or_path bert-base-uncased \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --train_file train-v1.1.json \
#   --predict_file dev-v1.1.json \
#   --per_gpu_train_batch_size 16 \
#   --per_gpu_eval_batch_size 16 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2.0 \
#   --max_seq_length 320 \
#   --doc_stride 128 \
#   --data_dir data \
#   --output_dir data/bert-base-uncased-squad-v1 2>&1 | tee train-energy-bert-base-squad-v1.log

# CUDA_VISIBLE_DEVICES=1 python run_qa.py \
#   --model_name_or_path csarron/bert-base-uncased-squad-v1 \
#   --dataset_name squad \
#   --do_eval \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --output_dir runs/squad

# CUDA_VISIBLE_DEVICES=2,3 WORLD_SIZE=2 python -m torch.distributed.launch --nproc_per_node 2 --master_port 42474 run_qa.py \
# CUDA_VISIBLE_DEVICES=0,1 python run_qa.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad_v2 \
#   --do_train \
#   --do_eval \
#   --overwrite_cache \
#   --save_steps 100000 \
#   --version_2_with_negative \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --num_train_epochs 5 \
#   --learning_rate 5e-5 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir runs/squad8

# prune on downstream directly
# CUDA_VISIBLE_DEVICES=1,2,3 WORLD_SIZE=3 python -m torch.distributed.launch --nproc_per_node 3 --master_port 42475 
# csarron/bert-base-uncased-squad-v1
# /ssd/jianwei/transformers/examples/pytorch/question-answering/runs/200_epoch

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch run_qa_no_trainer.py \
python run_qa_no_trainer.py \
  --model_name_or_path csarron/bert-base-uncased-squad-v1 \
  --dataset_name squad \
  --overwrite_cache True\
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 3 \
  --learning_rate 3e-5 \
  --weight_decay 0.05 \
  --max_seq_length 384 \
  --eval_step 1 \
  --doc_stride 128 \
  --output_dir /content/drive/MyDrive/moffett.ai/transformers-experiments
  # --jiant_prepruned_weight /ssd/jianwei/jiant/moffett/runs/bbs_moffett_nlp_experiment_on_squad_v1_Test_7_lr_1.5e-05_bs_16_te_40_ef_200/best_model.p
  # --pruning_sparsity 0.9375 \
  # --prune \
  # --pruning_epochs 160 \
  # --pruning_frequency 2000 \
  # --deploy_device none \
  # --group_size 64
  # --kd


# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --overwrite_cache True\
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --num_train_epochs 3 \
#   --learning_rate 6e-5 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir runs/squad_dense_multi_gpus_0 \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --overwrite_cache True\
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --num_train_epochs 3 \
#   --learning_rate 7e-5 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir runs/squad_dense_multi_gpus_1 \
 

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --overwrite_cache True\
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --num_train_epochs 3 \
#   --learning_rate 8e-5 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir runs/squad_dense_multi_gpus_2 \

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --overwrite_cache True\
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --num_train_epochs 3 \
#   --learning_rate 9e-5 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir runs/squad_dense_multi_gpus_3 \
  








# CUDA_VISIBLE_DEVICES=0,1 python run_qa.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name trivia_qa \
#   --dataset_config_name rc.wikipedia \
#   --do_train \
#   --do_eval \
#   --overwrite_cache \
#   --save_steps 100000 \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --output_dir runs/trivia \
#   --num_train_epochs 1 \
#   --learning_rate 3e-5

# 上游任务 500w kd 3 epoch, fintune 下游
# CUDA_VISIBLE_DEVICES=1,2,3 WORLD_SIZE=3 python -m torch.distributed.launch --nproc_per_node 3 --master_port 42475 run_qa_no_trainer.py \
#   --model_name_or_path /ssd/jianwei/Pretrained-Language-Model/TinyBERT/output/2021_11_03_094245_511937 \
#   --dataset_name squad \
#   --overwrite_cache True \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --num_train_epochs 12 \
#   --learning_rate 1.5e-5 \
#   --max_seq_length 320 \
#   --doc_stride 128 \
#   --output_dir runs/squad12 \
#   --prune \
#   --pruning_epochs 3 \
#   --pruning_frequency 200 \
#   --deploy_device none \
#   --group_size 16 \
#   --fixed_mask /ssd/jianwei/bert_moffett/output_dir/output_20211101_111049/mask.pt
#  # --kd

# tinybert 500w kd 3 epoch fintune 下游 
# CUDA_VISIBLE_DEVICES=1,2,3 WORLD_SIZE=3 python -m torch.distributed.launch --nproc_per_node 3 --master_port 42475 run_qa_no_trainer.py \
#   --model_name_or_path /ssd/jianwei/Pretrained-Language-Model/TinyBERT/output/2021_11_02_084830_426251 \
#   --dataset_name squad \
#   --overwrite_cache True \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --num_train_epochs 12 \
#   --learning_rate 3e-5 \
#   --max_seq_length 320 \
#   --doc_stride 128 \
#   --output_dir runs/squad13
#   # --prune \
#   # --pruning_epochs 3 \
#   # --pruning_frequency 200 \
#   # --deploy_device none \
#   # --group_size 16 \
#   # --fixed_mask /ssd/jianwei/bert_moffett/output_dir/output_20211101_111049/mask.pt
#   # --kd
