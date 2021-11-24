export TASK_NAME=mrpc

python run_glue_no_trainer.py \
  --model_name_or_path textattack/bert-base-uncased-MRPC \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 1e-3 \
  --seed 42 \
  --eval_step 9 \
  --num_train_epochs 10 \
  --output_dir /content/drive/MyDrive/moffett.ai/transformers-experiments/text-classification \
  --pruning_sparsity 0.9375 \
  --prune \
  --pruning_epochs 1 \
  --pruning_frequency 10 \
  --deploy_device none \
  --group_size 64 \
  --kd \
  # --do_eval
  # --sift \
  # --sift_version 1 \
  # --do_eval
    # 

