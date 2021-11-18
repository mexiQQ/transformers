
export TASK_NAME=mrpc

python run_glue_no_trainer.py \
  --model_name_or_path textattack/bert-base-uncased-MRPC \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --eval_step 20 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/moffett.ai/transformers-experiments/text-classification \
  --pruning_sparsity 0.9375 \
  --prune \
  --pruning_epochs 3 \
  --pruning_frequency 20 \
  --deploy_device none \
  --group_size 64 \
  --kd
