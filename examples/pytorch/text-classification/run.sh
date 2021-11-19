export TASK_NAME=mrpc

python run_glue_no_trainer.py \
  --model_name_or_path textattack/bert-base-uncased-MRPC \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-5 \
  --eval_step 150 \
  --num_train_epochs 500 \
  --output_dir /content/drive/MyDrive/moffett.ai/transformers-experiments/text-classification \
  --pruning_sparsity 0.9375 \
  --prune \
  --pruning_epochs 400 \
  --pruning_frequency 200 \
  --deploy_device none \
  --group_size 64 \
  --kd \
  --sift \
  --sift_version 1
