#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="/DATA/DATANAS1/zhangyip/models/stable-diffusion-2-1-base"
export INSTANCE_DIR="/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data/backpack"
export OUTPUT_DIR="./output_dirs/output_dreambooth_with_prior_with_encoder_w"
export CLASS_DIR="./class_data/backpack"

CUDA_VISIBLE_DEVICES=1 accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a backpack" \
  --instance_prompt="backpack</w>" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-5 \
  --color_jitter \
  --lr_scheduler="constant" \
  --save_steps=100 \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --with_prior_preservation=1 \
  --prior_loss_weight=1.0 \
  --class_prompt="backpack"