#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="/DATA/DATANAS1/zhangyip/models/stable-diffusion-2-1-base"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data/backpack"
export OUTPUT_DIR="./output_dirs/output_dreambooth_with_prior_without_encoder_3"
export CLASS_DIR="./class_data"

CUDA_VISIBLE_DEVICES=2 accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --instance_prompt="a <s1>|<s2> backpack" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=6000 \
  --lora_rank=4 \
  --with_prior_preservation=1 \
  --prior_loss_weight=1.0 \
  --class_prompt="backpack"
 

