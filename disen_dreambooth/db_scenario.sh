#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="/DATA/DATANAS1/zhangyip/models/stable-diffusion-2-1-base"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data/backpack"
export OUTPUT_DIR="./output_dreambooth"
export CLASS_DIR="./class_data/backpack"
export SCENARIO_DIR="./training_data/aux_images"

CUDA_VISIBLE_DEVICES=0 accelerate launch dreambooth_scenario.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="<s1>|<s2>" \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="backpack" \
  --scenario_root=$SCENARIO_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000\
  --save_steps=200 \
  --lora_rank=4 \
  --exp_name="text_reg_1.0" \
  --train_text_encoder \
  --learning_rate_text=5e-5 \
  --with_scenario \
  --scenario_weight=1 \
#   --with_prior_preservation \
#   --prior_loss_weight=0.1 \