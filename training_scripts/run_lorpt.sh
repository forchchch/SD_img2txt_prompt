 #https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="./data_example_pt"
export OUTPUT_DIR="./output_example_tisd2-1"

CUDA_VISIBLE_DEVICES=1 accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=6000 \
  --placeholder_token="<krk>" \
  --learnable_property="object"\
  --initializer_token="man" \
  --save_steps=500 \
  --unfreeze_lora_step=1500 \
  --stochastic_attribute="young, Asian man, wearing glasses" # these attributes will be randomly appended to the prompts
  