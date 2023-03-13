#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data_example"
export OUTPUT_DIR="./output_dreambooth_noreg"
export CLASS_DIR="./class_data"

CUDA_VISIBLE_DEVICES=3 accelerate launch train_lora_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a <s1>|<s2> man" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=6000 \
  --lora_rank=4 \
 

