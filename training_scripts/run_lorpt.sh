 #https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="/DATA/DATANAS1/zhangyip/models/stable-diffusion-2-1-base"
export INSTANCE_DIR="/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data/backpack"
export OUTPUT_DIR="./output_dirs/ti_baseline_krkbackpack_6"

CUDA_VISIBLE_DEVICES=3 accelerate launch train_lora_w_ti.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=256 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-5 \
  --learning_rate_ti=1e-3 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=200 \
  --max_train_steps=10000 \
  --placeholder_token="<krkbackpack>" \
  --learnable_property="object"\
  --initializer_token="backpack" \
  --save_steps=200 \
  --unfreeze_lora_step=5000 \
  --stochastic_attribute="" \
  # --scale_lr \
  # --just_ti
  # --stochastic_attribute="in the flowers, on the moon, in the forest, on the beach, in the water" # these attributes will be randomly appended to the prompts
