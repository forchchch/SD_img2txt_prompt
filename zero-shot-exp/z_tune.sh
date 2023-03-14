export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./training_data/images"
export SCENARIO_DIR="./training_data/aux_images"
export OUTPUT_DIR="./output"

CUDA_VISIBLE_DEVICES=2 accelerate launch z_tune.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --scenario_data_dir=$SCENARIO_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --save_steps=200 \
  --lora_rank=4 \
  --exp_name="addemb_1.0text_reg_lr1e-4" \
  --guidance_scale=0.0 \
  --prior_loss_weight=1.0 \
  --text_reg \
  # --joint_loss \