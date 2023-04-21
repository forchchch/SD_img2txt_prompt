import os
import random
import torch
import tqdm
import numpy as np

from base_evaluation import reconstruction_metric, text_img_match_metric, get_prompt_list
from evaluator import Evaluator

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler, 
    UNet2DConditionModel,
)

from lora_diffusion import tune_lora_scale, patch_pipe

# backpack      7     800
# backpack_dog  10    1100
# bear_plushie  4     500
# bear_plushie  7     800   0.7276 0.7716 0.3230
# berry_bowl    3     400
# can           10    1100
# candle        3     400
# cat           4     500
# cat2          3     400   0.6402 0.7077 0.3611
# cat2          10    1100  0.6840 0.7887 0.3363
# clock         5     600
# colorful_sneaker 5  600
# dog           5     600
# dog2          5     600
# dog3          5     600   8     900
# dog5          5     600
# dog6          5     600   0.8072  0.8424  0.2828
# dog6          3     400   
# dog7          5     600
# dog8          7     800   
# duck_toy          4     500  
# fancy_boot        5     600 
# grey_sloth_plushie 3  400
# monster_toy 5  600
# pink_sunglasses glasses 3 400
# poop_emoji toy 3 400
# rc_car toy 5 600
# red_cartoon cartoon 14 1500
# robot_toy toy 5 600
# shiny_sneaker sneaker  4 500
# teapot teapot 6 700
# vase vase 4 500
# wolf_plushie stuffed animal 4 500

prompt_token = "wolf_plushie"
class_token = "stuffed animal"
type_token = "object"

origin_root = f"/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data/{prompt_token}"
model_path = f"/DATA/DATANAS1/zhangyip/SD_img2txt_prompt/training_scripts/output_baseline/dreambooth_a_prompt/{prompt_token}/checkpoints/lora_weight_e4_s500.pt"
gen_root = f"/DATA/DATANAS1/chenhong/diffusion_research/lora/baseline_gen_images/{prompt_token}"

def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def generate_images(root, unique_token, class_token, mode='object'):
    seeds = [8, 9, 12, 16]
    os.makedirs(root, exist_ok=True)
    pretrained_model_name_or_path = "/DATA/DATANAS1/zhangyip/models/stable-diffusion-2-1-base"
    prompt_list = get_prompt_list(unique_token, class_token, mode)
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to("cuda")
    patch_pipe(
        pipeline,
        model_path,
        patch_text=True,
        patch_ti=False,
        patch_unet=True,
    )
    tune_lora_scale(pipeline.unet, 1.0)
    tune_lora_scale(pipeline.text_encoder, 1.00)
    for seed in seeds:
        path = os.path.join(root, f"seed{seed}")
        os.makedirs(path, exist_ok=True)
        with tqdm.tqdm(range(len(prompt_list))) as t:
            for idx in t:
                save_path = os.path.join(path, f"{idx}.jpg")
                # if not os.path.exists(save_path):
                prompt = prompt_list[idx]
                image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.0).images[0]
                image.save(save_path)

if __name__ == "__main__":
    generate = True
    if generate:
        generate_images(gen_root, f"{prompt_token}</w>", class_token, type_token)
    # model_name = "DINO"
    model_name = 'ViT-H-14'
    # evalor = Evaluator('cuda:0')
    evalor = Evaluator('cuda:0', model_name=model_name)
    evalor_dino = Evaluator('cuda:0', model_name='DINO')
    recon_sims = []
    match_metrics = []
    recon_sims_dino = []
    seeds = [8, 9, 12, 16]
    set_rng_seed(2023)
    with tqdm.tqdm(range(len(seeds))) as t:
        for idx in t:
            seed = seeds[idx]
            # set_rng_seed(seed)
            recon_sim = reconstruction_metric(origin_root, os.path.join(gen_root, f"seed{seed}"), evalor)
            # print(f"recon sim: {recon_sim}")
            match_metric = text_img_match_metric(os.path.join(gen_root, f"seed{seed}"), evalor, f"{prompt_token}</w>", class_token, type_token)
            # print(f"match metric: {match_metric}")
            recon_sim_dino = reconstruction_metric(origin_root, os.path.join(gen_root, f"seed{seed}"), evalor_dino)
            recon_sims.append(recon_sim)
            match_metrics.append(match_metric)
            recon_sims_dino.append(recon_sim_dino)
    print(f"Mean recon sim: {np.mean(recon_sims)}")
    print(f"Mean recon sims dino: {np.mean(recon_sims_dino)}")
    print(f"Mean match metric: {np.mean(match_metrics)}")

