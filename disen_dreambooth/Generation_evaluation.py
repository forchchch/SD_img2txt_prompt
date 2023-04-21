from evaluation import get_prompt_list
classes = [ ("backpack", "backpack"), ("backpack_dog", "backpack"),
("bear_plushie", "stuffed animal"), ("berry_bowl", "bowl"), 
("can", "can"), ("candle", "candle"), 
("cat", "cat"), ("cat2", "cat"),
("clock", "clock"), ("colorful_sneaker", "sneaker"),
("dog", "dog"), ("dog2", "dog"),
("dog3", "dog"), ("dog5", "dog"), ("dog6", "dog"),
("dog7", "dog"), ("dog8", "dog"),
("duck_toy", "toy"), ("fancy_boot", "boot"),
("grey_sloth_plushie", "stuffed animal"), ("monster_toy", "toy"),
("pink_sunglasses", "glasses"), ("poop_emoji", "toy"),
("rc_car", "toy"), ("red_cartoon", "cartoon"),
("robot_toy", "toy"), ("shiny_sneaker", "sneaker"),
("teapot", "teapot"), ("vase", "vase"),
("wolf_plushie", "stuffed animal")]
name, prompt_class = classes[23]
unique_token = name+ "</w>"
class_token = prompt_class
mode_list = ["object", "live"]
eval_mode = mode_list[0]
prompt_list = get_prompt_list(unique_token, class_token, mode = eval_mode)
print(unique_token)
print(eval_mode)

import sys
sys.path.append("../")
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from lora_diffusion import tune_lora_scale, patch_pipe
import open_clip
from PIL import Image
from disen_net import Image_adapter
import os
from visualization import joint_visualization
model_id = "/DATA/DATANAS1/zhangyip/models/stable-diffusion-2-1-base"

with torch.no_grad():
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    img_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    img_model = img_model.to("cuda")
from evaluator import Evaluator
evalor = Evaluator("cuda:0", model_name='ViT-H-14', source="laion2b_s32b_b79k")
evalor1 = Evaluator("cuda:0", model_name='DINO', source="laion2b_s32b_b79k")

dir_path = "./output_dreambooth/our_versions/" + name + "/global0.01disen0.001/checkpoint"
patch_pipe(
    pipe,
    os.path.join(dir_path, "lora_weight_e599_s3000.pt"),
    patch_text=False,
    patch_ti=False,
    patch_unet=True,
)
tune_lora_scale(pipe.unet, 1.0)
adapter = Image_adapter().to("cuda")
info = torch.load(os.path.join(dir_path, "lora_weight_e599_s3000.img_adapter.pt"))
adapter.load_state_dict(info)
ref_image = preprocess(Image.open( os.path.join("/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data", name, "02.jpg") )).unsqueeze(0).to("cuda")

#seed 12, 16, 8, 9 
# seed = 2023
# torch.manual_seed(seed)
save_root = "./generated_pictures"
global_eta = 0.0
image_num = 4
save_dir = os.path.join(save_root, name, "global0.01_disen0.001_step3000", str(global_eta))
os.makedirs(save_dir, exist_ok=True)


for m in range(len(prompt_list)):
    prompt = prompt_list[m]
    image = joint_visualization(pipe, img_model, prompt, ref_image, guidance=7.0, eta=global_eta, img_adapter=adapter, step=50, num_images_per_prompt = image_num)
    print(m,prompt)
    for n in range(image_num):
        image[n].save(os.path.join(save_dir, str(m)+str(n)+".jpg"))

origin_data_path = "/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data/" + name
generated_path = save_dir
from evaluation import reconstruction_metric, text_img_match_metric 
clip_sim = reconstruction_metric(origin_data_path, generated_path, evalor)
DINO_sim = reconstruction_metric(origin_data_path, generated_path, evalor1)
image_text_similarity = text_img_match_metric(generated_path, evalor, unique_token=unique_token, class_token=class_token, mode=eval_mode, img_num_per_prompt=image_num)
print("clip image similarity:", clip_sim)
print("DINO image similarity:", DINO_sim)
print("image text similarity:", image_text_similarity)
