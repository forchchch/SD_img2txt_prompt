from evaluation import get_prompt_list
classes = [ ("backpack", "backpack",0), ("backpack_dog", "backpack",1),
("bear_plushie", "stuffed animal",2), ("berry_bowl", "bowl",3), 
("can", "can",4), ("candle", "candle",5), 
("cat", "cat",6), ("cat2", "cat",7),
("clock", "clock", 8), ("colorful_sneaker", "sneaker",9),
("dog", "dog", 10), ("dog2", "dog", 11),
("dog3", "dog", 12), ("dog5", "dog", 13), ("dog6", "dog", 14),
("dog7", "dog", 15), ("dog8", "dog", 16),
("duck_toy", "toy", 17), ("fancy_boot", "boot", 18),
("grey_sloth_plushie", "stuffed animal", 19), ("monster_toy", "toy", 20),
("pink_sunglasses", "glasses", 21), ("poop_emoji", "toy", 22),
("rc_car", "toy", 23), ("red_cartoon", "cartoon", 24),
("robot_toy", "toy", 25), ("shiny_sneaker", "sneaker", 26),
("teapot", "teapot", 27), ("vase", "vase", 28),
("wolf_plushie", "stuffed animal", 29)]
name, prompt_class,_ = classes[2]
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

dir_path = "./output_dreambooth/text_encode_version/" + name + "/global0.01disen0.001/checkpoint"
patch_pipe(
    pipe,
    os.path.join(dir_path, "lora_weight_e159_s800.pt"),
    patch_text=True,
    patch_ti=False,
    patch_unet=True,
)
tune_lora_scale(pipe.unet, 1.0)
adapter = Image_adapter().to("cuda")
info = torch.load(os.path.join(dir_path, "lora_weight_e159_s800.img_adapter.pt"))
adapter.load_state_dict(info)
ref_image = preprocess(Image.open( os.path.join("/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data", name, "02.jpg") )).unsqueeze(0).to("cuda")

seed = 2023
torch.manual_seed(seed)
save_root = "./eval_pictures/text_pictures"
global_eta = 0.0
image_num = 4
save_dir = os.path.join(save_root, name, "global0.01disen0.001_step800", str(global_eta))
os.makedirs(save_dir, exist_ok=True)


for m in range(len(prompt_list)):
    prompt = prompt_list[m]
    for n in range(image_num):
        image = joint_visualization(pipe, img_model, prompt, ref_image, guidance=7.0, eta=global_eta, img_adapter=adapter, step=50, num_images_per_prompt = 1)
        print(m,n,prompt)
        image[0].save(os.path.join(save_dir, str(m)+str(n)+".jpg"))