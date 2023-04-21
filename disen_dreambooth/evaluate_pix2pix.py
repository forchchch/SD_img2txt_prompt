import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from evaluation import pix2pix_prompt_list
import logging
import os
from evaluator import Evaluator
from evaluation import reconstruction_metric, text_img_match_metric 

def get_logger(filename,name):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

model_id = "timbrooks/instruct-pix2pix"
# model_id = "/DATA/DATANAS1/zhangyip/models/stable-diffusion-2-1-base"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
evalor = Evaluator("cuda:0", model_name='ViT-H-14', source="laion2b_s32b_b79k")
evalor1 = Evaluator("cuda:0", model_name='DINO', source="laion2b_s32b_b79k")

out_path = "./output_dreambooth/pix2pix"
input_path = "/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data"
os.makedirs(out_path, exist_ok=True)
log_file_name = os.path.join(out_path, "logging.txt")
logger = get_logger(log_file_name, "log")

classes = [ ("backpack", "backpack", 0), ("backpack_dog", "backpack", 0),
("bear_plushie", "stuffed animal", 0), ("berry_bowl", "bowl", 0), 
("can", "can", 0), ("candle", "candle", 0), 
("cat", "cat", 1), ("cat2", "cat", 1),
("clock", "clock", 0), ("colorful_sneaker", "sneaker", 0),
("dog", "dog", 1), ("dog2", "dog", 1),
("dog3", "dog", 1), ("dog5", "dog", 1), ("dog6", "dog", 1),
("dog7", "dog", 1), ("dog8", "dog", 1),
("duck_toy", "toy", 0), ("fancy_boot", "boot", 0),
("grey_sloth_plushie", "stuffed animal", 0), ("monster_toy", "toy", 0),
("pink_sunglasses", "glasses", 0), ("poop_emoji", "toy", 0),
("rc_car", "toy", 0), ("red_cartoon", "cartoon", 0),
("robot_toy", "toy", 0), ("shiny_sneaker", "sneaker", 0),
("teapot", "teapot", 0), ("vase", "vase", 0),
("wolf_plushie", "stuffed animal", 0)]
mode_list = ["object", "live"]
pic_list = ["00.jpg", "01.jpg", "02.jpg", "03.jpg"]

total_num = len(classes)
with torch.no_grad():
    for i in range(total_num):
        i_name, i_class, i_mode = classes[i]
        eval_mode = mode_list[i_mode]
        logger.info(f"{i_name}, {i_class}, {eval_mode}")
        prompt_list = pix2pix_prompt_list("", i_class, eval_mode)
        save_dir = os.path.join(out_path,i_name)
        ref_dir = os.path.join(input_path, i_name)
        logger.info(f"{save_dir}, {ref_dir}")
        os.makedirs(save_dir, exist_ok=True)
        for m in range(len(prompt_list)):
            prompt = prompt_list[m]
            print("prompt:", prompt)
            for t in range(4):
                ref_image = Image.open( os.path.join(ref_dir, pic_list[t]) ).resize( (512,512) )
                image = pipe(prompt, image=ref_image, num_inference_steps=50).images[0]
                image.save( os.path.join(save_dir, str(m)+str(t)+".jpg") )
        clip_sim = reconstruction_metric(ref_dir, save_dir, evalor)
        DINO_sim = reconstruction_metric(ref_dir, save_dir, evalor1)
        image_text_similarity = text_img_match_metric(save_dir, evalor, unique_token=i_name+"</w>", class_token=i_class, mode=eval_mode, img_num_per_prompt=4)
        logger.info(f"clip image similarity:{clip_sim}")
        logger.info(f"DINO image similarity:{DINO_sim}")
        logger.info(f"image text similarity:{image_text_similarity}")
    
            
        
    