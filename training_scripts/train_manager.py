import os
import subprocess

# (prompt, class)
classes = [("backpack", "backpack"), ("backpack_dog", "backpack"),
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

MODEL_NAME="/DATA/DATANAS1/zhangyip/models/stable-diffusion-2-1-base"
INSTANCE_ROOT="/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data"
OUTPUT_ROOT="./output_baseline/dreambooth"
CLASS_ROOT="./class_data"

def trial(args):
    cmd = "CUDA_VISIBLE_DEVICES=3 accelerate launch train_lora_dreambooth.py"
    cmd += " --train_text_encoder"
    cmd += " --color_jitter"
    for k in args:
        v = args[k]
        cmd += ' --' + k
        if type(v) == str:
            cmd += f" {v}"
        elif type(v) == int:
            cmd += f" {int(v)}"
        elif type(v) == float:
            cmd += f" {float(v)}"
        else:
            raise NotImplementedError(f"not support key {k} with a {type(v)} value {v}")
    print(cmd)
    try:
        subprocess.run(cmd, shell=True)
    except subprocess.CalledProcessError:
        print(f"failed ...\n{cmd}")

if __name__ == "__main__":
    for prompt_tuple in classes:
        prompt, prompt_class = prompt_tuple
        args = {
            "pretrained_model_name_or_path": MODEL_NAME,
            "instance_data_dir": os.path.join(INSTANCE_ROOT, prompt), 
            "output_dir": os.path.join(OUTPUT_ROOT, prompt),
            "class_data_dir": os.path.join(CLASS_ROOT, prompt_class),
            "class_prompt": prompt_class,
            "instance_prompt": f"{prompt}\</w\>",
            "resolution": 512,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "learning_rate_text": 5e-5,
            "lr_scheduler": "constant",
            "save_steps": 100,
            "lr_warmup_steps": 0,
            "max_train_steps": 3000,
            "with_prior_preservation": 0.5,
        }
        trial(args)