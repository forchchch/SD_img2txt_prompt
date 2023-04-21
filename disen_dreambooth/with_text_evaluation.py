from evaluator import Evaluator
from evaluation import reconstruction_metric, text_img_match_metric 

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
image_num = 4

evalor = Evaluator("cuda:0", model_name='ViT-H-14', source="laion2b_s32b_b79k")
evalor1 = Evaluator("cuda:0", model_name='DINO', source="laion2b_s32b_b79k")
origin_data_path = "/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data/" + name
generated_path = "/DATA/DATANAS1/chenhong/diffusion_research/lora/disen_dreambooth/eval_pictures/text_pictures/bear_plushie/global0.01disen0.001_step800/0.0"



clip_sim = reconstruction_metric(origin_data_path, generated_path, evalor)
DINO_sim = reconstruction_metric(origin_data_path, generated_path, evalor1)
image_text_similarity = text_img_match_metric(generated_path, evalor, unique_token=unique_token, class_token=class_token, mode=eval_mode, img_num_per_prompt=image_num)
print("clip image similarity:", clip_sim)
print("DINO image similarity:", DINO_sim)
print("image text similarity:", image_text_similarity)