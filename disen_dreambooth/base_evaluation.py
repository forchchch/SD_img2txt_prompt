from visualization import joint_visualization
from evaluator import Evaluator
import os
import torch
import numpy
from PIL import Image


def get_prompt_list(unique_token, class_token, mode="train"):
    
    if mode == "train":
        prompt_list = [
        'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
        'a {0} {1} with a city in the background'.format(unique_token, class_token),
        'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
        'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
        'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
        'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
        'a {0} {1} on top of a mirror'.format(unique_token, class_token),
        'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
        'a cube shaped {0} {1}'.format(unique_token, class_token)
        ]
    
    if mode == "object":
        prompt_list = [
            'a {0} {1} in the jungle'.format(unique_token, class_token),
            'a {0} {1} in the snow'.format(unique_token, class_token),
            'a {0} {1} on the beach'.format(unique_token, class_token),
            'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
            'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
            'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
            'a {0} {1} with a city in the background'.format(unique_token, class_token),
            'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
            'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
            'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
            'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
            'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
            'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
            'a {0} {1} floating on top of water'.format(unique_token, class_token),
            'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
            'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
            'a {0} {1} on top of a mirror'.format(unique_token, class_token),
            'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
            'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
            'a {0} {1} on top of a white rug'.format(unique_token, class_token),
            'a red {0} {1}'.format(unique_token, class_token),
            'a purple {0} {1}'.format(unique_token, class_token),
            'a shiny {0} {1}'.format(unique_token, class_token),
            'a wet {0} {1}'.format(unique_token, class_token),
            'a cube shaped {0} {1}'.format(unique_token, class_token)
            ]

    if mode == "live":
        prompt_list = [
        'a {0} {1} in the jungle'.format(unique_token, class_token),
        'a {0} {1} in the snow'.format(unique_token, class_token),
        'a {0} {1} on the beach'.format(unique_token, class_token),
        'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
        'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
        'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
        'a {0} {1} with a city in the background'.format(unique_token, class_token),
        'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
        'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
        'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
        'a {0} {1} wearing a red hat'.format(unique_token, class_token),
        'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
        'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
        'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
        'a {0} {1} in a chef outfit'.format(unique_token, class_token),
        'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
        'a {0} {1} in a police outfit'.format(unique_token, class_token),
        'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
        'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
        'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
        'a red {0} {1}'.format(unique_token, class_token),
        'a purple {0} {1}'.format(unique_token, class_token),
        'a shiny {0} {1}'.format(unique_token, class_token),
        'a wet {0} {1}'.format(unique_token, class_token),
        'a cube shaped {0} {1}'.format(unique_token, class_token)
        ]
        
    return prompt_list
    

def obtain_metric(pipe, img_model, adapter, evaluator, ref_image ,unique_token, class_token, save_dir, mode="train"):
    with torch.no_grad():
        prompt_list = get_prompt_list(unique_token, class_token, mode)
        similarity = 0.0
        for m in range(len(prompt_list)):
            prompt = prompt_list[m]
            gen_image = joint_visualization(pipe, img_model, prompt, ref_image, guidance=7.0, eta=0.0, img_adapter=adapter, step=50)[0]
            gen_image.save( os.path.join(save_dir, str(m)+".jpg") )
            sim = evaluator.txt_img_similarity(prompt, gen_image).cpu().numpy()
            similarity += sim
        similarity = similarity/len(prompt_list)
    return similarity

def reconstruction_metric(origin_data_root, generated_data_root, evaluator):
    avg_sim = 0.0
    origin_pic_list = os.listdir(origin_data_root)
    gen_pic_list = os.listdir(generated_data_root)
    for m in range( len(origin_pic_list) ):
        origin_path = os.path.join( origin_data_root, origin_pic_list[m] )
        origin_image = Image.open( origin_path )
        for n in range( len(gen_pic_list) ):
            gen_path = os.path.join( generated_data_root, gen_pic_list[n] )
            gen_image = Image.open(gen_path)
            sim = evaluator.image_similarity(origin_image, gen_image).cpu().numpy()
            avg_sim += sim
    avg_sim /= len(origin_pic_list)*len(gen_pic_list)
    return avg_sim

def text_img_match_metric(generated_root, evaluator, unique_token="", class_token="backpack", mode="object"):
    prompt_list = get_prompt_list(unique_token, class_token, mode=mode)
    avg_sim = 0.0
    for m in range( len(prompt_list) ):
        img_path = os.path.join( generated_root, str(m)+".jpg" )
        gen_img = Image.open( img_path )
        prompt = prompt_list[m]
        sim = evaluator.txt_img_similarity(prompt, gen_img).cpu().numpy()
        avg_sim += sim
    return avg_sim/len(prompt_list)
    