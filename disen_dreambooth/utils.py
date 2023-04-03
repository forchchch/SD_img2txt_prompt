import os
import logging
import torch


def my_make_dir(out_root, exp_name):
    logging_dir = os.path.join(out_root, exp_name, "loggers")
    out_image_dir =  os.path.join(out_root, exp_name, "generated_images")
    out_check_dir = os.path.join(out_root, exp_name, "checkpoint")
        
    os.makedirs(logging_dir, exist_ok=True)    
    os.makedirs(out_image_dir, exist_ok=True) 
    os.makedirs(out_check_dir, exist_ok=True) 
    
    return logging_dir, out_image_dir, out_check_dir

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

def save_torch_model(model, path):
    torch.save(model.state_dict() ,path)

def cal_cos(text, img, cos):
    a = text.view(text.size()[0], -1)
    b = img.view(img.size()[0], -1)
    sim = cos(a, b).mean()
    return sim
    