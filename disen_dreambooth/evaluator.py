import open_clip
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn


class Evaluator(nn.Module):
    def __init__(self, device, mtype = torch.float16, model_name='ViT-H-14', source="laion2b_s32b_b79k"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        if model_name.startswith('ViT'):
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained = source)
            self.tokenizer = open_clip.get_tokenizer(model_name)
        
        if model_name == "DINO":
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            self.preprocess = transforms.Compose([
                                                    transforms.Resize(256, interpolation=3),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                ])                             # + skip convert PIL to tensor
        self.model = self.model.to(device)
        self.type = mtype
    # input two PIL images    
    def image_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


    def encode_images(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device).to(self.type)
        if self.model_name.startswith("ViT"):
            feature = self.model.encode_image(image)
        else:
            feature = self.model(image)
            
        return feature
    
    @torch.no_grad()
    def get_text_features(self, text, norm=True):

        tokens = self.tokenizer(text).to(self.device)
        text_features = self.model.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    @torch.no_grad()
    def get_image_features(self, img, norm=True):
        image_features = self.encode_images(img)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features


        