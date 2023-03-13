from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
import torch
from torch.utils.data import Dataset
import hashlib
import os

actions = ["running", "lying", "dancing", "standing", "sitting", "flying"]
scenarios = ["in the forest", "in the sky", "in the water", "in the room", "in the kitchen", "on the moon",
             "at night", "on the beach", "in the sunshine", "under a sakura tree", "beside a river", "in the flowers"]
out_root = "./training_data/aux_images"

class PromptDataset(Dataset):

    def __init__(self, prompt_list, num_samples):
        self.prompt_list = prompt_list
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples*len(self.prompt_list)

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt_list[int(index/self.num_samples)]
        example["index"] = index
        return example


images_per_prompt = 20
model_id = "stabilityai/stable-diffusion-2-1-base"
accelerator = Accelerator(gradient_accumulation_steps = 4, mixed_precision=None)
torch_dtype = ( torch.float16 if accelerator.device.type == "cuda" else torch.float32)
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
sample_dataset = PromptDataset(actions + scenarios, images_per_prompt)
sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=5)
sample_dataloader = accelerator.prepare(sample_dataloader)
pipeline.to(accelerator.device)

for example in sample_dataloader:
    images = pipeline(example["prompt"]).images

    for i, image in enumerate(images):
        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
        save_dir = os.path.join(out_root, example["prompt"][i])
        os.makedirs(save_dir,exist_ok=True)
        image_filename = os.path.join(save_dir, str( (example["index"][i].item()%images_per_prompt) )+".jpg")
        image.save(image_filename)

del pipeline
if torch.cuda.is_available():
    torch.cuda.empty_cache()