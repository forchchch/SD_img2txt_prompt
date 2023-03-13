from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import random
import os

actions = ["running", "lying", "dancing", "standing", "sitting", "flying"]
scenarios = ["in the forest", "in the sky", "in the water", "in the room", "in the kitchen", "on the moon",
             "at night", "on the beach", "in the sunshine", "under a sakura tree", "beside a river", "in the flowers"]
all_folders = actions + scenarios
img_per_scenario = 20

class z_Dataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        image_data_root,
        scenario_data_root,
        tokenizer,
        size=512,
        center_crop=False,
        color_jitter=False,
        resize=True,
        h_flip=False,
        text_reg = False
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize
        self.text_reg = text_reg

        self.instance_data_root = Path(image_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("the provided image data root doesn't exist.")

        self.instance_images_path = list(Path(image_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images
        
        if self.text_reg:
            self.scenario_root = scenario_data_root
            self.num_scenario_image = len(all_folders)*img_per_scenario
            self._length = max(self.num_instance_images, self.num_scenario_image)

        if resize:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize( (size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size) if center_crop else transforms.Lambda(lambda x: x),
                    transforms.ColorJitter(0.2, 0.1) if color_jitter else transforms.Lambda(lambda x: x),
                    transforms.RandomHorizontalFlip() if h_flip else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [  transforms.CenterCrop(size) if center_crop else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(0.2, 0.1) if color_jitter else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip() if h_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                ]
            )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open( self.instance_images_path[index % self.num_instance_images] )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_image"] = self.image_transforms(instance_image)
        text_plain = ""
        example["instance_plain_text"] = self.tokenizer(text_plain,
                                                        padding="do_not_pad",
                                                        truncation=True,
                                                        max_length=self.tokenizer.model_max_length,).input_ids
        if self.text_reg:
            scenario_img = Image.open(os.path.join(self.scenario_root, all_folders[int(index/img_per_scenario)], str(index%img_per_scenario)+".jpg"))
            scenario_prompt = all_folders[int(index/img_per_scenario)]
            example["scenario_image"] = self.image_transforms(scenario_img)
            example["scenario_text"] = self.tokenizer(scenario_prompt,
                                                    padding="do_not_pad",
                                                    truncation=True,
                                                    max_length=self.tokenizer.model_max_length,).input_ids            
        return example
