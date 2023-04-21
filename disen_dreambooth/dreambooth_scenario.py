import sys
sys.path.append("../")
import argparse
import hashlib
import itertools
import math
import os
import inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from lora_diffusion import (extract_lora_ups_down, inject_trainable_lora, safetensors_available, save_lora_weight, save_safeloras)
from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import random
import re
from utils import my_make_dir, get_logger, save_torch_model
from dataset import DreamBoothDataset, PromptDataset, DBScenarioDataset
from visualization import dreambooth_save, joint_visualization_train
import open_clip
from disen_net import Image_adapter
from utils import cal_cos
from evaluation import obtain_metric
from evaluator import Evaluator

clip_trans = transforms.Resize( (224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

def my_parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        required=True,
        help="Name to distinguish different trials and experiments.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--special_token",
        type=str,
        default=None,
        help="special tokens",
    )
    parser.add_argument(
        "--reference_folder_name",
        type=str,
        default=None,
        help="reference folder name",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--scenario_data_dir",
        type=str,
        default="./training_data/aux_images",
        help="A root folder for scenario images"
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )    
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--scenario_root",
        type=str,
        default="./training_data/aux_images"
    )
    parser.add_argument(
        "--scenario_weight",
        type=float,
        default=1.0,
        help="The weight of scenario loss."
    )
    parser.add_argument(
        "--global_weight",
        type=float,
        default=0.1,
        help="The weight of scenario loss."
    )
    parser.add_argument(
        "--disen",
        type=float,
        default=0.001,
        help="The weight of scenario loss."
    )
    parser.add_argument(
        "--with_scenario",
        action="store_true",
        help="Flag to add text scenario loss."
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--img_adapt",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--resume_unet",
        type=str,
        default=None,
        help=("File path for unet lora to resume training."),
    )
    parser.add_argument(
        "--resume_text_encoder",
        type=str,
        default=None,
        help=("File path for text encoder lora to resume training."),
    )
    parser.add_argument(
        "--resize",
        type=bool,
        default=True,
        required=False,
        help="Should images be resized to --resolution before training?",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )
    args = parser.parse_args()
    return args

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    if args.seed is not None:
        set_seed(args.seed)
    olog_dir, oimg_dir, ocheck_dir = my_make_dir(args.output_dir, args.exp_name)
    logger = get_logger(os.path.join(olog_dir, 'logging.txt'), "glogger")

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = ( torch.float16 if accelerator.device.type == "cuda" else torch.float32 )
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    ######################load the pretrained models from stable diffusion and build our models##########
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision,)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,)     
    
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, 
                                                 subfolder="text_encoder", revision=args.revision,)
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,)
    
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,
        subfolder="unet", revision=args.revision,)
    
    img_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k') 
    if args.img_adapt:
        img_adapter = Image_adapter()
    else:
        img_adapter = None
    
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    img_model.requires_grad_(False)
    
    ############injecting the models with lora##########
    unet_lora_params, _ = inject_trainable_lora(unet, r=args.lora_rank, loras=args.resume_unet) 
    if args.train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(text_encoder, target_replace_module=["CLIPAttention"], r=args.lora_rank)

    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
    
    ###########building the dataset and dataloader
    train_dataset = DBScenarioDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        scenario_root=args.scenario_root,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        color_jitter=args.color_jitter,
        resize=args.resize,
        use_scenario=args.with_scenario,
        use_prior = args.with_prior_preservation
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]            
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad( {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",).input_ids
        batch = {"input_ids": input_ids, "pixel_values": pixel_values,}
        
        if args.with_scenario:
            scenario_ids = [example["scenario_ids"] for example in examples]
            scenario_values = [example["scenario_images"] for example in examples]
            scenario_values = torch.stack(scenario_values)
            scenario_values = scenario_values.to(memory_format=torch.contiguous_format).float()

            scenario_ids = tokenizer.pad(
                {'input_ids': scenario_ids},
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",).input_ids
            
            batch["scenario_values"] = scenario_values
            batch["scenario_ids"] = scenario_ids
        if args.with_prior_preservation:
            prior_ids = [example["class_prompt_ids"] for example in examples]
            prior_values = [example["class_images"] for example in examples]
            prior_values = torch.stack(prior_values)
            prior_values = prior_values.to(memory_format=torch.contiguous_format).float()

            prior_ids = tokenizer.pad(
                {'input_ids': prior_ids},
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",).input_ids
            
            batch["prior_values"] = prior_values
            batch["prior_ids"] = prior_ids            
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )
    
    ############define the training details
    if args.scale_lr:
        args.learning_rate = (args.learning_rate* args.gradient_accumulation_steps* args.train_batch_size * accelerator.num_processes)
    text_lr = (args.learning_rate if args.learning_rate_text is None else args.learning_rate_text)
    params_to_optimize = (
        [
            {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
            {"params": itertools.chain(*text_encoder_lora_params), "lr": text_lr}
            
        ] if args.train_text_encoder
        else [
            {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
        ]
    )
    if args.img_adapt:
        params_to_optimize.append( {"params": img_adapter.parameters(),"lr": args.learning_rate} )
    
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_config( args.pretrained_model_name_or_path, subfolder="scheduler")
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if args.img_adapt:
        if args.train_text_encoder:
            (unet, text_encoder, optimizer, train_dataloader, lr_scheduler, img_adapter) = accelerator.prepare(unet, 
                                            text_encoder, optimizer, train_dataloader, lr_scheduler, img_adapter)
        else:
            (unet, optimizer, train_dataloader, lr_scheduler, img_adapter) = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler, img_adapter)
    else:
        if args.train_text_encoder:
            (unet, text_encoder, optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(unet, 
                                            text_encoder, optimizer, train_dataloader, lr_scheduler)
        else:
            (unet, optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
        
    vae.to(accelerator.device, dtype=weight_dtype)
    img_model.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = ( args.train_batch_size* accelerator.num_processes* args.gradient_accumulation_steps)

    logger.info(f"***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")    
    logger.info(f"  running scripts: {args}")
    
    progress_bar = tqdm( range(args.max_train_steps), disable=not accelerator.is_local_main_process )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0
    guidance_scale = args.guidance_scale
    original_prompt = args.instance_prompt
    edit_prompt = args.instance_prompt + " in front of a blue house"
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    ref_image = preprocess(Image.open("/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data/" + args.reference_folder_name + "/01.jpg")).unsqueeze(0).to(accelerator.device).to(weight_dtype)

    #######################begin the training process##################################
    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            instance_images = batch["pixel_values"].to(dtype=weight_dtype)
            latents = vae.encode( instance_images ).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device,)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            with torch.no_grad():
                img_state = img_model.encode_image( clip_trans(instance_images) ).unsqueeze(1)
            # Predict the noise residual
            if args.img_adapt:
                img_state = img_adapter(img_state)
            model_pred = unet(noisy_latents, timesteps, img_state+encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            loss_main = F.mse_loss(model_pred.float(), target.float(), reduction="mean") 
            if args.global_weight>0.0:
                text_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss_main += args.global_weight*F.mse_loss(text_pred.float(), target.float(), reduction="mean") + args.disen*cal_cos(encoder_hidden_states, img_state, cos)
                loss_main = loss_main/args.gradient_accumulation_steps
            accelerator.backward(loss_main)

            if args.with_prior_preservation:
                prior_images = batch["prior_values"].to(dtype=weight_dtype)
                prior_ids = batch["prior_ids"]

                prior_latents = vae.encode( prior_images ).latent_dist.sample()
                prior_latents = prior_latents * 0.18215   
                # Sample noise that we'll add to the latents
                prior_noise = torch.randn_like(prior_latents)
                prior_noisy_latents = noise_scheduler.add_noise(prior_latents, prior_noise, timesteps)
                # Get the text embedding for conditioning
                prior_text_prompt = text_encoder( prior_ids )[0]                 

                # Predict the noise residual
                prior_model_pred = unet(prior_noisy_latents, timesteps, prior_text_prompt).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    prior_target = prior_noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    prior_target = noise_scheduler.get_velocity(prior_latents, prior_noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss_prior = args.prior_loss_weight * F.mse_loss(prior_model_pred.float(), prior_target.float(), reduction="mean")
                loss_prior = loss_prior/args.gradient_accumulation_steps          
                accelerator.backward(loss_prior)

            if args.with_scenario:
                scenario_images = batch["scenario_values"].to(dtype=weight_dtype)
                scenario_ids = batch["scenario_ids"]

                scenario_latents = vae.encode( scenario_images ).latent_dist.sample()
                scenario_latents = scenario_latents * 0.18215   
                # Sample noise that we'll add to the latents
                scenario_noise = torch.randn_like(scenario_latents)
                scenario_noisy_latents = noise_scheduler.add_noise(scenario_latents, scenario_noise, timesteps)
                # Get the text embedding for conditioning
                scenario_text_prompt = text_encoder( scenario_ids )[0]                 

                # Predict the noise residual
                scenario_model_pred = unet(scenario_noisy_latents, timesteps, scenario_text_prompt).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    scenario_target = scenario_noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    scenario_target = noise_scheduler.get_velocity(scenario_latents, scenario_noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss_scenario = args.scenario_weight * F.mse_loss(scenario_model_pred.float(), scenario_target.float(), reduction="mean")            
                loss_scenario = loss_scenario/args.gradient_accumulation_steps
                accelerator.backward(loss_scenario)

            global_step += 1
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(*unet_lora_params, *text_encoder_lora_params) if args.train_text_encoder
                    else itertools.chain( *unet_lora_params)
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            if global_step%args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    if args.save_steps and (global_step - last_save)/args.gradient_accumulation_steps >= args.save_steps:
                        if accelerator.is_main_process:
                            # newer versions of accelerate allow the 'keep_fp32_wrapper' arg. without passing
                            # it, the models will be unwrapped, and when they are then used for further training,
                            # we will crash. pass this, but only to newer versions of accelerate. fixes
                            # https://github.com/huggingface/diffusers/issues/1566
                            accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(inspect.signature(accelerator.unwrap_model).parameters.keys())
                            extra_args = ( {"keep_fp32_wrapper": True}if accepts_keep_fp32_wrapper else {})
                            pipeline = StableDiffusionPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                unet=accelerator.unwrap_model(unet, **extra_args),
                                text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
                                revision=args.revision,
                                torch_dtype=weight_dtype)

                            filename_unet = (f"{ocheck_dir}/lora_weight_e{epoch}_s{global_step}.pt")
                            filename_text_encoder = f"{ocheck_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                            if args.img_adapt:
                                filename_img_adapter = f"{ocheck_dir}/lora_weight_e{epoch}_s{global_step}.img_adapter.pt"
                                save_torch_model( accelerator.unwrap_model(img_adapter, **extra_args), filename_img_adapter)
                            logger.info(f"save weights {filename_unet}, {filename_text_encoder}")
                            save_lora_weight(pipeline.unet, filename_unet)
                            if args.train_text_encoder:
                                save_lora_weight(
                                    pipeline.text_encoder,
                                    filename_text_encoder,
                                    target_replace_module=["CLIPAttention"])
                            current_img_dir = os.path.join( oimg_dir, str(epoch)+"_"+str(global_step) )
                            os.makedirs( current_img_dir, exist_ok=True)
                            dreambooth_save(pipeline,  original_prompt, current_img_dir+"/recon.jpg", guidance_scale )
                            joint_visualization_train(pipeline, img_model, original_prompt, guidance_scale, current_img_dir+"/recon_sum.jpg" , preprocess, eta=1.0, img_adapter=img_adapter)
                            dreambooth_save(pipeline,  edit_prompt, current_img_dir+"/edit.jpg", guidance_scale )
                            #evaluator = Evaluator(device = accelerator.device, model_name = "ViT-H-14", mtype=weight_dtype).to(accelerator.device).to(weight_dtype)
                            #similarity = obtain_metric(pipeline, img_model, img_adapter, evaluator, ref_image, unique_token=args.special_token, class_token=args.class_prompt, save_dir=current_img_dir, mode="train")
                            last_save = global_step
                            #logger.info(f"epoch:{epoch}, step:{step}, generation similarity:{similarity}")

                if global_step%10 == 0:
                    logger.info(f"epoch:{epoch}, step:{step}, loss main:{loss_main.detach().item()}")
    accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet, **extra_args),
            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
            revision=args.revision,
            torch_dtype = weight_dtype)

        filename_unet = (f"{ocheck_dir}/lora_weight_final.pt")
        filename_text_encoder = f"{ocheck_dir}/lora_weight_final.text_encoder.pt"
        logger.info(f"save weights {filename_unet}, {filename_text_encoder}")
        save_lora_weight(pipeline.unet, filename_unet)
        if args.train_text_encoder:
            save_lora_weight(
                pipeline.text_encoder,
                filename_text_encoder,
                target_replace_module=["CLIPAttention"])
        if args.img_adapt:
            filename_img_adapter = f"{ocheck_dir}/lora_weight_final.img_adapter.pt"
            save_torch_model( accelerator.unwrap_model(img_adapter, **extra_args), filename_img_adapter)

    accelerator.end_training()        
           
    return

if __name__ == "__main__":
    args = my_parse_args()
    main(args)