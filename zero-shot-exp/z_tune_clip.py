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
from z_dataset import z_Dataset
from visualization import prompt_based_generation, clip_train_generation
import open_clip
from z_net import Image_adapter, Multimodal_adapter


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
        "--guidance_scale",
        type=float,
        default=None,
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
        "--scenario_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--text_reg",
        action = "store_true",
        help="Flag to use the textual prior preservation loss.",
    )
    parser.add_argument(
        "--text_reg_cimg",
        action = "store_true",
        help="Flag to use the textual prior preservation loss.",
    )
    parser.add_argument(
        "--joint_loss",
        action = "store_true",
        help="Flag to use the textual prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="both",
        help="The output format of the model predicitions and checkpoints.",
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
    print("here we print text reg:", args.text_reg)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    if args.seed is not None:
        set_seed(args.seed)
    olog_dir, oimg_dir, ocheck_dir = my_make_dir(args.output_dir, args.exp_name)
    logger = get_logger(os.path.join(olog_dir, 'logging.txt'), "glogger")
    
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

    img2txt_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k') 
    img_adapter = Multimodal_adapter()   
    
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    img2txt_model.requires_grad_(False)
    

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
    train_dataset = z_Dataset(
        image_data_root = args.instance_data_dir,
        scenario_data_root= args.scenario_data_dir,
        tokenizer = tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        color_jitter=args.color_jitter,
        resize=args.resize,
        text_reg=args.text_reg
        )

    def collate_fn(examples):
        instance_ids = [ example["instance_plain_text"] for example in examples ]
        instance_images = [ example["instance_image"] for example in examples ]
        instance_images = torch.stack(instance_images)
        instance_images = instance_images.to(memory_format=torch.contiguous_format).float()
        instance_ids = tokenizer.pad({"input_ids": instance_ids}, padding="max_length",
                                    max_length=tokenizer.model_max_length,  return_tensors="pt",).input_ids
        
        batch = {
            "instance_images": instance_images,
            "instance_ids": instance_ids
        }
        if args.text_reg:        
        
            scenario_ids = [example["scenario_text"] for example in examples]
            scenario_images = [example["scenario_image"] for example in examples]
            scenario_images = torch.stack(scenario_images)
            scenario_images = scenario_images.to(memory_format=torch.contiguous_format).float()
            scenario_ids = tokenizer.pad({"input_ids": scenario_ids}, padding="max_length",
                                        max_length=tokenizer.model_max_length,  return_tensors="pt",).input_ids        
            
            batch["scenario_images"] = scenario_images
            batch["scenario_ids"] = scenario_ids            
            
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
            {"params": itertools.chain(*text_encoder_lora_params), "lr": text_lr},
            {"params": img_adapter.parameters(), "lr": args.learning_rate}
            
        ] if args.train_text_encoder
        else [
            {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
            {"params": img_adapter.parameters(), "lr": args.learning_rate}
        ]
    )
    
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
    if args.train_text_encoder:
        (unet, text_encoder, optimizer, train_dataloader, lr_scheduler,img_adapter) = accelerator.prepare(unet, 
                                         text_encoder, optimizer, train_dataloader, lr_scheduler, img_adapter)
    else:
        (unet, optimizer, train_dataloader, lr_scheduler,img_adapter) = accelerator.prepare(unet, 
                                          optimizer, train_dataloader, lr_scheduler, img_adapter)
        
    vae.to(accelerator.device, dtype=weight_dtype)
    img2txt_model.to(accelerator.device, dtype=weight_dtype)
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
        accelerator.init_trackers("zero-shot-tune", config=vars(args))

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
    
    #######################begin the training process##################################
    for epoch in range(args.num_train_epochs):
        unet.train()
        img_adapter.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            instance_images = batch["instance_images"].to(dtype=weight_dtype)
            
            instance_ids = batch["instance_ids"]

            instance_latents = vae.encode( instance_images ).latent_dist.sample()
            instance_latents = instance_latents * 0.18215   
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(instance_latents)
            bsz = instance_latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint( 0, noise_scheduler.config.num_train_timesteps, (bsz,), device=instance_latents.device, )
            timesteps = timesteps.long()
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(instance_latents, noise, timesteps)
            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder( instance_ids )[0] 
            
            # Get the img2text embedding for conditionining
            with torch.no_grad():
                original_feature = img2txt_model.encode_image( clip_trans(instance_images) ).unsqueeze(1)
            img_feature = img_adapter(original_feature, encoder_hidden_states)
            # tv_hidden_state = torch.cat([encoder_hidden_states, instance_i2t_emb], dim=1)

            tv_hidden_state = encoder_hidden_states + img_feature
            # Predict the noise residual
            
            instance_model_pred = unet(noisy_latents, timesteps, tv_hidden_state).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(instance_latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(instance_model_pred.float(), target.float(), reduction="mean")
            if args.text_reg:
                scenario_images = batch["scenario_images"].to(dtype=weight_dtype)
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

                loss = loss + args.prior_loss_weight*F.mse_loss(scenario_model_pred.float(), scenario_target.float(), reduction="mean")
                
                if args.text_reg_cimg:
                    with torch.no_grad():
                        aoriginal_feature = img2txt_model.encode_image( clip_trans(scenario_images) ).unsqueeze(1)
                    aimg_feature = img_adapter(aoriginal_feature, scenario_text_prompt)
                    # tv_hidden_state = torch.cat([encoder_hidden_states, instance_i2t_emb], dim=1)

                    atv_hidden_state = aimg_feature + scenario_text_prompt
                    a_scenario_model_pred = unet(scenario_noisy_latents, timesteps, atv_hidden_state).sample
                    loss = loss +  args.prior_loss_weight*F.mse_loss(a_scenario_model_pred.float(), scenario_target.float(), reduction="mean")
                
            loss = loss/args.gradient_accumulation_steps              
            accelerator.backward(loss)
            global_step += 1
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder.parameters(), img_adapter.parameters()) if args.train_text_encoder
                    else itertools.chain( unet.parameters(), img_adapter.parameters())
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
                                revision=args.revision,)

                            filename_unet = (f"{ocheck_dir}/lora_weight_e{epoch}_s{global_step}.pt")
                            filename_text_encoder = f"{ocheck_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                            filename_i2tp = f"{ocheck_dir}/lora_weight_e{epoch}_s{global_step}.i2tp.pt"
                            logger.info(f"save weights {filename_unet}, {filename_text_encoder}, {filename_i2tp}")
                            save_lora_weight(pipeline.unet, filename_unet)
                            save_torch_model( accelerator.unwrap_model(img_adapter, **extra_args), filename_i2tp )
                            if args.train_text_encoder:
                                save_lora_weight(
                                    pipeline.text_encoder,
                                    filename_text_encoder,
                                    target_replace_module=["CLIPAttention"])
                            clip_train_generation(pipeline, img2txt_model, img_adapter, "", os.path.join(oimg_dir, str(epoch)+"_"+str(global_step)+"_"+"recon.jpg"), guidance_scale, eta=0.5, preprocess=preprocess, device = accelerator.device)
                            clip_train_generation(pipeline, img2txt_model, img_adapter, "in the flowers", os.path.join(oimg_dir, str(epoch)+"_"+str(global_step)+"_"+"edit.jpg"), guidance_scale, eta=0.5, preprocess= preprocess, device = accelerator.device)
                            last_save = global_step

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                logger.info(f"epoch:{epoch}, step:{step}, loss:{loss.detach().item()}")
    accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet, **extra_args),
            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
            revision=args.revision,)

        filename_unet = (f"{ocheck_dir}/lora_weight_final.pt")
        filename_text_encoder = f"{ocheck_dir}/lora_weight_final.text_encoder.pt"
        filename_i2tp = f"{ocheck_dir}/lora_weight_final.i2tp.pt"
        logger.info(f"save weights {filename_unet}, {filename_text_encoder}, {filename_i2tp}")
        save_lora_weight(pipeline.unet, filename_unet)
        save_torch_model( accelerator.unwrap_model(img_adapter, **extra_args), filename_i2tp )
        if args.train_text_encoder:
            save_lora_weight(
                pipeline.text_encoder,
                filename_text_encoder,
                target_replace_module=["CLIPAttention"])

    accelerator.end_training()        
           
    return

if __name__ == "__main__":
    args = my_parse_args()
    main(args)