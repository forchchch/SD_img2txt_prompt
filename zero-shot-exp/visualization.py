import sys
sys.path.append("../")
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler,AutoencoderKL
from lora_diffusion import tune_lora_scale, patch_pipe
from PIL import Image
from z_dataset import z_Dataset
from torchvision import transforms


test_num_inference_steps=50
test_img_resolution = 512
num_images_per_prompt = 1
test_prompt_embeds = None
test_generator = None
test_latents = None
init_image = Image.open("./training_data/images/dog1.jpg").convert("RGB").resize((test_img_resolution, test_img_resolution))
image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

def prompt_based_generation(pipe, vae, i2t_prompt_model, prompt, file_name, test_guidance_scale):
    i2t_prompt_model.eval()
    pipe = pipe.to(vae.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    with torch.no_grad():
        img_feature = vae.encode( image_transform(init_image).cuda().view(1,3,test_img_resolution, test_img_resolution) ).latent_dist.sample()
        img_feature = img_feature * 0.18215
        ###here we start to write the eidttable inference process
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = pipe._execution_device
        do_classifier_free_guidance = test_guidance_scale > 1.0
        prompt_embeds = pipe._encode_prompt(prompt,device,num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt = None,
                prompt_embeds = None,
                negative_prompt_embeds = None)
        img_feature = torch.cat([torch.zeros_like(img_feature) ,img_feature]) if do_classifier_free_guidance else img_feature
        i2t_emb = i2t_prompt_model(img_feature, prompt_embeds)
        tv_prompt = torch.cat([prompt_embeds,i2t_emb], dim=1)

        pipe.scheduler.set_timesteps(test_num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        num_channels_latents = pipe.unet.in_channels
        height = test_img_resolution
        width = test_img_resolution
        latents = pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    test_generator,
                    test_latents,
                )
        num_warmup_steps = len(timesteps) - test_num_inference_steps * pipe.scheduler.order
        with pipe.progress_bar(total=test_num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states = tv_prompt, cross_attention_kwargs=None).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + test_guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
            
            image = pipe.decode_latents(latents)
            image, has_nsfw_concept = pipe.run_safety_checker(image, device, prompt_embeds.dtype)
            image = pipe.numpy_to_pil(image)
            image[0].save(file_name)
    i2t_prompt_model.train()


def jupyter_prompt_based_generation(pipe, vae, i2t_prompt_model, prompt, reference_image, guidance):
    i2t_prompt_model.eval()
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    with torch.no_grad():
        img_feature = vae.encode( image_transform(reference_image).cuda().view(1,3,test_img_resolution, test_img_resolution) ).latent_dist.sample()
        img_feature = img_feature * 0.18215
        ###here we start to write the eidttable inference process
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = pipe._execution_device
        do_classifier_free_guidance = guidance > 1.0
        prompt_embeds = pipe._encode_prompt(prompt,device,num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt = None,
                prompt_embeds = None,
                negative_prompt_embeds = None)
        img_feature = torch.cat([torch.zeros_like(img_feature),img_feature]) if do_classifier_free_guidance else img_feature
        i2t_emb = i2t_prompt_model(img_feature, prompt_embeds)
        tv_prompt = torch.cat([prompt_embeds,i2t_emb], dim=1)

        pipe.scheduler.set_timesteps(test_num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        num_channels_latents = pipe.unet.in_channels
        height = test_img_resolution
        width = test_img_resolution
        latents = pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    test_generator,
                    test_latents,
                )
        num_warmup_steps = len(timesteps) - test_num_inference_steps * pipe.scheduler.order
        with pipe.progress_bar(total=test_num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states = tv_prompt, cross_attention_kwargs=None).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
            
            image = pipe.decode_latents(latents)
            image, has_nsfw_concept = pipe.run_safety_checker(image, device, prompt_embeds.dtype)
            image = pipe.numpy_to_pil(image)
    i2t_prompt_model.train()
    return image


def only_text_prompt_generation(pipe, prompt, guidance):
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    with torch.no_grad():
        ###here we start to write the eidttable inference process
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = pipe._execution_device
        do_classifier_free_guidance = guidance > 1.0
        prompt_embeds = pipe._encode_prompt(prompt,device,num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt = None,
                prompt_embeds = None,
                negative_prompt_embeds = None)

        pipe.scheduler.set_timesteps(test_num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        num_channels_latents = pipe.unet.in_channels
        height = test_img_resolution
        width = test_img_resolution
        latents = pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    test_generator,
                    test_latents,
                )
        num_warmup_steps = len(timesteps) - test_num_inference_steps * pipe.scheduler.order
        with pipe.progress_bar(total=test_num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states = prompt_embeds, cross_attention_kwargs=None).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
            
            image = pipe.decode_latents(latents)
            image, has_nsfw_concept = pipe.run_safety_checker(image, device, prompt_embeds.dtype)
            image = pipe.numpy_to_pil(image)
    return image    