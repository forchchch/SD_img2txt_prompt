import sys
sys.path.append("../")
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from lora_diffusion import tune_lora_scale, patch_pipe
from PIL import Image



def dreambooth_save(pipe, prompt, save_dir, guidance_scale):
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    tune_lora_scale(pipe.unet, 1.0)
    tune_lora_scale(pipe.text_encoder, 1.0)
    image = pipe(prompt, num_inference_steps=50, guidance_scale=guidance_scale).images[0]
    image.save(save_dir)


def joint_visualization_train(pipe, i2t_prompt_model, prompt, guidance, save_dir, preprocess, eta=0.5):
    num_images_per_prompt = 1
    test_num_inference_steps = 50
    ref_image1 = preprocess(Image.open("/DATA/DATANAS1/chenhong/diffusion_research/dreambooth_data/backpack/03.jpg")).unsqueeze(0).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    with torch.no_grad():
        
        img_feature = i2t_prompt_model.encode_image( ref_image1 ).unsqueeze(1) 
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
        tv_prompt = eta*img_feature + prompt_embeds
        #tv_prompt = torch.cat([prompt_embeds,i2t_emb], dim=1)

        pipe.scheduler.set_timesteps(test_num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        num_channels_latents = pipe.unet.in_channels
        height = 512
        width = 512
        latents = pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    None,
                    None,
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
    print("hhhhh")
    image[0].save(save_dir)

def joint_visualization(pipe, i2t_prompt_model, prompt, reference_image, guidance, eta=0.5):
    num_images_per_prompt = 1
    test_num_inference_steps = 50
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    with torch.no_grad():
        
        img_feature = i2t_prompt_model.encode_image( reference_image ).unsqueeze(1) 
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
        tv_prompt = eta*img_feature + prompt_embeds
        #tv_prompt = torch.cat([prompt_embeds,i2t_emb], dim=1)

        pipe.scheduler.set_timesteps(test_num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        num_channels_latents = pipe.unet.in_channels
        height = 512
        width = 512
        latents = pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    None,
                    None,
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
    return image