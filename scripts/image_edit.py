import os.path

import torch as t
import pandas as pd
from PIL import Image
from daam import trace
import torch.nn.functional as F

from utils import utils
from utils.itdiffusion import DiffusionModel  # Info theoretic diffusion library and flow sampler
from utils.stablediffusion import StableDiffuser

from configs.image_edit_configs import parse_args_and_update_config


def main():
    config = parse_args_and_update_config()
    t.manual_seed(config.seed)

    # set hyper-parameters
    data_in_dir = config.data_in_dir
    res_out_dir = config.res_out_dir
    n_samples_per_point = config.n_samples_per_point
    batch_size = config.batch_size
    num_steps = config.num_steps
    sdm_version = config.sdm_version
    clip = config.clip

    # load diffusion models
    if sdm_version == 'sdm_2_0_base':
        sdm = StableDiffuser("stabilityai/stable-diffusion-2-base")
    elif sdm_version == 'sdm_2_1_base':
        sdm = StableDiffuser("stabilityai/stable-diffusion-2-1-base")

    logsnr_max, logsnr_min = sdm.logsnr_max, sdm.logsnr_min
    logsnr_loc = logsnr_min + 0.5 * (logsnr_max - logsnr_min)
    logsnr_scale = (1. / (2. * clip)) * (logsnr_max - logsnr_min)

    latent_shape = (sdm.channels, sdm.width, sdm.height)
    itd = DiffusionModel(sdm.unet, latent_shape, logsnr_loc=logsnr_loc, logsnr_scale=logsnr_scale, clip=clip,
                         logsnr2t=sdm.logsnr2t).to(sdm.device)

    # Defines range of sigma/snr to use during sampling, based on training
    
    sigma_min, sigma_max = utils.logsnr2sigma(logsnr_max), utils.logsnr2sigma(logsnr_min)

    # Set schedule in Karras et al terms, "sigmas", where z = x + sigma epsilon.
    schedule = utils.get_sigmas_karras(num_steps, sigma_min, sigma_max, device=itd.device)
    # For generation, use schedule. For reversible sampling use the following, which
    # doesn't go all the way to the limit sigma=0, snr=inf We can't approx score there so can't reverse
    schedule_reversible = schedule[:-1]

    # Step function for ODE flow. Choose second order "Heun" solver, s_churn = 0. gives deterministic
    step_function = utils.get_step(order=2, s_churn=0.)

    word_pairs = [
        ("dog", "cat"),
        ("zebra", "horse"),
        ("bed", "table"),
        ("bear", "elephant"),
        ("airplane", "kite"),
        ("person", "monkey"),
        ("people", "monkeys"),
        ("teddy bear", "robot")
    ]
    word_swaps = {}  # Create an empty dictionary for word swaps
    for a, b in word_pairs:
        word_swaps[a] = b
        word_swaps[b] = a
        word_swaps[a + 's'] = b + 's'
        word_swaps[b + 's'] = a + 's'

    # TODO: change the data pipeline
    df = pd.read_csv('./datasets/coco/COCO100-IT.csv')
    for i, row in df.iterrows():
        try:
            img_name = f"{row['Image ID']:012d}.jpg"
            path = os.path.join(data_in_dir, img_name)
            print("Opening", path)
            img = Image.open(path).convert('RGB')
            object, prompt = row['correct_obj'], row['correct_obj+context']
            mod_prompt = utils.perform_word_swaps(prompt, {object: '_', object + 's': '_'})
            word_swap = word_swaps.get(object, object)
            swap_prompt = utils.perform_word_swaps(prompt, word_swaps)

            # Encode image to SD latent space
            x_real_transformed = sdm.sdm_pipe.image_processor.preprocess(img, height=512, width=512).squeeze().permute((1, 2, 0))
            x_real = sdm.encode_latents(img)

            # Encode prompts to CLIP embedding space
            v_org = sdm.encode_prompts(prompt).expand(batch_size, -1, -1)
            v_null = sdm.encode_prompts('').expand(batch_size, -1, -1)
            v_obj = sdm.encode_prompts(object).expand(batch_size, -1, -1)
            v_mod = sdm.encode_prompts(mod_prompt).expand(batch_size, -1, -1)
            v_swap = sdm.encode_prompts(swap_prompt).expand(batch_size, -1, -1)

            # Run in reverse to get the latent
            latent_real = utils.reverse(sdm, step_function, schedule_reversible, x_real, prompt)
            
            ######################################################
            ##                   No intervention                ##
            ######################################################
            # Then run forward (no intervention) and check recovery of real image - also track attention
            with t.cuda.amp.autocast(dtype=t.float16), t.no_grad():
                with trace(sdm.sdm_pipe) as tc:
                    recover_real = utils.generate(sdm, step_function, schedule_reversible, latent_real, prompt)

            # Recover heat map
            heat_map = tc.compute_global_heat_map()
            heat_map = heat_map.compute_word_heat_map(object)  # keyword
            heat_map_lr = heat_map.value
            heat_map = F.interpolate(heat_map.value.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0).squeeze(0)
            # Decode real image without intervention
            recover_real_decode = sdm.decode_latents(recover_real)[0]

            ######################################################
            ##              Omit & Swap intervention            ##
            ######################################################
            # Then run with a change in the prompt
            recover_mod = utils.generate(sdm, step_function, schedule_reversible, latent_real, mod_prompt)
            recover_mod_decode = sdm.decode_latents(recover_mod)[0]

            # Same, but swap a word instead
            recover_swap = utils.generate(sdm, step_function, schedule_reversible, latent_real, swap_prompt)
            recover_swap_decode = sdm.decode_latents(recover_swap)[0]

            # Get Info heat maps
            with t.cuda.amp.autocast(dtype=t.float16), t.no_grad():
                results_dict = itd.ll_ratio(x_real, v_null, v_obj, n_points=20,
                                           n_samples_per_point=n_samples_per_point,
                                           batch_size=batch_size, dim=(1,))
            mi_pixel_lr = results_dict['ll_ratio_pixel_app']
            mi_pixel = F.interpolate(results_dict['ll_ratio_pixel_app'].unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0).squeeze(0)
            mi = results_dict['ll_ratio_pixel'].sum()

            with t.cuda.amp.autocast(dtype=t.float16), t.no_grad():
                results_dict = itd.ll_ratio(x_real, v_mod, v_org, n_points=20,
                                           n_samples_per_point=n_samples_per_point,
                                           batch_size=batch_size, dim=(1,))
            cmi_pixel_lr = results_dict['ll_ratio_pixel_app']
            cmi_pixel = F.interpolate(results_dict['ll_ratio_pixel_app'].unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0).squeeze(0)
            cmi = results_dict['ll_ratio_pixel'].sum()

            with t.cuda.amp.autocast(dtype=t.float16), t.no_grad():
                results_dict = itd.ll_ratio(x_real, v_swap, v_org, n_points=20,
                                           n_samples_per_point=n_samples_per_point,
                                           batch_size=batch_size, dim=(1,))
            cmi_pixel_swap_lr = results_dict['ll_ratio_pixel_app']
            cmi_pixel_swap = F.interpolate(results_dict['ll_ratio_pixel_app'].unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0).squeeze(0)
            cmi_swap = results_dict['ll_ratio_pixel'].sum()

            data_to_save = {
                'mod_prompt': mod_prompt,
                'object': object,
                'word_swap': word_swap,
                'x_real_transformed': x_real_transformed,
                'recover_real_decode': recover_real_decode,
                'recover_mod_decode': recover_mod_decode,
                'recover_swap_decode': recover_swap_decode,
                'recover_mod': recover_mod,
                'recover_swap': recover_swap,
                'recover_real': recover_real,
                'cmi_pixel': cmi_pixel,
                'cmi_pixel_swap': cmi_pixel_swap,
                'mi_pixel': mi_pixel,
                'cmi_pixel_lr': cmi_pixel_lr,
                'cmi_pixel_swap_lr': cmi_pixel_swap_lr,
                'mi_pixel_lr': mi_pixel_lr,
                'heat_map_lr': heat_map_lr,
                'heat_map': heat_map,
                'cmi': cmi,
                'cmi_swap': cmi_swap,
                'mi': mi
            }

            # Save the dictionary to a file
            if not os.path.exists(res_out_dir):
                os.makedirs(res_out_dir)
            t.save(data_to_save, os.path.join(res_out_dir, img_name[:-4] + '.pt'))
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
