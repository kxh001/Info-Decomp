import os
import torch as t
from tqdm import tqdm
import torch.nn.functional as F

from utils import utils
from utils.itdiffusion import DiffusionModel
from utils.stablediffusion import StableDiffuser
from configs.eval_itd_configs import parse_args_and_update_config

def main():
    config = parse_args_and_update_config()
    t.manual_seed(config.seed)

    # set hyper-parameters
    res_type = config.res_type
    sdm_version = config.sdm_version
    eval_metrics = config.eval_metrics
    res_out_dir = config.res_out_dir
    data_in_dir = config.data_in_dir
    csv_name = config.csv_name
    z_sample_num = config.n_samples_per_point
    snr_num = config.num_steps
    bs = config.batch_size
    logsnr_loc = config.logsnr_loc
    logsnr_scale = config.logsnr_scale
    clip = config.clip
    upscale_mode = config.upscale_mode
    int_mode = config.int_mode

    print(f'sdm: {sdm_version} | result: {res_type} | dataset: {csv_name} | integration: {int_mode} '
          f'| upscale: {upscale_mode} | z sample #: {z_sample_num} | snr #: {snr_num} | eval metrics: {eval_metrics}')

    # load diffusion models
    if sdm_version == 'sdm_2_0_base':
        sdm = StableDiffuser("stabilityai/stable-diffusion-2-base")
    elif sdm_version == 'sdm_2_1_base':
        sdm = StableDiffuser("stabilityai/stable-diffusion-2-1-base")

    latent_shape = (sdm.channels, sdm.width, sdm.height)
    itd = DiffusionModel(sdm.unet, latent_shape, logsnr_loc=logsnr_loc, logsnr_scale=logsnr_scale, clip=clip, logsnr2t=sdm.logsnr2t)

    # load data
    # TODO: convert it from CSV file to JSON file
    img_dir = os.path.join(data_in_dir, f'val2017')
    csv_dir = os.path.join(data_in_dir, f'{csv_name}.csv')
    annotation_file = os.path.join(data_in_dir, f'annotations/instances_val2017.json')
    dataset = utils.CocoDataset(img_dir, annotation_file, csv_dir)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

    # assign noise levels
    logsnrs = t.linspace(logsnr_loc - clip * logsnr_scale, logsnr_loc + clip * logsnr_scale, snr_num).to(sdm.device)

    # evaluate
    with t.no_grad():
        results = {}
        mmses_list = []
        mmses_diff_appx_list = []
        pixel_mmses_list = []
        pixel_mmses_diff_appx_list = []
        nll_list = []
        mi_appx_list = []
        pixel_nll_list = []
        pixel_mi_appx_list = []

        for batch_idx, batch in tqdm(enumerate(dataloader)):
            if eval_metrics == 'mi':
                images = batch['image']
                correct_prompts = batch['category']
                wrong_prompts = [""] * len(images)
                none_prompts = [""] * len(images)
            elif eval_metrics == 'cmi':
                images = batch['image']
                correct_prompts = batch['caption']
                wrong_prompts = batch['context']
                none_prompts = [""] * len(images)
            else:
                assert "Please input the correct strategy type, choosing among: 'mi', 'cmi'. "

            # compute latent variable z and text embeddings in stable diffusion model
            latent_images = sdm.encode_latents(images)
            text_embeddings = sdm.encode_prompts(correct_prompts)
            wro_embeddings = sdm.encode_prompts(wrong_prompts)
            uncond_embeddings = sdm.encode_prompts(none_prompts)
            conds = t.stack([text_embeddings, wro_embeddings, uncond_embeddings])

            if res_type == 'mse_1D':
                print(f'Calculate image-level mmses and its difference for Batch {batch_idx}...')
                mmses, mmses_diff_appx = itd.image_level_mmse(latent_images, conds, logsnrs, total=z_sample_num)

                # Post-process results
                mmses = mmses.permute(2, 1, 0) # bs * 3 * snr_num
                mmses_diff_appx = mmses_diff_appx.permute(2, 1, 0)  # bs * 3 * snr_num
                mmses_list.append(mmses)
                mmses_diff_appx_list.append(mmses_diff_appx)
                print('Done\n')
            elif res_type == 'mse_2D':
                print(f'Calculate pixel-level mmses and its difference for Batch {batch_idx}...')
                pixel_mmses, pixel_mmses_diff_appx = itd.pixel_level_mmse(latent_images, conds, logsnrs, total=z_sample_num)

                # Post-process results
                pixel_mmses_up = t.zeros(list(pixel_mmses.shape[:-2]) + [512, 512])
                pixel_mmses_diff_appx_up = t.zeros(list(pixel_mmses_diff_appx.shape[:-2]) + [512, 512])
                for i in range(snr_num):
                    pixel_mmses_up[i] = F.interpolate(pixel_mmses[i], size=(512, 512), mode=upscale_mode)
                    pixel_mmses_diff_appx_up[i] = F.interpolate(pixel_mmses_diff_appx[i], size=(512, 512), mode=upscale_mode)
                pixel_mmses_up = pixel_mmses_up.permute(2, 1, 0, 3, 4)  # bs * 3 * snr_num * h * w
                pixel_mmses_diff_appx_up = pixel_mmses_diff_appx_up.permute(2, 1, 0, 3, 4)  # bs * 3 * snr_num * h * w
                pixel_mmses_list.append(pixel_mmses_up)
                pixel_mmses_diff_appx_list.append(pixel_mmses_diff_appx_up)
                print('Done\n')
            elif res_type == 'nll_1D':
                print(f'Calculate image-level nll and {eval_metrics} for Batch {batch_idx}...')
                # nll, mi_appx = itd.image_level_nll_fast(latent_images, conds, total=snr_num, int_mode=int_mode)
                nll, mi_appx = itd.image_level_nll(latent_images, conds, snr_num=snr_num, z_sample_num=z_sample_num, int_mode=int_mode)

                # Post-process results
                nll = nll.permute(1, 0) # bs * 3
                mi_appx = mi_appx.permute(1, 0) # bs * 3
                nll_list.append(nll)
                mi_appx_list.append(mi_appx)
                print('Done\n')
            elif res_type == 'nll_2D':
                print(f'Calculate pixel-level nll and {eval_metrics} for Batch {batch_idx}...')
                # pixel_nll, pixel_mi_appx = itd.pixel_level_nll_fast(latent_images, conds, total=snr_num, int_mode=int_mode)
                pixel_nll, pixel_mi_appx = itd.pixel_level_nll(latent_images, conds, snr_num=snr_num, z_sample_num=z_sample_num, int_mode=int_mode)

                # Post-process results
                pixel_nll = F.interpolate(pixel_nll, size=(512, 512), mode=upscale_mode)  # 3 * bs * h' * w'
                pixel_mi_appx = F.interpolate(pixel_mi_appx, size=(512, 512), mode=upscale_mode)  # 3 * bs * h' * w'
                pixel_nll = pixel_nll.permute(1, 0, 2, 3) # bs * 3 * h' * w'
                pixel_mi_appx = pixel_mi_appx.permute(1, 0, 2, 3) # bs * 3 * h' * w'
                pixel_nll_list.append(pixel_nll)
                pixel_mi_appx_list.append(pixel_mi_appx)
                print('Done\n')
            else:
                assert "Please input the correct results type, choosing among: 'mse_1D', 'mse_2D', 'nll_1D', 'nll_2D'."


        if res_type == 'mse_1D':
            results['mmses'] = t.cat(mmses_list)  # N * 3 * snr_num
            results['mmses_diff_appx'] = t.cat(mmses_diff_appx_list)  # N * 3 * snr_num
        elif res_type == 'mse_2D':
            results['pixel_mmses'] = t.cat(pixel_mmses_list)  # N * 3 * snr_num
            results['pixel_mmses_diff_appx'] = t.cat(pixel_mmses_diff_appx_list)  # N * 3 * snr_num
        elif res_type == 'nll_1D':
            results['nll'] = t.cat(nll_list)
            results['mi'] = t.cat(mi_appx_list)
        elif res_type == 'nll_2D':
            results['pixel_nll'] = t.cat(pixel_nll_list)
            results['pixel_mi'] = t.cat(pixel_mi_appx_list)

        # save results
        if not os.path.exists(res_out_dir):
            os.makedirs(res_out_dir)
        out_file_name = f'{sdm_version}-{res_type}-{csv_name}-{int_mode}-{z_sample_num}-{snr_num}-{eval_metrics}.pt'
        out_path = os.path.join(res_out_dir, out_file_name)
        t.save(results, out_path)
        print(f'Results are saved to: {out_path}')

if __name__ == "__main__":
    main()