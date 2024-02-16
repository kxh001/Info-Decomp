import os
import textwrap
import torch as t
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

from utils import utils
from utils.itdiffusion import DiffusionModel
from utils.stablediffusion import StableDiffuser
from configs.visual_configs import parse_args_and_update_config

plt.style.use('seaborn-v0_8-paper')
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})

def min_max_norm(im):
    # min-max normalization
    im -= im.min()
    im /= im.max()
    return im

def add_noise(itd, sdm, latent_images, logsnrs):
    '''
     Calculate noisy images at multiple SNRs for visualization use.
    '''
    noisy_images_list = []
    for i in tqdm(range(len(logsnrs))):
        logsnr = logsnrs[i] * t.ones(latent_images.shape[0]).to(logsnrs.device)
        z, _ = itd.noisy_channel(latent_images, logsnr.to(latent_images.device))
        noisy_images = sdm.decode_latents(z)
        noisy_images_list.append(t.tensor(noisy_images))
    return t.stack(noisy_images_list)

def plot_mmse_curve(logsnrs, mi, mi_appx, cmi, cmi_appx):
    fig, ax = plt.subplots(2, 5, figsize=(23, 10))
    colors = ['skyblue', 'lightcoral']
    titles = ['airplane', 'bear', 'bed', 'cat', 'dog', 'elephant', 'horse', 'person', 'teddy bear', 'zebra']
    labels = ['$E[(\epsilon - \hat \epsilon_\\alpha(x))^2] - E[(\epsilon - \hat \epsilon_\\alpha(x|y*))^2]$',
              '$E[(\hat \epsilon_\\alpha(x) - \hat \epsilon_\\alpha(x|y_*))^2]$',
              '$E[(\epsilon - \hat \epsilon_\\alpha(x|c))^2] - E[(\epsilon - \hat \epsilon_\\alpha(x|y))^2]$',
              '$E[(\hat \epsilon_\\alpha(x|c) - \hat \epsilon_\\alpha(x|y))^2]$']
    for i in range(2):
        for j in range(5):
            ax[i][j].set_ylim(-15, 35)
            ax[i][j].plot(logsnrs, mi[i * 5 + j], label=labels[0], linestyle=':', color=colors[0])
            ax[i][j].plot(logsnrs, mi_appx[i * 5 + j], label=labels[1], linestyle='-', color=colors[0])
            ax[i][j].plot(logsnrs, cmi[i * 5 + j], label=labels[2], linestyle=':', color=colors[1])
            ax[i][j].plot(logsnrs, cmi_appx[i * 5 + j], label=labels[3], linestyle='-', color=colors[1])
            ax[i][j].set_title(f'{titles[i * 5 + j]}', fontsize=15)
            ax[i][j].set_ylabel('bits', fontsize=15)
            ax[i][j].set_xlabel('$\\alpha$', fontsize=15)
    ax[0][4].legend(loc='upper right', ncol=2, fontsize=15)
    return fig

def plot_mmse_and_mi_appx(img, nimgs, mi_appx, cmi_appx, mse_appx, cmse_appx, logsnrs):
    ylabels = ['Add noise', '$E[(\hat \epsilon_\\alpha(x) - \hat \epsilon_\\alpha(x|y_*))^2]$', '$E[(\hat \epsilon_\\alpha(x|c) - \hat \epsilon_\\alpha(x|y))^2]$']
    snr_num = len(logsnrs)
    row_num = len(ylabels)
    cmap = 'jet'
    fig, ax = plt.subplots(row_num, snr_num + 1, figsize=(14, 4.5), frameon=False)
    ax[0][-1].imshow(t.clamp(img, 0, 1))
    ax[0][-1].set_title('Real COCO')
    mse_appxs= min_max_norm(t.stack([mse_appx, cmse_appx]))

    for j in range(snr_num):
        nimg_ = t.clamp(nimgs[j], 0, 1)
        mse_appx_ = mse_appxs[0][j]
        cmse_appx_ = mse_appxs[1][j]
        ax[0][j].imshow(nimg_)
        ax[0][j].set_title(f"$\\alpha$ = {logsnrs[j]:.2f}")
        ax[1][j].imshow(mse_appx_, cmap=cmap)
        ax[2][j].imshow(cmse_appx_, cmap=cmap)

    mi_appx_ = min_max_norm(mi_appx)
    cmi_appx_ = min_max_norm(cmi_appx)
    ax[1][-1].imshow(mi_appx_, cmap=cmap)
    ax[1][-1].set_title('$\mathfrak{i}^o(x;y_*)$')
    ax[2][-1].imshow(cmi_appx_, cmap=cmap)
    ax[2][-1].set_title('$\mathfrak{i}^o(x;y_*|c)$')

    for i in range(row_num):
        for j in range(1, snr_num + 1):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    for i in range(row_num):
        if i > 0:
            ax[i][0].set_ylabel(ylabels[i], fontsize=7)
        else:
            ax[i][0].set_ylabel(ylabels[i])
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])

    return fig

def plot_heatmaps(img, caption, obj, mi, cmi, attn, type_):
    titles = ['Real COCO', '$\mathfrak{i}^o(x;y_*|c)$', '$\mathfrak{i}^o(x;y_*)$', 'Attention']
    sample_num = len(titles)
    cmap = 'jet'
    fig, ax = plt.subplots(1, sample_num, figsize=(7, 3))

    cmi_mi = min_max_norm(t.stack([cmi, mi]))
    cmi_ = cmi_mi[0]
    mi_ = cmi_mi[1]
    attn_ = min_max_norm(attn)
    img_ = img.permute(1, 2, 0)
    ax[0].imshow(t.clamp(img_, 0, 1))
    ax[1].imshow(cmi_, cmap=cmap, vmax=1, vmin=0)
    ax[2].imshow(mi_, cmap=cmap, vmax=1, vmin=0)
    ax[3].imshow(attn_, cmap=cmap, vmax=1, vmin=0)

    for i in range(sample_num):
        ax[i].set_title(titles[i], fontsize=14)
        ax[i].axis('off')

    # split caption if it's too long
    wrapped_text = "\n".join(textwrap.wrap(caption, width=52))
    text = f'c = {wrapped_text}\n$y_*$ = {obj} ({type_})'
    ax[0].text(0, 700 + 50 * len(wrapped_text.split('\n')), text, va="bottom", ha='left', fontsize=13)

    return fig

def plot_overlay(im, heatmap, fig, ax, normalize=False, vmax=None,
                 title=None, fontsize=14, last=False, inset_text=None):
    heatmap = heatmap.squeeze().cpu().numpy()
    if type(im) is Image.Image:
        im = np.array(im) / 255.
    else:
        im = (np.array(im) + 1) / 2  # 0..1 image
    ninety = np.percentile(heatmap.flatten(), 90)
    norm_heat = np.clip((heatmap - heatmap.min()) / (ninety- heatmap.min() + 1e-8), 0, 1)
    if normalize:
        x = norm_heat
        vmax = 1
    else:
        x = heatmap
        if vmax is None:
            vmax = heatmap.max()
    out = ax.imshow(x, vmin=0, vmax=vmax, cmap='jet')
    if last:
        cax = ax.inset_axes([1.02, 0.1, 0.1, 0.8])
        cax.axis('off')
        cbar = fig.colorbar(out, ax=cax, orientation="vertical", shrink=1)
        cbar.set_ticks([])  # Add ticks at 0 and the maximum value

    alpha = (1 - norm_heat)[:, :, np.newaxis]
    im = np.concatenate((im, alpha), axis=-1)
    ax.imshow(im)
    ax.set_title(title, fontsize=fontsize)
    ax.axis('off')
    if inset_text is not None:
        ax.text(
            0.05, 0.05,  # Adjust the coordinates for the starting position
            "{:.1f} bits total".format(inset_text),
            fontsize=fontsize,
            color="white",
            transform=ax.transAxes,  # Use axes-relative coordinates
            verticalalignment="bottom",  # Align text to the bottom
            horizontalalignment="left",  # Align text to the left
        )

def plot_img(im, ax, title=None, fontsize=14):
    if type(im) is Image.Image:
        im = np.array(im) / 255.
    else:
        im = (np.array(im) + 1) / 2  # 0..1 image
    ax.imshow(im)
    ax.set_title(title, fontsize=fontsize)
    ax.axis('off')


def plot_text(ax, c, y, yp, fontsize=14):
    ax.text(
        0.05, 0.9,  # Adjust the coordinates for the starting position of string1
        c,
        fontsize=fontsize,  # Adjust the font size as needed
        color='black',  # Color of string1 (black)
        verticalalignment="top",  # Align text to the top
        horizontalalignment="left",  # Align text to the left
    )
    ax.text(
        0.05, 0.5,  # Adjust the coordinates for the starting position of string1
        y,
        fontsize=fontsize,  # Adjust the font size as needed
        color='green',  # Color of string1 (black)
        verticalalignment="top",  # Align text to the top
        horizontalalignment="left",  # Align text to the left
    )
    ax.text(
        0.05, 0.25,  # Adjust the coordinates for the starting position of string1
        yp,
        fontsize=fontsize,  # Adjust the font size as needed
        color='red',  # Color of string1 (black)
        verticalalignment="top",  # Align text to the top
        horizontalalignment="left",  # Align text to the left
    )
    ax.axis('off')

def main():
    config = parse_args_and_update_config()
    t.manual_seed(config.seed)

    # set hyper-parameters
    fig_out_dir = config.fig_out_dir
    res_in_dir = config.res_in_dir
    data_in_dir = config.data_in_dir
    sdm_version = config.sdm_version
    csv_name = config.csv_name
    visual_type = config.visual_type

    if visual_type == 'mmse_curve':
        # assign noise levels
        logsnrs = t.linspace(-5.0, 7.0, 200)

        # load results
        in_path1 = os.path.join(res_in_dir, f'{sdm_version}-mse_1D-COCO10-logistic-50-200-mi.pt')
        in_path2 = os.path.join(res_in_dir, f'{sdm_version}-mse_1D-COCO10-logistic-50-200-cmi.pt')
        results1 = t.load(in_path1)
        results2 = t.load(in_path2)
        mi = results1['mmses'][:, 0, :] - results1['mmses'][:, 1, :]  # 10 * snr_num
        mi_appx = results1['mmses_diff_appx'][:, 0, :]  # 10 * snr_num
        cmi = results2['mmses'][:, 0, :] - results2['mmses'][:, 1, :]  # 10 * snr_num
        cmi_appx = results2['mmses_diff_appx'][:, 0, :]  # 10 * snr_num

        # visualization
        print('Starting visualization...')
        out_path = os.path.join(fig_out_dir, f"{sdm_version}/COCO10/mmse_gap-curves")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fig = plot_mmse_curve(logsnrs, -mi, mi_appx, -cmi, cmi_appx)
        fig.savefig(os.path.join(out_path, f"mmses.png"), dpi=300)
        print('Done')

    elif visual_type == 'scatter_plot':
        if csv_name == 'COCO100-IT':
            titles = ['airplane', 'bear', 'bed', 'cat', 'dog', 'elephant', 'horse', 'person', 'teddy bear', 'zebra']
        elif csv_name == 'COCO-WL':
            titles = ['verb', 'num', 'adj', 'adv', 'prep', 'pronoun', 'conj']

        # load results
        in_path1 = os.path.join(res_in_dir, f'{sdm_version}-nll_1D-{csv_name}-logistic-1-200-mi.pt')
        in_path2 = os.path.join(res_in_dir, f'{sdm_version}-nll_1D-{csv_name}-logistic-1-200-cmi.pt')
        results1 = t.load(in_path1)
        results2 = t.load(in_path2)
        mi = results1['pixel_mi'][:, 0]  # N * 3
        cmi = results2['pixel_mi'][:, 0]  # N * 3

        # calculate Pearson correlation
        correlation_coefficient = np.corrcoef(mi, cmi)[0, 1]

        # visualization
        print('Visualization starts ...')
        colors = plt.cm.rainbow(np.linspace(0, 1, len(titles)))
        plt.plot([0, max(mi.max(), cmi.max())], [0, max(mi.max(), cmi.max())], color='red', linestyle='--', label='$\mathfrak{i}^o(x;y_*) = \mathfrak{i}^o(x;y_*|c)$')
        for i in range(len(titles)):
            plt.scatter(mi[i * 10:(i + 1) * 10], cmi[i * 10:(i + 1) * 10], color=colors[i], alpha=0.5, s=20, label=titles[i])
        plt.text(80, 1, f'Pearson correlation: {correlation_coefficient :.2f}', fontsize=12, color='black')
        plt.legend(loc='upper left', ncol=2)
        plt.title(f'{csv_name}', fontsize=14)
        plt.xlabel('$\mathfrak{i}^o(x;y_*)$', fontsize=13)
        plt.ylabel('$\mathfrak{i}^o(x;y_*|c)$', fontsize=13)
        out_path = os.path.join(fig_out_dir, f"{sdm_version}/{csv_name}/")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        plt.savefig(os.path.join(out_path, f"scatter-plot-{csv_name}.png"), dpi=300)
        print('Done')

    elif visual_type == 'denoising_diffusion':
        # load diffusion models
        if sdm_version == 'sdm_2_0_base':
            sdm = StableDiffuser("stabilityai/stable-diffusion-2-base")
        elif sdm_version == 'sdm_2_1_base':
            sdm = StableDiffuser("stabilityai/stable-diffusion-2-1-base")
        latent_shape = (sdm.channels, sdm.width, sdm.height)
        itd = DiffusionModel(sdm.unet, latent_shape, logsnr_loc=1.0, logsnr_scale=2.0, clip=3.0, logsnr2t=sdm.logsnr2t)

        # assign noise levels
        logsnrs = t.linspace(-5.0, 7.0, 10).to(sdm.device)

        # load data
        # TODO: change CSV file to JSON file
        img_dir = os.path.join(data_in_dir, f'val2017')
        csv_dir = os.path.join(data_in_dir, f'{csv_name}.csv')
        annotation_file = os.path.join(data_in_dir, f'annotations/instances_val2017.json')
        dataset = utils.CocoDataset(img_dir, annotation_file, csv_dir)
        dataloader = t.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
        img_list = []
        n_img_list = []
        for batch_idx, batch in enumerate(dataloader):
            image_batch = batch['image']
            latent_images = sdm.encode_latents(image_batch)
            print('Add noise to image...')
            noisy_images_batch = add_noise(itd, sdm, latent_images, logsnrs)
            print('Done\n')
            n_img_list.append(noisy_images_batch)
            img_list.append(image_batch)
        images = t.cat(img_list).squeeze()
        images = images.permute(0, 2, 3, 1)
        noisy_images = t.cat(n_img_list)
        noisy_images = noisy_images.permute(1, 0, 2, 3, 4)

        # load results
        in_path1 = os.path.join(res_in_dir, f'{sdm_version}-nll_2D-COCO10-logistic-1-200-mi.pt')
        in_path2 = os.path.join(res_in_dir, f'{sdm_version}-nll_2D-COCO10-logistic-1-200-cmi.pt')
        in_path3 = os.path.join(res_in_dir, f'{sdm_version}-mse_2D-COCO10-logistic-50-10-mi.pt')
        in_path4 = os.path.join(res_in_dir, f'{sdm_version}-mse_2D-COCO10-logistic-50-10-cmi.pt')
        results1 = t.load(in_path1)
        results2 = t.load(in_path2)
        results3 = t.load(in_path3)
        results4 = t.load(in_path4)
        mi_appx = results1['pixel_mi'][:, 0, :, :]  # 100 * h * w
        cmi_appx = results2['pixel_mi'][:, 0, :, :]  # 100 * h * w
        mse_appx = results3['pixel_mmses_diff_appx'][:, 0, :, :]  # 100 * h * w
        cmse_appx = results4['pixel_mmses_diff_appx'][:, 0, :, :]  # 100 * h * w

        # visualization
        print('Starting visualization...')
        for i in tqdm(range(len(images))):
            out_path = os.path.join(fig_out_dir, f"{sdm_version}/{csv_name}")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fig = plot_mmse_and_mi_appx(images[i], noisy_images[i], mi_appx[i], cmi_appx[i], mse_appx[i], cmse_appx[i], logsnrs)
            plt.subplots_adjust(wspace=0, hspace=0.15)
            fig.savefig(os.path.join(out_path, f"denoising_diffusion-{i}.png"), dpi=300)
            plt.close()
        print('Done')

    elif visual_type == 'mi_cmi_attn':
        # load data
        # TODO: change CSV file to JSON file
        img_dir = os.path.join(data_in_dir, f'val2017')
        csv_dir = os.path.join(data_in_dir, f'{csv_name}.csv')
        annotation_file = os.path.join(data_in_dir, f'annotations/instances_val2017.json')
        dataset = utils.CocoDataset(img_dir, annotation_file, csv_dir)
        dataloader = t.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        img_list = []
        word_list = []
        context_list = []
        obj_list = []
        for batch_idx, batch in enumerate(dataloader):
            image_batch = batch['image']
            word_batch = batch['category']
            context_batch = batch['context_']
            obj_batch = batch['cobj']
            img_list.append(image_batch)
            word_list.append(word_batch)
            context_list.append(context_batch)
            obj_list.append(obj_batch)
        images = t.cat(img_list).squeeze()
        contexts = np.array(context_list)[0]
        objs = np.array(obj_list)[0]

        # load results: mi_2D and attnmaps
        in_path1 = f'./results/itd/{sdm_version}-nll_2D-{csv_name}-logistic-1-100-mi.pt'
        in_path2 = f'./results/itd/{sdm_version}-nll_2D-{csv_name}-logistic-1-100-cmi.pt'
        in_path3 = f'./results/daam/{sdm_version}-attnmaps-{csv_name}-steps=100.pt'
        results1 = t.load(in_path1)
        results2 = t.load(in_path2)
        results3 = t.load(in_path3)
        mi = results1['pixel_mi'][:, 0, :, :]  # N * h * w
        cmi = results2['pixel_mi'][:, 0, :, :]  # N * h * w
        attn = results3['attnmaps']  # N * h * w

        # visualization
        print('Starting visualization...')
        out_path = os.path.join(fig_out_dir, f"{sdm_version}/{csv_name}/")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if csv_name == 'COCO100-IT':
            type_ = 'noun'
            for i in tqdm(range(len(images))):
                fig = plot_heatmaps(images[i], contexts[i], objs[i], mi[i], cmi[i], attn[i], type_)
                fig.savefig(os.path.join(out_path, f"{i}.png"), dpi=300)
                plt.close(fig)
        elif csv_name == 'COCO-WL':
            types = ['verb', 'num', 'adj', 'adv', 'prep', 'pronoun', 'conj']
            for i in tqdm(range(len(images))):
                type_ = types[i // 10]
                fig = plot_heatmaps(images[i], contexts[i], objs[i], mi[i], cmi[i], attn[i], type_)
                fig.savefig(os.path.join(out_path, f"mi_cmi_attn-{i}.png"), dpi=300)
                plt.close(fig)
        print('Done')

    elif visual_type == 'image_edit':
        df = pd.read_csv('./datasets/COCO100-IT.csv')

        for i, row in df.iterrows():
            img_name = f"{row['Image ID']:012d}.jpg"
            path = os.path.join(res_in_dir, img_name)

            try:
                loaded_data = t.load(path[:-4] + '.pt')

                mod_prompt = loaded_data['mod_prompt']
                object = loaded_data['object']
                word_swap = loaded_data['word_swap']
                x_real_transformed = loaded_data['x_real_transformed'].cpu()
                recover_real_decode = loaded_data['recover_real_decode']
                recover_mod_decode = loaded_data['recover_mod_decode']
                recover_swap_decode = loaded_data['recover_swap_decode']
                cmi_pixel = loaded_data['cmi_pixel'].cpu()
                cmi_pixel_swap = loaded_data['cmi_pixel_swap'].cpu()
                mi_pixel = loaded_data['mi_pixel'].cpu()
                heat_map = loaded_data['heat_map'].cpu()
                mi = loaded_data['mi']
                cmi = loaded_data['cmi']
                cmi_swap = loaded_data['cmi_swap']

                # Build the plots for each image
                k = 7  # text, real, real rec, edited, CMI, MI, attention
                fig1, axs1 = plt.subplots(1, k, figsize=(3 * k, 3))  # Adjust the figsize as needed
                k = 6  # text, real, real rec, edited, CMI, attention
                fig3, axs3 = plt.subplots(1, k, figsize=(3 * k, 3))  # Adjust the figsize as needed
                k = 5  # text, real, edited, CMI, attention
                fig2, axs2 = plt.subplots(1, k, figsize=(3 * k, 3))
                fig4, axs4 = plt.subplots(1, k, figsize=(3 * k, 3))

                w = 22
                wrap1 = '\n'.join(textwrap.wrap('c = ' + mod_prompt, width=w))
                wrap2 = '\n'.join(textwrap.wrap('y = ' + object, width=w))
                wrap3 = '\n'.join(textwrap.wrap("y' = " + word_swap, width=w))

                plot_text(axs1[0], wrap1, wrap2, '')
                plot_text(axs2[0], wrap1, wrap2, '')
                plot_text(axs3[0], wrap1, wrap2, wrap3)
                plot_text(axs4[0], wrap1, wrap2, wrap3)

                plot_img(x_real_transformed, axs1[1], title='Real COCO image')
                plot_img(x_real_transformed, axs2[1], title='Real COCO image')
                plot_img(x_real_transformed, axs3[1], title='Real COCO image')
                plot_img(x_real_transformed, axs4[1], title='Real COCO image')

                plot_img(recover_real_decode, axs1[2], title='Reconstructed')
                plot_img(recover_real_decode, axs3[2], title='Reconstructed')

                plot_img(recover_mod_decode, axs1[3], title='Intervention')
                plot_img(recover_mod_decode, axs2[2], title='Intervention')
                plot_img(recover_swap_decode, axs3[3], title='Intervention')
                plot_img(recover_swap_decode, axs4[2], title='Intervention')

                # add total CMI
                thresh = 0.1
                plot_overlay(x_real_transformed, cmi_pixel, fig1, axs1[4], title='$\mathfrak{i}(x;y|c)$',
                             normalize=False,
                             vmax=thresh, inset_text=cmi)
                plot_overlay(x_real_transformed, cmi_pixel, fig2, axs2[3], title='$\mathfrak{i}(x;y|c)$',
                             normalize=False,
                             vmax=thresh, inset_text=cmi)
                plot_overlay(x_real_transformed, cmi_pixel_swap, fig3, axs3[4],
                             title="$\mathfrak{i}(x;y|c) - \mathfrak{i}(x;y'|c)$", normalize=False, vmax=thresh,
                             inset_text=cmi_swap)
                plot_overlay(x_real_transformed, cmi_pixel_swap, fig4, axs4[3],
                             title="$\mathfrak{i}(x;y|c) - \mathfrak{i}(x;y'|c)$", normalize=False, vmax=thresh,
                             inset_text=cmi_swap)

                # add MI for one
                plot_overlay(x_real_transformed, mi_pixel, fig1, axs1[5], title='$\mathfrak{i}(x;y)$', normalize=False,
                             vmax=thresh,
                             inset_text=mi)

                # Attention
                plot_overlay(x_real_transformed, heat_map, fig1, axs1[6], title='Attention', normalize=True, last=True)
                plot_overlay(x_real_transformed, heat_map, fig2, axs2[4], title='Attention', normalize=True, last=True)
                plot_overlay(x_real_transformed, heat_map, fig3, axs3[5], title='Attention', normalize=True, last=True)
                plot_overlay(x_real_transformed, heat_map, fig4, axs4[4], title='Attention', normalize=True, last=True)

                fig1.savefig(os.path.join(fig_out_dir, 'mod_' + img_name))
                fig2.savefig(os.path.join(fig_out_dir, 'mod2_' + img_name))
                fig3.savefig(os.path.join(fig_out_dir, 'swap_' + img_name))
                fig4.savefig(os.path.join(fig_out_dir, 'swap2_' + img_name))
                plt.close()
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()


