import os
import torch as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryJaccardIndex

from utils import utils
from configs.calc_configs import parse_args_and_update_config

def min_max_norm(hm):
    hm -= hm.amin(dim=(-1, -2), keepdim=True)
    hm /= hm.amax(dim=(-1, -2), keepdim=True)
    return hm

def cat_file_name(sdm_version, res_type, csv_name, int_mode, z_sample_num, snr_num, eval_metrics):
    if res_type == 'attnmaps':
        in_file_name = f'{sdm_version}-{res_type}-{csv_name}-{snr_num}.pt'
    else:
        in_file_name = f'{sdm_version}-{res_type}-{csv_name}-{int_mode}-{z_sample_num}-{snr_num}-{eval_metrics}.pt'
    return in_file_name

def main():
    config = parse_args_and_update_config()
    t.manual_seed(config.seed)
    bijacidx = BinaryJaccardIndex()

    # set hyper-parameters
    res_in_dir = config.res_in_dir
    output_dir = config.output_dir
    data_in_dir = config.data_in_dir
    sdm_version = config.sdm_version
    z_sample_num = config.n_samples_per_point
    csv_name = config.csv_name
    snr_num = config.num_steps
    bs = config.batch_size
    int_mode = config.int_mode
    calc_type = config.calc_type
    res_type = config.res_type
    eval_metrics = config.eval_metrics

    print(f'sdm: {sdm_version} | result: {res_type} | dataset: {csv_name} | integration: {int_mode} '
          f'| sample #: {z_sample_num} | snr #: {snr_num} | eval metrics: {eval_metrics} | calc type: {calc_type}')

    # load data
    img_dir = os.path.join(data_in_dir, f'val2017')
    csv_dir = os.path.join(data_in_dir, f'{csv_name}.csv')
    annotation_file = os.path.join(data_in_dir, f'annotations/instances_val2017.json')
    dataset = utils.CocoDataset(img_dir, annotation_file, csv_dir)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

    # load masks
    mask_list = []
    for batch_idx, batch in enumerate(dataloader):
        mask_batch = batch['mask']
        mask_list.append(mask_batch)
    masks = t.cat(mask_list).squeeze().to(t.int)  # dataloader.dataset.__len__() * 1 * h * w

    if calc_type == 'iou_baseline':
        # test the whole image mask IoU
        print("Analyze variance of IoU for the whole image mask ...")
        pred_masks = t.ones_like(masks)
        ious = [bijacidx(pred_masks[i], masks[i]).item() for i in range(len(masks))]
        stds = np.array(ious).std()
        miou = np.array(ious).mean()
        print(f'The mIoU is {miou}, and IoU variance is {stds}.')

    elif calc_type == 'iou':
        # load pixel_mi or attnmaps
        in_file_name = cat_file_name(sdm_version, res_type, csv_name, int_mode, z_sample_num, snr_num, eval_metrics)
        in_path = os.path.join(res_in_dir, in_file_name)
        results = t.load(in_path)
        if res_type == 'nll_2D':
            '''
               The pixel_mi gives 3 attributes at dim=1:
               [pixel_mi(eps_hat_c - eps_hat_w), pixel_mi(eps_hat_c - eps_hat_n), pixel_mi(eps_hat_w - eps_hat_n)]
            '''
            heatmaps = results['pixel_mi'][:, 0, :, :].squeeze(1)  # # N * 3 * h * w
        else:
            heatmaps = results['attnmaps']  # N * h * w

        # plot mIoU curve
        print("Plot mIoU curves ...")
        mious = []
        thresholds = t.linspace(0.1, 0.5, 50)
        heatmaps = min_max_norm(heatmaps)
        for threshold in thresholds:
            threshold_map = t.ones_like(heatmaps) * threshold  # hard threshold map
            pred_masks = utils.process_mask(heatmaps, threshold_map, is_soft=False)
            ious = [bijacidx(pred_masks[i], masks[i]).item() for i in range(len(masks))]
            miou = sum(ious) / len(masks)
            mious.append(miou)

        plt.plot(thresholds, mious, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('mIoU')
        plt.title(f'max mIoU: {100 * max(mious):.2f}')
        plt.grid()

        iou_out_dir = os.path.join(output_dir, f'iou_curves')
        if not os.path.exists(iou_out_dir):
            os.makedirs(iou_out_dir)
        out_path = os.path.join(iou_out_dir, f'{sdm_version}-{snr_num}-{eval_metrics}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print('Done.')

        # IoU variance analysis
        print("Analyze variance of IoU ...")
        threshold = thresholds[t.tensor(mious).argmax()]
        threshold_map = t.ones_like(heatmaps) * threshold  # hard threshold map
        pred_masks = utils.process_mask(heatmaps, threshold_map, is_soft=False)
        ious = [bijacidx(pred_masks[i], masks[i]).item() for i in range(len(masks))]
        stds = np.array(ious).std()
        miou = np.array(ious).mean()
        print(f'The best mIoU is {miou}, and IoU variance is {stds}.')

    elif calc_type == 'pearson':
        df = pd.read_csv(f'./dataset/COCO100-IT.csv')
        pixel_changes, attentions_pixel, cmis_pixel = [], [], []
        pixel_changes_score, attentions, cmis = [], [], []
        for i, row in df.iterrows():
            img_name = f"{row['Image ID']:012d}.jpg"
            path = os.path.join(res_in_dir, img_name)
            print("Opening", path)
            try:
                loaded_data = t.load(path[:-4] + '.pt')

                cmi_pixel = loaded_data['cmi_pixel_lr'].cpu().numpy()
                heat_map = loaded_data['heat_map_lr'].cpu().numpy()
                recover_real = loaded_data['recover_real'].cpu().numpy()
                recover_mod = loaded_data['recover_mod'].cpu().numpy()
                cmi = loaded_data['cmi'].cpu().numpy()

                # pixel change in omit experiments
                pixel_change = np.square(np.array(recover_real) - np.array(recover_mod)).sum(axis=1).squeeze()

                # pixel-level
                pixel_changes.append(pixel_change.flatten())
                attentions_pixel.append(heat_map.flatten())
                cmis_pixel.append(cmi_pixel.flatten())
                # image-level
                pixel_changes_score.append(pixel_change.sum())
                attentions.append(heat_map.sum())
                cmis.append(cmi)

            except Exception as e:
                print(f"An error occurred: {e}")

        corr_attn_score = np.corrcoef(attentions, pixel_changes_score)[0, 1]
        corr_cmi_score = np.corrcoef(cmis, pixel_changes_score)[0, 1]

        print('Image level:')
        print('Correlation between attention and pixel change: ', corr_attn_score)
        print('Correlation between CMI and pixel change: ', corr_cmi_score)

        corr_attns = []
        corr_cmis = []
        for i in range(len(cmis)):
            corr_attn = np.corrcoef(attentions_pixel[i], pixel_changes[i])[0, 1]
            corr_cmi = np.corrcoef(cmis_pixel[i], pixel_changes[i])[0, 1]
            corr_attns.append(corr_attn)
            corr_cmis.append(corr_cmi)

        print('Pixel level (avg on images):')
        print('Mean correlation between attention and pixel change: ', np.mean(corr_attns))
        print('Mean correlation between CMI and pixel change: ', np.mean(corr_cmis))


        # calculate bootstrapping error bar
        corr_attn_score_list, corr_cmi_score_list = [], []
        corr_attn_list, corr_cmi_list = [], []
        attentions = np.array(attentions)
        pixel_changes_score = np.array(pixel_changes_score)
        cmis = np.array(cmis)

        selected_idx = np.random.randint(len(cmis), size=(100, int(0.9 * len(cmis)))) # repeat 100 times
        for idx in selected_idx:
            # image level
            corr_attn_score_ = np.corrcoef(attentions[idx], pixel_changes_score[idx])[0, 1]
            corr_cmi_score_ = np.corrcoef(cmis[idx], pixel_changes_score[idx])[0, 1]
            corr_attn_score_list.append(corr_attn_score_)
            corr_cmi_score_list.append(corr_cmi_score_)

            # pixel level
            corr_attns = []
            corr_cmis = []
            for i in idx:
                corr_attn = np.corrcoef(attentions_pixel[i], pixel_changes[i])[0, 1]
                corr_cmi = np.corrcoef(cmis_pixel[i], pixel_changes[i])[0, 1]
                corr_attns.append(corr_attn)
                corr_cmis.append(corr_cmi)
            corr_attn_list.append(np.mean(corr_attns))
            corr_cmi_list.append(np.mean(corr_cmis))
        print('\n\n')
        print('Image level bootstrapping error:')
        print('Correlation between attention and pixel change: ', np.std(corr_attn_score_list) / 10)
        print('Correlation between CMI and pixel change: ', np.std(corr_cmi_score_list) / 10)

        print('Pixel level (avg on images) bootstrapping error:')
        print('Mean correlation between attention and pixel change: ', np.std(corr_attn_list) / 10)
        print('Mean correlation between CMI and pixel change: ', np.std(corr_cmi_list) / 10)

if __name__ == "__main__":
    main()
