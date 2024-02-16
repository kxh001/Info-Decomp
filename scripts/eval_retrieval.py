import os
import json
import numpy as np
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.itdiffusion import DiffusionModel
from utils.stablediffusion import StableDiffuser
from utils.aro_datasets import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order

from configs.eval_retrieval_configs import parse_args_and_update_config


def main():
    config = parse_args_and_update_config()
    t.manual_seed(config.seed)

    # set hyper-parameters
    res_out_dir = config.res_out_dir
    data_in_dir = config.data_in_dir
    z_sample_num = config.n_samples_per_point
    bs = config.batch_size
    logsnr_loc = config.logsnr_loc
    logsnr_scale = config.logsnr_scale
    clip = config.clip
    int_mode = config.int_mode

    sdm_version = config.sdm_version
    dataset = config.dataset

    print(
        f"sdm: {sdm_version} | dataset: {dataset} | integration: {int_mode} "
        f"| z sample #: {z_sample_num}"
    )

    # load diffusion models
    if sdm_version == "sdm_2_0_base":
        sdm = StableDiffuser("stabilityai/stable-diffusion-2-base")
    elif sdm_version == "sdm_2_1_base":
        sdm = StableDiffuser("stabilityai/stable-diffusion-2-1-base")

    latent_shape = (
        sdm.unet.config.out_channels,
        sdm.unet.config.sample_size,
        sdm.unet.config.sample_size,
    )
    itd = DiffusionModel(
        sdm.unet,
        latent_shape,
        logsnr_loc=logsnr_loc,
        logsnr_scale=logsnr_scale,
        clip=clip,
        logsnr2t=sdm.logsnr2t,
        encode_latents=sdm.encode_latents,
        encode_prompts=sdm.encode_prompts,
    )
    itd.model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (512, 512),
                interpolation=transforms.functional.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )

    if dataset == "relation":
        dataset_init = VG_Relation
        data_in_dir = os.path.join(data_in_dir, "vg")
        res_out_dir = os.path.join(res_out_dir, "vg")
        option_size = 2
        true_caption_last = False
    elif dataset == "attribution":
        dataset_init = VG_Attribution
        data_in_dir = os.path.join(data_in_dir, "vg")
        res_out_dir = os.path.join(res_out_dir, "vg")
        option_size = 2
        true_caption_last = False
    elif dataset == "coco":
        dataset_init = COCO_Order
        data_in_dir = os.path.join(data_in_dir, "coco")
        res_out_dir = os.path.join(res_out_dir, "coco")
        option_size = 5
        true_caption_last = True
    elif dataset == "flickr":
        dataset_init = Flickr30k_Order
        data_in_dir = os.path.join(data_in_dir, "flickr")
        res_out_dir = os.path.join(res_out_dir, "flickr")
        option_size = 5
        true_caption_last = True
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not os.path.exists(res_out_dir):
        os.makedirs(res_out_dir)

    dataset = dataset_init(
        image_preprocess=preprocess,
        download=True,
        root_dir=data_in_dir,
        true_caption_last=true_caption_last,
    )
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
    scores = itd.get_retrieval_scores_batched(dataloader, z_sample_num, int_mode)

    # post-processing & evaluate scores
    scores = scores[..., -option_size:]
    records = dataset.evaluate_scores(scores)

    with open(os.path.join(res_out_dir, f"{dataset}_scores.npy"), "wb") as f:
        np.save(f, scores)

    with open(os.path.join(res_out_dir, f"{dataset}_records.json"), "w") as f:
        json.dump(records, f, indent=4)


if __name__ == "__main__":
    main()
