import argparse

def parse_args_and_update_config():
    parser = argparse.ArgumentParser()

    # In-N-Out Path
    parser.add_argument('--res_out_dir', type=str, default='./results/daam_itd', help="The output directory.")
    parser.add_argument('--data_in_dir', type=str, default='./datasets/coco', help="Please refer to the specific dataset directory.")

    # Model
    parser.add_argument('--sdm_version', type=str, default='sdm_2_1_base', help="Select among: 'sdm_2_0_base' | 'sdm_2_1_base'.")
    parser.add_argument('--batch_size', type=int, default=1, help='The DAAM only accepts SINGLE batch size.')
    parser.add_argument('--infer_step', type=int, default=50, help="Diffusion steps.")
    parser.add_argument('--guide_scale', type=int, default=1, help="Classifier-free guidance, fix it as 1.")
    parser.add_argument('--logsnr_loc', type=float, default=1., help='The logsnr location.')
    parser.add_argument('--logsnr_scale', type=float, default=2., help='The logsnr scale.')
    parser.add_argument('--clip', type=float, default=3., help='The logsnr clip.')

    # Dataset
    parser.add_argument('--csv_name', type=str, default="COCO100-IT", help="Select among: 'COCO100-IT' | 'COCO-IT' | 'COCO-WL'.")

    # Random seed
    parser.add_argument('--seed', type=int, default=42, help="101010")

    args = parser.parse_args()
    return args