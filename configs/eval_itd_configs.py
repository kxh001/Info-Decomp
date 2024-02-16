import argparse

def parse_args_and_update_config():
    parser = argparse.ArgumentParser()

    # In-N-Out Path
    parser.add_argument('--res_out_dir', type=str, default='./results/itd', help="The output directory.")
    parser.add_argument('--data_in_dir', type=str, default='./datasets/coco', help="Please refer to the specific dataset directory.")

    # Model
    parser.add_argument('--sdm_version', type=str, default='sdm_2_1_base', help="Select among: 'sdm_2_0_base' | 'sdm_2_1_base'.")
    parser.add_argument('--n_samples_per_point', type=int, default=1, help="If 'res_type == nll_XD', set it as 1; elif 'res_type == mse_XD'set it as 50.")
    parser.add_argument('--num_steps', type=int, default=100, help="If 'res_type == mses_2D', set it as 10; else set it as 100.")
    parser.add_argument('--batch_size', type=int, default=10, help='The batch size.')
    parser.add_argument('--logsnr_loc', type=float, default=1., help='The logsnr location.')
    parser.add_argument('--logsnr_scale', type=float, default=2., help='The logsnr scale.')
    parser.add_argument('--clip', type=float, default=3., help='The logsnr clip.')
    parser.add_argument('--upscale_mode', type=str, default='bilinear', help="Select between: 'bilinear' | 'bicubic'.")
    parser.add_argument('--int_mode', type=str, default='logistic', help="Select between 'uniform' | 'logistic'.")

    # Dataset
    parser.add_argument('--csv_name', type=str, default="COCO100-IT", help="Select among: 'COCO100-IT' | 'COCO-IT' | 'COCO-WL'.")

    # Random seed
    parser.add_argument('--seed', type=int, default=42, help="101010")

    # Others
    parser.add_argument('--res_type', type=str, default='nll_2D', help="Select among: 'nll_2D' | 'nll_1D' | 'mse_2D' | 'mse_1D'.")
    parser.add_argument('--eval_metrics', type=str, default='cmi', help="Select between 'mi' | 'cmi'.")

    args = parser.parse_args()
    return args