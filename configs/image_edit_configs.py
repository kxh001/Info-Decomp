import argparse

def parse_args_and_update_config():
    parser = argparse.ArgumentParser()

    # In-N-Out Path
    parser.add_argument('--res_out_dir', type=str, default='./results/image_edit', help="The input directory of results.")
    parser.add_argument('--data_in_dir', type=str, default='./datasets/coco/val2017', help="Please refer to the specific dataset directory.")

    # Model
    parser.add_argument('--sdm_version', type=str, default='sdm_2_1_base', help="Select between: 'sdm_2_0_base' | 'sdm_2_1_base'.")
    parser.add_argument('--n_samples_per_point', type=int, default=120, help="")
    parser.add_argument('--batch_size', type=int, default=120, help="The batch size.")
    parser.add_argument('--num_steps', type=int, default=100, help="The diffusion steps.")
    parser.add_argument('--logsnr_loc', type=float, default=1., help='The logsnr location.')
    parser.add_argument('--logsnr_scale', type=float, default=2., help='The logsnr scale.')
    parser.add_argument('--clip', type=float, default=3., help='The logsnr clip.')

    # Random seed
    parser.add_argument('--seed', type=int, default=42, help="101010")

    args = parser.parse_args()
    return args