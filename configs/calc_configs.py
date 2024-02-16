import argparse

def parse_args_and_update_config():
    parser = argparse.ArgumentParser()

    # In-N-Out Path
    parser.add_argument('--res_in_dir', type=str, default='./results/itd', help="The input directory.")
    parser.add_argument('--data_in_dir', type=str, default='./datasets/coco', help="Please refer to the specific dataset directory.")
    parser.add_argument('--output_dir', type=str, default='./figs/itd', help="The output directory.")

    # Model
    parser.add_argument('--sdm_version', type=str, default='sdm_2_1_base', help="Select among: 'sdm_2_0_base' | 'sdm_2_1_base'.")
    parser.add_argument('--n_samples_per_point', type=int, default=1, help="If 'res_type == nll_2D', set it as 1.")
    parser.add_argument('--num_steps', type=int, default=100, help="The steps for attention heatmaps.")
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size.')
    parser.add_argument('--int_mode', type=str, default='logistic', help="Select between 'uniform' | 'logistic'.")

    # Dataset
    parser.add_argument('--csv_name', type=str, default="COCO-IT", help="Only evaluate segmentation on 'COCO-IT'.")

    # Random seed
    parser.add_argument('--seed', type=int, default=42, help="101010")

    # Others
    parser.add_argument('--calc_type', type=str, default='iou', help="Select among: 'iou' | 'iou_baseline' | 'pearson'.")
    parser.add_argument('--res_type', type=str, default='nll_2D', help="Select between: 'nll_2D' | 'attnmaps'.")
    parser.add_argument('--eval_metrics', type=str, default='cmi', help="Select between 'mi' | 'cmi'.")

    args = parser.parse_args()
    return args