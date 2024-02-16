import argparse
def parse_args_and_update_config():
    parser = argparse.ArgumentParser()

    # In-N-Out Path
    parser.add_argument('--fig_out_dir', type=str, default='./figs', help="The output directory of visualization.")
    parser.add_argument('--res_in_dir', type=str, default='./results', help="The input directory of results.")
    parser.add_argument('--data_in_dir', type=str, default='./datasets/coco', help="Please refer to the specific dataset directory.")

    # Model
    parser.add_argument('--sdm_version', type=str, default='sdm_2_1_base', help="Select between: 'sdm_2_0_base' | 'sdm_2_1_base'.")

    # Dataset
    parser.add_argument('--csv_name', type=str, default="COCO100-IT", help="Select between: 'COCO100-IT' | 'COCO-WL'.")

    # Random seed
    parser.add_argument('--seed', type=int, default=42, help="101010")

    # Others
    parser.add_argument('--visual_type', type=str, default='mmse_curve', help="Select between: 'mmse_curve' | 'scatter_plot' | 'denoising_diffusion' | 'mi_cmi_attn' | 'image_edit'.")

    args = parser.parse_args()
    return args

