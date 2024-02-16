import argparse


def parse_args_and_update_config():
    parser = argparse.ArgumentParser()

    # In-N-Out Path
    parser.add_argument(
        "--res_out_dir",
        type=str,
        default="./results/retrieval",
        help="The output directory.",
    )
    parser.add_argument(
        "--data_in_dir",
        type=str,
        default="./datasets/aro",
        help="Please refer to the specific dataset directory.",
    )

    # Model
    parser.add_argument(
        "--sdm_version",
        type=str,
        default="sdm_2_1_base",
        help="Select among: 'sdm_2_0_base' | 'sdm_2_1_base'.",
    )
    parser.add_argument(
        "--n_samples_per_point",
        type=int,
        default=100,
        help="If 'res_type == nll_XD', set it as 1; elif 'res_type == mse_XD'set it as 50.",
    )
    # on A6000, we recommend batch_size=10 for VG-R and VG-A (relation & attribution), and
    # batch_size=5 for coco & flickr
    parser.add_argument("--batch_size", type=int, default=10, help="The batch size.")
    parser.add_argument(
        "--logsnr_loc", type=float, default=1.0, help="The logsnr location."
    )
    parser.add_argument(
        "--logsnr_scale", type=float, default=2.0, help="The logsnr scale."
    )
    parser.add_argument("--clip", type=float, default=3.0, help="The logsnr clip.")
    parser.add_argument(
        "--int_mode",
        type=str,
        default="logistic",
        help="Select between 'uniform' | 'logistic'.",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="relation",
        help="Select among: 'relation' | 'attribution' | 'coco' | 'flickr'.",
        choices=["relation", "attribution", "coco", "flickr"],
    )

    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="101010")

    args = parser.parse_args()
    return args
