"""
Main information theoretic diffusion model class.
"""

import math
import torch as t
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Internal imports
from .utils import logistic_integrate, trunc_uniform_integrate, one_step_test


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        denoiser,
        x_shape=(2,),
        learning_rate=0.001,
        logsnr_loc=0.0,
        logsnr_scale=2.0,
        clip=3.0,
        logsnr2t=None,
        **kwargs,
    ):  # Log SNR importance sampling distribution parameters
        super().__init__()
        self.save_hyperparameters(
            ignore=["denoiser"]
        )  # saves full argument dict into self.hparams
        self.model = denoiser  # First argument of "model" is data, second is log SNR (per sample)
        self.d = np.prod(x_shape)  # Total dimensionality
        self.h_g = (
            0.5 * self.d * math.log(2 * math.pi * math.e)
        )  # Differential entropy for N(0,I)
        self.h_g_scalar = 0.5 * math.log(
            2 * math.pi * math.e
        )  # Differential entropy for N(0,1), scalar
        self.left = (-1,) + (1,) * (
            len(x_shape)
        )  # View for left multiplying a batch of samples
        self.automatic_optimization = False  # Pytorch Lightning flag
        self.logsnr2t = logsnr2t
        self.encode_latents = kwargs.get("encode_latents", None)
        self.encode_prompts = kwargs.get("encode_prompts", None)

    def forward(self, x, t, cond=None):
        if cond is None:
            assert "Please give conditioning."
        else:
            return self.model(x, t, encoder_hidden_states=cond).sample

    def training_step(self, batch, batch_idx):
        return

    def validation_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        """Pytorch Lightning optimizer hook."""
        optimizer = t.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def noisy_channel(self, x, logsnr):
        """Add Gaussian noise to x, return "z" and epsilon."""
        logsnr = logsnr.view(self.left)  # View for left multiplying
        eps = t.randn((len(logsnr),) + self.hparams.x_shape).to(x.device)
        return t.sqrt(t.sigmoid(logsnr)) * x + t.sqrt(t.sigmoid(-logsnr)) * eps, eps

    def mse(self, x, logsnr, text_embs, use_re=False, mse_mode="epsilon", mse_dim=1):
        """
        Localized errors for recovering epsilon from noisy channel, for given log SNR values.
        We'd like to use the same noise for various text embeddings.
        """

        def calc_err(a, b):
            if use_re:  # relative error
                err = t.abs((a - b) / b)
                return t.mean(err, dim=1)
            else:  # mean square error
                err = a - b
                if mse_dim == 1:
                    err = err.flatten(start_dim=1)
                    return t.einsum("ij,ij->i", err, err)
                else:
                    return t.einsum("ichw,ichw->ihw", err, err)  # MSE per pixel

        bs = len(x)
        cond_num = len(text_embs) // bs
        z, eps = self.noisy_channel(x, logsnr)
        # expend z and logsnr
        z = z.repeat(
            (cond_num,) + (1,) * (len(z.shape) - 1)
        )  # cond_num*bs x 4 x 64 x 64
        logsnr = logsnr.repeat(cond_num)
        eps_hat = self(z, self.logsnr2t(logsnr), cond=text_embs)

        if mse_mode == "x":
            logsnr = logsnr.view(self.left)
            x = x.repeat((cond_num,) + (1,) * (len(x.shape) - 1))
            x_hat = t.sqrt(1 + t.exp(-logsnr)) * z - eps_hat * t.exp(-logsnr / 2)
            mse = calc_err(x_hat, x)
            mses_diff_appx = t.cat(
                [
                    calc_err(x_hat[:bs], x_hat[bs:-bs]),  # mse(C-W)
                    calc_err(x_hat[:bs], x_hat[-bs:]),  # mse(C-N)
                    calc_err(x_hat[bs:-bs], x_hat[-bs:]),
                ]
            )  # mse(W-N)
        elif mse_mode == "epsilon":
            eps = eps.repeat((cond_num,) + (1,) * (len(eps.shape) - 1))
            mse = calc_err(eps_hat, eps)  # cond_num*bs x 4 x 64 x 64
            mses_diff_appx = []

            for wrong_i in range(cond_num - 2):
                mses_diff_appx.append(
                    calc_err(
                        eps_hat[wrong_i * bs : (wrong_i + 1) * bs],
                        eps_hat[(cond_num - 2) * bs : (cond_num - 1) * bs],
                    )
                )
            # compute mse {wrong - null}
            for wrong_i in range(cond_num - 2):
                mses_diff_appx.append(
                    calc_err(
                        eps_hat[wrong_i * bs : (wrong_i + 1) * bs],
                        eps_hat[(cond_num - 1) * bs : cond_num * bs],
                    )
                )
            # compute mse {correct - null}
            mses_diff_appx.append(
                calc_err(
                    eps_hat[(cond_num - 2) * bs : (cond_num - 1) * bs],
                    eps_hat[(cond_num - 1) * bs : cond_num * bs],
                )
            )
            mses_diff_appx = t.cat(mses_diff_appx)
        return mse.detach().cpu(), mses_diff_appx.detach().cpu()

    def nll(self, x, text_embs, mse_dim=1, int_mode="uniform"):
        """
        Contribution to -log p(x) from each pixel, for a single image, x. Point-wise MI or KL. (enable for batch)
        """
        if int_mode == "uniform":
            logsnrs, weights = trunc_uniform_integrate(
                (len(x)),
                self.hparams.logsnr_loc,
                self.hparams.logsnr_scale,
                clip=3.0,
                device=x.device,
            )  # bs*bs_
        elif int_mode == "logistic":
            logsnrs, weights = logistic_integrate(
                len(x),
                self.hparams.logsnr_loc,
                self.hparams.logsnr_scale,
                device=x.device,
            )  # bs*bs_
        mses, mses_diff_appx = self.mse(
            x, logsnrs, text_embs, use_re=False, mse_mode="epsilon", mse_dim=mse_dim
        )  # [3*bs*bs_ x h x w, 2*bs*bs_ x h x w, 3*bs*bs_ x h x w]
        logsnrs = logsnrs.repeat(len(mses) // len(x)).cpu()  # 3*bs*bs_

        # P.S.: here, mse ~= mmse, becasue we just sample one z for one logsnr.
        if mse_dim == 1:
            weights_ = weights.repeat((len(mses)) // len(x)).cpu()  # 3*bs*bs_
            weights_appx = weights.repeat(
                (len(mses_diff_appx)) // len(x)
            ).cpu()  # 3*bs*bs_
            mmse_gap = mses - self.d * t.sigmoid(logsnrs)
            nll = self.h_g + 0.5 * (weights_ * mmse_gap)
            mi_appx = 0.5 * (weights_appx * mses_diff_appx)
        else:
            weights = weights.repeat((len(mses)) // len(x)).cpu()  # 3*bs*bs_
            mses_gap = mses - t.sigmoid(logsnrs).view((-1, 1, 1))  # 3*bs*bs_ x h x w
            nll = self.h_g_scalar + 0.5 * (
                weights.view((-1, 1, 1)) * mses_gap
            )  # 3*bs*bs_ x h x w
            mi_appx = 0.5 * (
                weights.view((-1, 1, 1)) * mses_diff_appx
            )  # 3*bs*bs_ x h x w
        return nll, mi_appx

    def image_level_mmse(self, latent_images, conds, logsnrs, bs_=5, total=50):
        """
        Calculate the image-level mmses based on different conds. It gives two outputs with 6 attributes:
        [mmse(eps - eps_hat_c), mmse(eps - eps_hat_w), mmse(eps - eps_hat_n),
        mmse_diff_ap(eps_hat_c - eps_hat_w), mmse_diff_ap(eps_hat_c - eps_hat_n), mmse_diff_ap(eps_hat_w - eps_hat_n)]
        """
        mmse_list = []
        mmse_diff_appx_list = []
        bs = len(latent_images)
        latent_images = latent_images.repeat(
            (bs_,) + (1,) * (len(latent_images.shape) - 1)
        )  # bs*bs_ x c x h x w
        conds = conds.repeat_interleave(bs_, dim=0)  # bs_*cond_num x bs x h x w
        conds = conds.view((-1,) + conds.shape[2:])  # cond_num*bs_*bs x h x w

        for logsnr in tqdm(logsnrs):
            logsnr = logsnr.repeat(bs * bs_)  # bs*bs_
            mse_list = []
            mse_diff_appx_list = []
            for _ in range(total // bs_):  # inner batch scheme for z
                mse, mse_diff_appx = self.mse(
                    latent_images, logsnr, conds, mse_dim=1
                )  # 3*bs_*bs
                mse = mse.reshape(-1, bs_, bs)  # 3 x bs_ x bs
                mse_diff_appx = mse_diff_appx.reshape(-1, bs_, bs)  # 3 x bs_ x bs
                mse_list.append(mse.detach().cpu())
                mse_diff_appx_list.append(mse_diff_appx.detach().cpu())
            mses = t.cat(mse_list, dim=1)  # 3 x total x bs
            mses_diff_appx = t.cat(mse_diff_appx_list, dim=1)  # 3 x total x bs
            mmse = t.mean(mses, dim=1)  # 3 * bs
            mmse_diff_appx = t.mean(mses_diff_appx, dim=1)  # 3 * bs
            mmse_list.append(mmse)
            mmse_diff_appx_list.append(mmse_diff_appx)
        mmses = t.stack(mmse_list)  # snr_num * 3 * bs
        mmses_diff_appx = t.stack(mmse_diff_appx_list)  # snr_num * 3 * bs
        return mmses, mmses_diff_appx  # bs * 3 * snr_num

    def pixel_level_mmse(self, latent_images, conds, logsnrs, bs_=5, total=50):
        """
        Calculate pixel-level mmse for every latent pixel at multiple SNRs. It gives two outputs with 6 attributes:
        [mmse(eps - eps_hat_c), mmse(eps - eps_hat_w), mmse(eps - eps_hat_n),
        mmse_diff_ap(eps_hat_c - eps_hat_w), mmse_diff_ap(eps_hat_c - eps_hat_n), mmse_diff_ap(eps_hat_w - eps_hat_n)]
        """
        pixel_mmse_list = []
        pixel_mmse_diff_appx_list = []
        bs = len(latent_images)
        latent_images = latent_images.repeat(
            (bs_,) + (1,) * (len(latent_images.shape) - 1)
        )  # bs*bs_ x c x h x w
        conds = conds.repeat_interleave(bs_, dim=0)  # bs_*cond_num x bs x h x w
        conds = conds.view((-1,) + conds.shape[2:])  # cond_num*bs_*bs x h x w
        for logsnr in tqdm(logsnrs):
            logsnr = logsnr.repeat(bs * bs_)  # bs*bs_
            pixel_mse_list = []
            pixel_mse_diff_appx_list = []
            for _ in range(total // bs_):
                pixel_mse, pixel_mse_diff_appx = self.mse(
                    latent_images, logsnr, conds, mse_dim=2
                )  # 3*bs_*bs x h x w
                pixel_mse = pixel_mse.reshape(
                    (-1, bs_, bs) + pixel_mse.shape[1:]
                )  # 3 x bs_ x bs x h x w
                pixel_mse_diff_appx = pixel_mse_diff_appx.reshape(
                    (-1, bs_, bs) + pixel_mse_diff_appx.shape[1:]
                )  # 3 x bs_ x bs x h x w
                pixel_mse_list.append(pixel_mse.detach().cpu())
                pixel_mse_diff_appx_list.append(pixel_mse_diff_appx.detach().cpu())
            pixel_mses = t.cat(pixel_mse_list, dim=1)  # 3 x total x bs x h x w
            pixel_mses_diff_appx = t.cat(
                pixel_mse_diff_appx_list, dim=1
            )  # 3 x total x bs x h x w
            pixel_mmse = t.mean(pixel_mses, dim=1)  # 3 x bs x h x w
            pixel_mmse_diff_appx = t.mean(pixel_mses_diff_appx, dim=1)  # 3 x bs x h x w
            pixel_mmse_list.append(pixel_mmse)
            pixel_mmse_diff_appx_list.append(pixel_mmse_diff_appx)
        pixel_mmses = t.stack(pixel_mmse_list)  # snr_num * 3 * bs * h * w
        pixel_mmses_diff_appx = t.stack(
            pixel_mmse_diff_appx_list
        )  # snr_num * 3 * bs * h * w
        return pixel_mmses, pixel_mmses_diff_appx

    def image_level_nll(
        self, latent_images, conds, snr_num, bs_=5, z_sample_num=1, int_mode="logistic"
    ):
        """
        Calculate nll (averaged on channels) for every latent pixel. It gives two outputs with these attributes:
            - nll:     [
                        {nll(eps - eps_hat_c)},       # nll between epsilon and the correct condition
                        {nll(eps - eps_hat_w), ...},  # nlls between epsilon and all wrong conditions
                        {nll(eps - eps_hat_n)},       # nll between epsilon and the null condition
            ]
            - mi_appx: [
                        {miap(eps_hat_c - eps_hat_w), ...}, # mi-approxs between the correct condition and all wrong conditions
                        {miap(eps_hat_c - eps_hat_n)},      # mi-approx  between the correct condition and the null condition
                        {miap(eps_hat_w - eps_hat_n), ...}, # mi-approxs between all wrong conditions and the null condition
            ]
        For example, assuming there're 1 correct condition and 4 wrong conditions (as is the case for COCO and Fickr30K),
        nll will have 6=1+4+1 columns, and mi_appx will have 9=4+1+4 columns.
        """
        # sample SNRs
        if int_mode == "uniform":
            logsnrs, weights = trunc_uniform_integrate(
                snr_num,
                self.hparams.logsnr_loc,
                self.hparams.logsnr_scale,
                clip=3.0,
                device=latent_images.device,
            )  # bs*bs_
        elif int_mode == "logistic":
            logsnrs, weights = logistic_integrate(
                snr_num,
                self.hparams.logsnr_loc,
                self.hparams.logsnr_scale,
                device=latent_images.device,
            )  # bs*bs_
        elif int_mode == "one-step":
            logsnrs, weights = one_step_test(
                self.hparams.logsnr_loc,
                self.hparams.logsnr_scale,
                clip=3.0,
                device=latent_images.device,
            )
        else:
            assert "Please select the correct integration mode: 'uniform', 'logistic', 'one-step'."

        if z_sample_num == 1:
            bs_ = 1
        mmses, mmses_diff_appx = self.image_level_mmse(
            latent_images, conds, logsnrs, bs_=bs_, total=z_sample_num
        )

        logsnrs = logsnrs.cpu()
        weights = weights.cpu()
        mmses_gap = mmses - t.sigmoid(logsnrs).view(
            (-1,) + (1,) * (len(mmses.shape) - 1)
        )  # n * 3 * bs
        nll = self.h_g_scalar + 0.5 * (
            weights.view((-1,) + (1,) * (len(mmses.shape) - 1)) * mmses_gap
        ).mean(
            dim=0
        )  # 3 * bs
        mi_appx = 0.5 * (
            weights.view((-1,) + (1,) * (len(mmses_diff_appx.shape) - 1))
            * mmses_diff_appx
        ).mean(
            dim=0
        )  # 3 * bs
        return nll, mi_appx

    def pixel_level_nll(
        self, latent_images, conds, snr_num, bs_=5, z_sample_num=1, int_mode="logistic"
    ):
        """
        Calculate nll (averaged on channels) for every latent pixel. It gives two outputs with 6 attributes:
            - pixel_nll:     [pnll(eps - eps_hat_c),        pnll(eps - eps_hat_w),        pnll(eps - eps_hat_n)       ]
            - pixel_mi_appx: [pmiap(eps_hat_c - eps_hat_w), pmiap(eps_hat_c - eps_hat_n), pmiap(eps_hat_w - eps_hat_n)]
        """
        # sample SNRs
        if int_mode == "uniform":
            logsnrs, weights = trunc_uniform_integrate(
                snr_num,
                self.hparams.logsnr_loc,
                self.hparams.logsnr_scale,
                clip=3.0,
                device=latent_images.device,
            )  # bs*bs_
        elif int_mode == "logistic":
            logsnrs, weights = logistic_integrate(
                snr_num,
                self.hparams.logsnr_loc,
                self.hparams.logsnr_scale,
                device=latent_images.device,
            )  # bs*bs_
        elif int_mode == "one-step":
            logsnrs, weights = one_step_test(
                self.hparams.logsnr_loc,
                self.hparams.logsnr_scale,
                clip=3.0,
                device=latent_images.device,
            )
        else:
            assert "Please select the correct integration mode: 'uniform', 'logistic', 'one-step'."

        if z_sample_num == 1:
            bs_ = 1
        pixel_mmses, pixel_mmses_diff_appx = self.pixel_level_mmse(
            latent_images, conds, logsnrs, bs_=bs_, total=z_sample_num
        )

        logsnrs = logsnrs.cpu()
        weights = weights.cpu()
        mmses_gap = pixel_mmses - t.sigmoid(logsnrs).view(
            (-1,) + (1,) * (len(pixel_mmses.shape) - 1)
        )  # snr_num * 3 * bs * h * w
        pixel_nll = self.h_g_scalar + 0.5 * (
            weights.view((-1,) + (1,) * (len(pixel_mmses.shape) - 1)) * mmses_gap
        ).mean(
            dim=0
        )  # 3 * bs * h * w
        pixel_mi_appx = 0.5 * (
            weights.view((-1,) + (1,) * (len(pixel_mmses_diff_appx.shape) - 1))
            * pixel_mmses_diff_appx
        ).mean(
            dim=0
        )  # 3 * bs * h * w
        return pixel_nll, pixel_mi_appx

    def image_level_nll_fast(
        self, latent_images, conds, bs_=5, total=100, int_mode="logistic"
    ):
        """
        Calculate nll (averaged on channels) for every latent pixel. It gives two outputs with these attributes:
            - nll:     [
                        {nll(eps - eps_hat_c)},       # nll between epsilon and the correct condition
                        {nll(eps - eps_hat_w), ...},  # nlls between epsilon and all wrong conditions
                        {nll(eps - eps_hat_n)},       # nll between epsilon and the null condition
            ]
            - mi_appx: [
                        {miap(eps_hat_c - eps_hat_w), ...}, # mi-approxs between the correct condition and all wrong conditions
                        {miap(eps_hat_c - eps_hat_n)},      # mi-approx  between the correct condition and the null condition
                        {miap(eps_hat_w - eps_hat_n), ...}, # mi-approxs between all wrong conditions and the null condition
            ]
        For example, assuming there're 1 correct condition and 4 wrong conditions (as is the case for COCO and Fickr30K),
        nll will have 6=1+4+1 columns, and mi_appx will have 9=4+1+4 columns.
        """
        bs = len(latent_images)
        latent_images = latent_images.repeat(
            (bs_,) + (1,) * (len(latent_images.shape) - 1)
        )
        conds = conds.repeat_interleave(bs_, dim=0)
        conds = conds.view((-1,) + conds.shape[2:])

        weighted_mmse_list = []
        weighted_mmse_diff_appx_list = []
        if total < 5:
            bs_ = 1
        for _ in tqdm(range(total // bs_)):  # inner batch scheme for logsnrs
            weighted_mmse, weighted_mmse_diff_appx = self.nll(
                latent_images, conds, mse_dim=1, int_mode=int_mode
            )
            weighted_mmse = weighted_mmse.reshape(
                (-1, bs_, bs) + weighted_mmse.shape[1:]
            )  # 3 x bs_ x bs
            weighted_mmse_diff_appx = weighted_mmse_diff_appx.reshape(
                (-1, bs_, bs) + weighted_mmse_diff_appx.shape[1:]
            )  # 3 x bs_ x bs
            weighted_mmse_list.append(weighted_mmse.detach().cpu())
            weighted_mmse_diff_appx_list.append(weighted_mmse_diff_appx)

        weighted_mmses = t.cat(weighted_mmse_list, dim=1)  # 3 x total x bs
        weighted_mmses_diff_appx = t.cat(
            weighted_mmse_diff_appx_list, dim=1
        )  # 3 x total x bs
        nll = t.mean(weighted_mmses, dim=1)  # 3 x bs
        mi_appx = t.mean(weighted_mmses_diff_appx, dim=1)  # 3 x bs
        return nll, mi_appx  # bs x 3

    def pixel_level_nll_fast(
        self, latent_images, conds, bs_=5, total=100, int_mode="logistic"
    ):
        """
        Calculate nll (averaged on channels) for every latent pixel. It gives two outputs with 6 attributes:
        [nll(eps - eps_hat_c), nll(eps - eps_hat_w), nll(eps - eps_hat_n),
        miap(eps_hat_c - eps_hat_w), miap(eps_hat_c - eps_hat_n), miap(eps_hat_w - eps_hat_n)]
        For fast computation, z_sample_num = 1.
        """

        bs = len(latent_images)
        latent_images = latent_images.repeat(
            (bs_,) + (1,) * (len(latent_images.shape) - 1)
        )  # bs*bs_ x c x h x w
        conds = conds.repeat_interleave(bs_, dim=0)  # bs_*cond_num x bs x h x w
        conds = conds.view((-1,) + conds.shape[2:])  # cond_num*bs_*bs x h x w

        pixel_weighted_mmse_list = []
        pixel_weighted_mmse_diff_appx_list = []
        if total < 5:
            bs_ = 1
        for _ in tqdm(range(max(1, total // bs_))):  # inner batch scheme for logsnrs
            pixel_weighted_mmse, pixel_weighted_mmse_diff_appx = self.nll(
                latent_images, conds, mse_dim=2, int_mode=int_mode
            )  # [3*bs_*bs x h x w, 3*bs_*bs x h x w]
            pixel_weighted_mmse = pixel_weighted_mmse.reshape(
                (-1, bs_, bs) + pixel_weighted_mmse.shape[1:]
            )  # 3 x bs_ x bs x h x w
            pixel_weighted_mmse_diff_appx = pixel_weighted_mmse_diff_appx.reshape(
                (-1, bs_, bs) + pixel_weighted_mmse_diff_appx.shape[1:]
            )  # 3 x bs_ x bs x h x w
            pixel_weighted_mmse_list.append(pixel_weighted_mmse.detach().cpu())
            pixel_weighted_mmse_diff_appx_list.append(
                pixel_weighted_mmse_diff_appx.detach().cpu()
            )

        pixel_weighted_mmses = t.cat(
            pixel_weighted_mmse_list, dim=1
        )  # 3 x total x bs x h x w
        pixel_weighted_mmses_diff_appx = t.cat(
            pixel_weighted_mmse_diff_appx_list, dim=1
        )  # 3 x total x bs x h x w
        pixel_nll = t.mean(pixel_weighted_mmses, dim=1)  # 3 x bs x h x w
        pixel_mi_appx = t.mean(pixel_weighted_mmses_diff_appx, dim=1)  # 3 x bs x h x w
        return pixel_nll, pixel_mi_appx

    def ll_ratio(
        self,
        x,
        v1,
        v2,
        n_points=20,
        n_samples_per_point=20,
        batch_size=10,
        dim=(1, 2, 3),
    ):
        """Get log likelihood ratio estimate and approximation for a PIL image between two text prompts
        using n_samples of estimator, evaluating batch_size denoising evaluations per step.
        Output mean and standard error
        Note that we set the manual seed before each call. This should ensure that the same noise is
        used for each calculation (each call with a different image and prompt).
        This is very useful as some noise in the difference of MSEs cancels out when comparing
        nll of same image with different prompts.
        n_points - is number of logsnr values to use for integration
        n_samples_per_point - is number of samples to use for each logsnr value
        batch_size - limits the number of images that can be processed at once (due to memory constraints)
        """
        mses, mses_std, mses_app, mses_app_std = [], [], [], []
        with t.no_grad():
            logsnrs, weights = logistic_integrate(
                n_points,
                self.hparams.logsnr_loc,
                self.hparams.logsnr_scale,
                self.hparams.clip,
                device=x.device,
            )  # use same device as module
            for i, logsnr in enumerate(tqdm(logsnrs)):
                llr, llr_app = [], []
                for _ in range(n_samples_per_point // batch_size):
                    z, eps = self.noisy_channel(x, logsnr.expand(batch_size))
                    eps_hat_1 = self(
                        z, self.logsnr2t(logsnr.expand(batch_size)), cond=v1
                    )
                    eps_hat_2 = self(
                        z, self.logsnr2t(logsnr.expand(batch_size)), cond=v2
                    )
                    mse_1 = (
                        (eps - eps_hat_1).square().sum(dim=dim)
                    )  # MSE of epsilon estimate, per sample
                    mse_2 = (
                        (eps - eps_hat_2).square().sum(dim=dim)
                    )  # MSE of epsilon estimate, per sample
                    mse_del = (
                        (eps_hat_2 - eps_hat_1).square().sum(dim=dim)
                    )  # MSE of epsilon estimate, per sample
                    this_mse = mse_1 - mse_2
                    llr.append(this_mse)
                    llr_app.append(mse_del)
                llr = t.cat(llr, dim=0)
                llr_app = t.cat(llr_app, dim=0)
                mses.append(t.mean(llr, dim=0).cpu())
                mses_std.append(t.std(llr, dim=0).cpu() / np.sqrt(n_samples_per_point))
                mses_app.append(t.mean(llr_app, dim=0).cpu())
                mses_app_std.append(
                    t.std(llr_app, dim=0).cpu() / np.sqrt(n_samples_per_point)
                )
            mses = t.stack(mses, dim=0)
            mses_app = t.stack(mses_app, dim=0)
            mses_std = t.stack(mses_std, dim=0)
            mses_app_std = t.stack(mses_app_std, dim=0)
        ll_ratio_pixel = 0.5 * (
            weights.cpu().view((-1,) + (len(mses.shape) - 1) * (1,)) * mses
        ).mean(dim=0)
        ll_ratio = ll_ratio_pixel.sum()
        ll_ratio_pixel_app = 0.5 * (
            weights.cpu().view((-1,) + (len(mses_app.shape) - 1) * (1,)) * mses_app
        ).mean(dim=0)
        ll_ratio_app = ll_ratio_pixel_app.sum()
        results_dict = {
            "logsnrs": logsnrs.cpu(),
            "weights": weights.cpu(),
            "mses": mses,
            "mses_std": mses_std,
            "ll_ratio_pixel": ll_ratio_pixel,
            "ll_ratio": ll_ratio,
            "mses_app": mses_app,
            "mses_app_std": mses_app_std,
            "ll_ratio_pixel_app": ll_ratio_pixel_app,
            "ll_ratio_app": ll_ratio_app,
        }
        return results_dict

    @t.no_grad()
    def get_retrieval_scores_batched(
        self,
        joint_loader: DataLoader,
        z_sample_num: int,
        int_mode: str = "logistic",
    ):
        mis = []
        for i, batch in tqdm(enumerate(joint_loader)):
            image_embeds = []
            for i_option in batch["image_options"]:
                image_embed = self.encode_latents(i_option)
                image_embeds.append(image_embed)
            image_embeds = t.cat(image_embeds, dim=0)  # bsz, num_channels, h, w

            text_embeds = []
            for t_options in batch[
                "caption_options"
            ]:  # 2-element tuple: permuted text and positive text
                t_options = list(t_options)
                text_embeds.append(self.encode_prompts(t_options))
            t_emp = [""] * len(image_embeds)
            text_embeds.append(self.encode_prompts(t_emp))
            # shape: (num_options, bsz, seq_len, emb_dim)
            text_embeds = t.stack(text_embeds)  # [false*, true, empty]
            _, mi = self.image_level_nll_fast(
                image_embeds,
                text_embeds,
                bs_=5,
                total=z_sample_num,
                int_mode=int_mode,
            )
            mis.append(mi)
            if i == 3:
                break

        mi_scores = np.concatenate(mis, axis=1)
        return np.expand_dims(mi_scores, axis=1).transpose(2, 1, 0)  # N x N_i x N_t
