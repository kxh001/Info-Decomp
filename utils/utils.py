# Initialization
import os
import re
import torch as t
import numpy as np
import pandas as pd
import seaborn as sns
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import trange
from torch.utils.data import Dataset

from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms

sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.style.use('seaborn-v0_8-paper')  # also try 'seaborn-talk', 'fivethirtyeight'


######################################################
##           Importance Sampling Integrate          ##
######################################################
def logistic_integrate(npoints, loc, scale, clip=4., device='cpu', deterministic=False):
    """Return sample point and weights for integration, using
    a truncated logistic distribution as the base, and importance weights.
    """
    loc, scale, clip = t.tensor(loc, device=device), t.tensor(scale, device=device), t.tensor(clip, device=device)

    # IID samples from uniform, use inverse CDF to transform to tar_prompt distribution
    if deterministic:
        t.manual_seed(0)
    ps = t.rand(npoints, dtype=loc.dtype, device=device)
    ps = t.sigmoid(-clip) + (t.sigmoid(clip) - t.sigmoid(-clip)) * ps  # Scale quantiles to clip
    logsnr = loc + scale * t.logit(ps)  # Using quantile function for logistic distribution

    # importance weights
    weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
    return logsnr, weights

def trunc_uniform_integrate(npoints, loc, scale, clip=4., device='cpu', deterministic=False):
    """Return sample point and weights for integration, using
    a truncated distribution proportional to 1 / (1+snr) as the base, and importance weights.
    loc, scale, clip  - are same as for continuous density estimator, just used to fix the range
    parameter, eps=1 is the form implied by optimal Gaussian MMSE at low SNR.
    True MMSE drops faster, so we use a smaller constant
    """
    loc, scale, clip = t.tensor(loc, device=device), t.tensor(scale, device=device), t.tensor(clip, device=device)
    left_logsnr, right_logsnr = loc - clip * scale, loc + clip * scale  # truncated range

    # IID samples from uniform, use inverse CDF to transform to target distribution
    if deterministic:
        t.manual_seed(0)
    ps = t.rand(npoints, dtype=loc.dtype, device=device)
    logsnrs = left_logsnr + (right_logsnr - left_logsnr) * ps  # Use quantile function

    # importance weights
    weights = t.ones(npoints, device=device) / (right_logsnr - left_logsnr)
    return logsnrs, weights

def one_step_test(loc, scale, clip=4., device='cpu'):
    logsnr = t.tensor([loc + clip * scale], device=device)
    loc, scale, clip = t.tensor(loc, device=device), t.tensor(scale, device=device), t.tensor(clip, device=device)
    weight = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc) / scale) * t.sigmoid(-(logsnr - loc) / scale))
    return logsnr, weight

######################################################
##               Visualization utils                ##
######################################################
def perform_word_swaps(input_string, substitutions):
    # Create a regular expression pattern to match words
    pattern = r'\b(' + '|'.join(re.escape(word) for word in substitutions.keys()) + r')\b'
    # Define a function to replace matched words with their substitutions
    def replace(match):
        return substitutions[match.group(0).lower()]
    # Use re.sub() to apply the substitutions, ignoring case
    transformed_string = re.sub(pattern, replace, input_string, flags=re.IGNORECASE)
    return transformed_string

######################################################
##              Dataloader Constructor              ##
######################################################
class CocoDataset(Dataset):
    def __init__(self, img_dir, annotation_file, csv_file):
        self.img_dir = img_dir
        self.coco = COCO(annotation_file)
        self.csv = pd.read_csv(csv_file)
        # Fill NaN values with an empty string
        self.csv.fillna("", inplace=True)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if t.is_tensor(idx):
            idx = idx.tolist()

        img_id = int(self.csv.iloc[idx, 0])
        cat_nums = self.csv.iloc[idx, 1]
        context_cobj = self.csv.iloc[idx, 2]
        context = self.csv.iloc[idx, 3]
        cobj = self.csv.iloc[idx, 4]
        if self.csv.shape[1] > 5:
            context_ = self.csv.iloc[idx, 5]
        else:
            context_ = [""]

        # Load image
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Load mask
        cat_id = self.coco.getCatIds(catNms=[cat_nums])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        mask = self.coco.annToMask(anns[0])
        for i in range(1, len(anns)):
            mask += self.coco.annToMask(anns[i])
        mask = (mask > 0).astype(np.uint8) * 255
        mask = Image.fromarray(mask)

        img, mask = self._transform((img, mask), 512)
        sample = {'image': img, 'mask': mask, 'caption': context_cobj,
                  'context': context, 'obj': cobj, 'category': cat_nums,
                  'context_': context_}

        return sample

    def _transform(self, sample, input_res):
        img, mask = sample
        img = img.resize((input_res, input_res), Image.BILINEAR)
        mask = mask.resize((input_res, input_res), Image.NEAREST)
        return (transforms.ToTensor()(img), transforms.ToTensor()(mask).long())

def process_mask(hm, threshold, is_soft=False):
    if is_soft:
        condition = (hm < threshold[0]) | (hm > threshold[1])
        result = t.where(condition, hm, t.zeros_like(hm))
    else:
        mask = hm > threshold
        result = mask.to(t.int)
    return result


######################################################
##              Sampling and Inpainting             ##
######################################################
def logsnr2sigma(logsnrs):
    return t.exp(-to_tensor(logsnrs) / 2)

def sigma2logsnr(sigmas):
    return -2 * t.log(to_tensor(sigmas))

@t.no_grad()
def generate(model, step_function, schedule, x=None, prompt='', batch_size=1):
    """Generates a batch of images from a diffusion model and a noise schedule,
    specified in karras style in terms of sigma."""
    if x is None:
        x = t.randn([batch_size, model.channels, model.width, model.height], device=model.device) * schedule[0]
    else:
        batch_size = x.shape[0]
    v = model.encode_prompts(prompt).expand(batch_size, -1, -1)
    model.simunet.eval()
    for i in trange(len(schedule) - 1, desc='Generating images'):
        x = step_function(model.simunet, x, v, schedule[i], schedule[i+1])
    return x

@t.no_grad()
def reverse(model, step_function, schedule, x, prompt=''):
    """Reverse diffusion to get "latent"/Gaussian from normalizing flow for input x."""
    v = model.encode_prompts(prompt).expand(len(x), -1, -1)
    # import IPython; IPython.embed()
    model.simunet.eval()
    for i in trange(len(schedule) - 1, desc='Get latent from image'):
        x = step_function(model.simunet, x, v, schedule[-(i+1)], schedule[-(i+2)])
    return x


@t.no_grad()
def inpaint(model, step_function, schedule, x0, mask):
    """Inpaints images using a diffusion model and a binary mask, where "1" is the masked part to inpaint."""
    model.eval()
    x = t.randn_like(x0) * schedule[0]
    for i in trange(len(schedule) - 1, desc='Inpainting image'):
        sigma = schedule[i]
        noisy_original = x0 + sigma * t.randn_like(x0)
        x = x * mask + noisy_original * (1 - mask)  # project onto original (noisified) for unmasked part
        x = step_function(model, x, schedule[i], schedule[i+1])
    x = x * mask + x0 * (1 - mask)
    return x


def translate_denoiser(model, x, v, sigma_hat):
    """Translates our denoiser to Karras conventions."""
    s_in = x.new_ones([x.shape[0]])
    logsnr = sigma2logsnr(sigma_hat)  # convert sigma to logsnr for our model
    scale = t.sqrt(t.sigmoid(logsnr))  # "scale" of x_t in Karras convention
    x = x * scale  # Scale input to variance preserving convention for our denoiser
    eps_hat = model(x, logsnr * s_in, v)  # Predicts noise
    denoised = x / t.sqrt(t.sigmoid(logsnr)) - t.exp(-logsnr / 2) * eps_hat  # Predicts original x
    return denoised


def get_step(order=1, s_churn=0.):
    """Note that s_churn here is defined as the term that Karras calls s_churn / (len(schedule)-1)."""
    return partial(stochastic_step, order=order, s_churn=s_churn)


@t.no_grad()
def stochastic_step(model, x, v, sigma0, sigma1, order=1, s_churn=0., s_noise=1.):
    """Implements Algorithm *2* (and with s_churn=0, also Alg. 1) from Karras et al. (2022).
    Code assumes Karras conventions, with commented wrappers to use our denoiser"""
    gamma = min(s_churn, 2 ** 0.5 - 1)  # s_churn = 0 turns off noise
    sigma_hat = sigma0 * (gamma + 1)  # Increase the first sigma, by adding noise
    if gamma > 0:
        eps = t.randn_like(x) * s_noise  # Karras use 1.007 for s_noise, but 1. works well/more principled
        x = x + eps * (sigma_hat ** 2 - sigma0 ** 2) ** 0.5
    denoised = translate_denoiser(model, x, v, sigma_hat)  # original had a pre-wrapped model: model(x, sigma_hat * s_in)
    d = to_d(x, sigma_hat, denoised)  # TODO: If we allow reverse steps must change
    dt = sigma1 - sigma_hat
    if order == 1 or sigma1 == 0:
        x = x + d * dt  # Euler method
    elif order == 2:  # Heun's method
        x_2 = x + d * dt
        denoised_2 = translate_denoiser(model, x_2, v, sigma1)
        d_2 = to_d(x_2, sigma1, denoised_2)
        d_prime = (d + d_2) / 2
        x = x + d_prime * dt
    else:
        assert False, "first and second order only supported"
    return x


def append_zero(x):
    return t.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = t.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_tensor(x):
    if isinstance(x, t.Tensor):
        return x
    else:
        return t.tensor(x)
