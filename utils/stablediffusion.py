from diffusers import StableDiffusionPipeline

import torch as t
import torch.nn as nn

class StableDiffuser():
    '''
    Construct stable diffusion model via diffusers and transformers manually.
    '''
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.sdm_pipe = StableDiffusionPipeline.from_pretrained(model_id) # load pretrained diffusion model card
        self.unet = self.sdm_pipe.unet.to(self.device)
        self.simunet = UNetSimp(self.unet, self.sdm_pipe.scheduler)
        self.channels, self.width, self.height = self.unet.config.out_channels, self.unet.config.sample_size, self.unet.config.sample_size
        self.vae = self.sdm_pipe.vae.to(self.device)
        self.text_encoder = self.sdm_pipe.text_encoder.to(self.device)
        self.tokenizer = self.sdm_pipe.tokenizer
        self.scheduler = self.sdm_pipe.scheduler
        self.logsnr_min, self.logsnr_max = self.t2logsnr().min(), self.t2logsnr().max()

    ### Updated hooks for stable diffusion pipeline with diffusers > 0.19
    def encode_prompts(self, prompts, repeat=1):
        """A string or list of strings that are prompts, return embeddings."""
        return self.sdm_pipe._encode_prompt(prompts, self.device,
                                            num_images_per_prompt=repeat, do_classifier_free_guidance=False)

    def encode_latents(self, images):
        """Encode images to latents using built-in stable diffusion preprocessing.
        Input: PIL Image, Output: pytorch tensor on device."""
        with t.no_grad():
            pt_images = self.sdm_pipe.image_processor.preprocess(images, height=512, width=512)
            return self.sdm_pipe.vae.encode(pt_images.to(self.device)).latent_dist.sample() * self.vae.config.scaling_factor

    def decode_latents(self, latents):
        """Take latent images and output PIL."""
        with t.no_grad():
            out = self.sdm_pipe.vae.decode(latents / self.vae.config.scaling_factor).sample
            out_pil = self.sdm_pipe.image_processor.postprocess(out)
            return out_pil

    def t2logsnr(self):
        """The logsnr values that were trained, according to the scheduler."""
        logsnrs_trained = t.log(self.scheduler.alphas_cumprod) - t.log1p(-self.scheduler.alphas_cumprod) # Logit function is inverse of sigmoid
        return logsnrs_trained

    def logsnr2t(self, logsnr):
        """Directly use alphas_cumprod schedule from scheduler object"""
        logsnrs_trained = self.t2logsnr().to(logsnr.device)
        assert len(logsnr.shape) == 1, "not a 1-d tensor"
        timestep = t.argmin(t.abs(logsnr.view(-1, 1) - logsnrs_trained), dim=1)
        return timestep

class UNetSimp(nn.Module):
    """Simplify UNet output so dataparallel works."""
    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    def forward(self, x, logsnr, cond):
        """IT Diffusion uses logsnr, which has to be converted to "timestep" for DDPM style models."""
        t = self.logsnr2t(logsnr)
        return self.unet(x, t, encoder_hidden_states=cond).sample

    def logsnr2t(self, logsnr):
        """Directly use alphas_cumprod schedule from scheduler object"""
        logsnrs_trained = self.t2logsnrs().to(logsnr.device)
        assert len(logsnr.shape) == 1, "not a 1-d tensor"
        timestep = t.argmin(t.abs(logsnr.view(-1, 1) - logsnrs_trained), dim=1)
        return timestep

    def t2logsnrs(self):
        """The logsnr values that were trained, according to the scheduler."""
        logsnrs_trained = t.log(self.scheduler.alphas_cumprod) - t.log1p(-self.scheduler.alphas_cumprod)
        return logsnrs_trained