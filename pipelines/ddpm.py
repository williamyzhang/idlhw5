
from typing import List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm
import torch 
import torch.nn as nn
from utils import randn_tensor



class DDPMPipeline:
    def __init__(self, model, scheduler, vae=None, class_embedder=None):
        # model can be UNet or DiT
        self.model = model
        self.scheduler = scheduler
        
        # NOTE: this is for latent DDPM
        self.vae = None
        if vae is not None:
            self.vae = vae
            
        # NOTE: this is for CFG
        if class_embedder is not None:
            self.class_embedder = class_embedder

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    
    @torch.no_grad()
    def __call__(
        self, 
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale : Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device = None,
    ):
        if hasattr(self.model, "x_embedder"):
            h, w = self.model.x_embedder.img_size
            c = self.model.in_channels
            image_shape = (batch_size, c, h, w)
        else:
            image_shape = (batch_size, self.model.input_ch, self.model.input_size, self.model.input_size)
        if device is None:
            print('Device not specified, using model device')
            device = next(self.model.parameters()).device
        
        if classes is not None:
            # convert classes to tensor
            print('Classes:', classes)
            if isinstance(classes, int):
                classes = [classes] * batch_size
            elif isinstance(classes, list):
                assert len(classes) == batch_size, "Length of classes must be equal to batch_size"
                classes = torch.tensor(classes, device=device)
            else:
                classes = classes.to(device)

            if hasattr(self, "class_embedder"):
                uncond_classes = torch.full((batch_size,), self.class_embedder.num_classes, device=device) 
                class_embeds = self.class_embedder(classes)
                uncond_embeds = self.class_embedder(uncond_classes)

        if guidance_scale is not None:
            print('Guidance scale:', guidance_scale)

        # TODO: starts with random noise
        image = randn_tensor(image_shape, generator=generator, device=device)
        # print('Initial noise image shape:', image.shape)

        # TODO: set step values using set_timesteps of scheduler
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
        
        # TODO: inverse diffusion process with for loop
        for t in self.progress_bar(self.scheduler.timesteps):
            t = t.to(device)
            
            # model prediction with optional CFG
            if guidance_scale is not None:
                if hasattr(self, "class_embedder"):
                    model_input = torch.cat([image, image], dim=0)
                    c = torch.cat([uncond_embeds, class_embeds], dim=0)
                    model_output = self.model(model_input, t, c)
                    uncond_model_output, cond_model_output = model_output.chunk(2)
                    model_output = uncond_model_output + guidance_scale * (cond_model_output - uncond_model_output)
                else:
                    if classes is None:
                        raise ValueError("Provide classes for CFG when no class_embedder is used.")
                    eps_uncond, _ = self.model(image, t, y=None)
                    eps_cond, _ = self.model(image, t, y=classes)
                    model_output = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                out = self.model(image, t, classes if classes is not None and not hasattr(self, "class_embedder") else None)
                if isinstance(out, tuple):
                    out = out[0]
                model_output = out
            
            # TODO: 2. compute previous image: x_t -> x_t-1 using scheduler
            image = self.scheduler.step(
                model_output=model_output, 
                timestep=t, 
                sample=image,
                generator=generator) 
            
        
        # NOTE: this is for latent DDPM
        # TODO: use VQVAE to get final image
        if self.vae is not None:
            # NOTE: remember to rescale your images
            image = image * 2 - 1 
            image = self.vae.decode(image)
            # TODO: clamp your images values
            image = image.clamp(-1, 1)
        
        # TODO: return final image, re-scale to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1) 
        
        # convert to PIL images
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)
        
        return image
        



