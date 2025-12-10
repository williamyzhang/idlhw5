import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision.utils  import make_grid
from torchvision import datasets

from models import UNet, VAE, ClassEmbedder, DiT
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # device    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # setup model
    logger.info("Creating model")
    if args.latent_ddpm:
        latent_img = args.unet_in_size // 8
        model = DiT(
            img_size=latent_img,
            patch_size=getattr(args, "dit_patch_size", 2),
            in_channels=4,
            hidden_size=getattr(args, "dit_hidden_size", 768),
            depth=getattr(args, "dit_depth", 12),
            num_heads=getattr(args, "dit_num_heads", 12),
            mlp_ratio=getattr(args, "dit_mlp_ratio", 4.0),
            num_classes=args.num_classes,
            class_dropout_prob=0.1,
            learn_sigma=True,
        )
    else:
        model = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
     )
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg and not args.latent_ddpm:
        class_embedder = ClassEmbedder(embed_dim=args.unet_ch, n_classes=args.num_classes, cond_drop_rate=0.1)
        
    # send to device
    model = model.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler
    if args.use_ddim:
        scheduler_class = DDIMScheduler
    else:
        scheduler_class = DDPMScheduler
    # TODO: scheduler
    # scheduler = scheduler_class(None)

    # load checkpoint
    print("Loading checkpoint from:", args.ckpt)
    print("checkpoint inference steps:", args.num_inference_steps)
    load_checkpoint(model, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline = DDPMPipeline(
        model=model,
        scheduler=scheduler,
        vae=vae,
        class_embedder=class_embedder,
    )

    
    logger.info("***** Running Inference *****")
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = None 
            all_images.append(gen_images)
    else:
        # generate 5000 images
        for _ in tqdm(range(0, 5000, args.batch_size)):
            gen_images = pipeline(
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device, 
            )
            all_images.append(gen_images)
    
    # TODO: load validation images as reference batch
    if args.use_cifar10:
        val_dataset = datasets.CIFAR10(root='./', train=False, download=False)
    else:
        val_dataset = datasets.ImageFolder(root=args.data_dir)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_images = []
    for batch in val_loader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        val_images.append(images)
        if len(val_images) * args.batch_size >= 5000:
            break
    val_images = torch.cat(val_images, dim=0)[:5000]
    val_images = (val_images + 1) / 2  # rescale to [0, 1]

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics 
    
    from torchmetrics.image.fid import FrechetInceptionDistance, InceptionScore
    
    # TODO: compute FID and IS
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore().to(device)

    for images in tqdm(all_images):
        fid_metric.update(images, real=False)
        is_metric.update(images)

    fid = fid_metric.compute()
    is_score = is_metric.compute()

    logger.info(f"FID: {fid:.2f}")
    logger.info(f"IS: {is_score:.2f}")

if __name__ == '__main__':
    main()
