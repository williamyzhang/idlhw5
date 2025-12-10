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
from torchvision import transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from models import UNet, VAE, ClassEmbedder
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
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
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
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(
            embed_dim=args.unet_ch,
            n_classes=args.num_classes,
            cond_drop_rate=0.1,
        )
        
    # send to device
    unet = unet.to(device)
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
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        class_embedder=class_embedder,
    )

    
    logger.info("***** Running Inference *****")
    pil_to_tensor = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL to tensor in [0, 1]
    ])

    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    sample_images_per_class = []

    save_dir = os.path.join(args.output_dir, 'generated_samples')
    # save_dir = os.path.join('generated_samples/epoch28_ddpm_cfg_cifar10')
    print(f"save dir: {save_dir}")

    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            ) 
            # Save first 3 images as a grid for this class
            sample_images = torch.stack([pil_to_tensor(img) for img in gen_images[:3]])
            sample_grid = make_grid(sample_images, nrow=3)
            sample_grid_img = transforms.ToPILImage()(sample_grid)
            sample_grid_img.save(os.path.join(save_dir, f"class_{i}_grid.png"))

            # Convert PIL images to tensors
            gen_images = torch.stack([pil_to_tensor(img) for img in gen_images]).to(device)
            all_images.append(gen_images)

        # # Create and save grid of samples (3 images per class)
        # os.makedirs("generated_samples", exist_ok=True)
        # sample_grid = make_grid(torch.stack(sample_images_per_class), nrow=3)
        # sample_grid_img = transforms.ToPILImage()(sample_grid)
        # sample_grid_img.save("generated_samples/class_samples_grid.png")
        # wandb_logger.log({'gen_images': wandb.Image(grid_image)})
    else:
        # generate 5000 images
        for _ in tqdm(range(0, 5000, args.batch_size)):
            gen_images = pipeline(
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device, 
            )
            # Convert PIL images to tensors
            gen_images = torch.stack([pil_to_tensor(img) for img in gen_images]).to(device)
            all_images.append(gen_images)
    
    # Concatenate all generated images
    all_images = torch.cat(all_images, dim=0)[:5000]

    # TODO: load validation images as reference batch
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.use_cifar10:
        print("Using CIFAR-10 as validation dataset")
        val_dataset = datasets.CIFAR10(root='./', train=False, download=False, transform=transform)
    else:
        val_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_images = []
    for (batch) in val_loader:
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
    
    # TODO: compute FID and IS
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore().to(device)

    # Convert to uint8 in batches to save memory
    batch_size_metric = args.batch_size  # Smaller batch size for metric computation
    
    # Update FID with real images in batches
    for i in tqdm(range(0, len(val_images), batch_size_metric), desc="Processing real images"):
        batch = val_images[i:i+batch_size_metric]
        batch_uint8 = (batch * 255).clamp(0, 255).to(torch.uint8).to(device)
        fid_metric.update(batch_uint8, real=True)
        del batch_uint8
        torch.cuda.empty_cache()
    
    # Update FID and IS with generated images in batches
    for i in tqdm(range(0, len(all_images), batch_size_metric), desc="Processing generated images"):
        batch = all_images[i:i+batch_size_metric]
        print(f"Batch dtype: {batch.dtype}")
        print(f"Batch range: [{batch.min()}, {batch.max()}]")   
        batch_uint8 = (batch * 255).clamp(0, 255).to(torch.uint8).to(device)
        fid_metric.update(batch_uint8, real=False)
        is_metric.update(batch_uint8)
        del batch_uint8
        torch.cuda.empty_cache()

    # print(f"Image dtype: {images.dtype}")
    # print(f"Image range: [{images.min()}, {images.max()}]")

    # val_images_uint8 = ((val_images + 1) * 127.5).to(torch.uint8)
    # images_uint8 = ((images + 1) * 127.5).to(torch.uint8)


    # for images in tqdm(all_images):
    #     # Update FID with real images first
    #     fid_metric.update(val_images_uint8.to(device), real=True)
    #     fid_metric.update(images_uint8.to(device), real=False)
    #     is_metric.update(images_uint8.to(device))

    fid = fid_metric.compute()
    is_score = is_metric.compute()

    logger.info(f"FID: {fid}")
    logger.info(f"IS (mean, std): {is_score[0]}, {is_score[1]}")

if __name__ == '__main__':
    main()