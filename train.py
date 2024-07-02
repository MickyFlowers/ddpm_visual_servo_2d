import argparse
import datetime
import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from ddpm import script_utils
from torch.utils.tensorboard import SummaryWriter
from ddpm.ddpm_dataset import DDPMDataset
import numpy as np
from loguru import logger


def main():
    args = create_argparser().parse_args()
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_tensorboard:
            if args.log_dir is None:
                raise ValueError(
                    "args.log_to_tensorboard set to True but args.project_name is None"
                )
            writer = SummaryWriter(log_dir=args.log_dir)

        batch_size = args.batch_size
        logger.info("Loading Dataset")
        dataset = DDPMDataset(args.data_dir, transform=script_utils.get_transform())
        dataset_size = len(dataset)
        logger.info(f"dataset size: {dataset_size}")
        dataset_list = np.arange(0, dataset_size)
        train_list = np.random.choice(
            dataset_list, int(dataset_size * 0.95), replace=False
        )
        test_list = np.setdiff1d(dataset_list, train_list)
        train_dataset = torch.utils.data.Subset(dataset, train_list)
        test_dataset = torch.utils.data.Subset(dataset, test_list)
        logger.info(f"train dataset size: {len(train_dataset)}")
        logger.info(f"test dataset size: {len(test_dataset)}")

        train_loader = script_utils.cycle(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=10,
                persistent_workers=True,
            )
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, drop_last=True, num_workers=1
        )

        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            diffusion.train()
            import time

            start_time = time.time()
            tar, ref, seg = next(train_loader)
            end_time = time.time()
            logger.debug(f"Get Batch Time: {end_time - start_time}")
            tar = tar.to(device)
            ref = ref.to(device)
            seg = seg.to(device)

            loss = diffusion(tar, ref, seg)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()

            if iteration % args.log_rate == 0:
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for tar, ref, seg in test_loader:

                        ref = ref.to(device)
                        tar = tar.to(device)
                        seg = seg.to(device)

                        loss = diffusion(tar, ref, seg)
                        test_loss += loss.item()
                for tar, ref, seg in test_loader:
                    ref = ref.to(device)
                    tar = tar.to(device)
                    seg = seg.to(device)
                    break
                samples = diffusion.sample(args.batch_size, device, ref, seg)
                samples = ((samples + 1) / 2).clip(0, 1).numpy()
                gt = ((tar + 1) / 1).clip(0, 1).detach().cpu().numpy()
                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate
                if args.log_to_tensorboard:
                    writer.add_scalar("loss/test_loss", test_loss, iteration)
                    writer.add_scalar("loss/train_loss", acc_train_loss, iteration)
                    writer.add_image(f"sample", samples, iteration, dataformats="NCHW")
                    writer.add_image(f"gt", gt, iteration, dataformats="NCHW")

                acc_train_loss = 0

            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)

        if args.log_to_tensorboard:
            writer.close()
    except KeyboardInterrupt:
        if args.log_to_tensorboard:
            writer.close()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = (
        torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        data_dir="/cyx/data/ddpm_vs/img",
        learning_rate=1e-4,
        batch_size=8,
        iterations=800000,
        log_to_tensorboard=True,
        log_rate=100,
        checkpoint_rate=1000,
        model_save_dir="./models",
        log_dir="./logs",
        project_name=None,
        run_name=run_name,
        model_checkpoint=None,
        optim_checkpoint=None,
        schedule_low=1e-4,
        schedule_high=0.02,
        device=device,
    )
    defaults["log_dir"] = os.path.join(defaults["log_dir"], defaults["run_name"])
    defaults["model_save_dir"] = os.path.join(
        defaults["model_save_dir"], defaults["run_name"]
    )
    if not os.path.exists(defaults["log_dir"]):
        os.makedirs(defaults["log_dir"])
    if not os.path.exists(defaults["model_save_dir"]):
        os.makedirs(defaults["model_save_dir"])

    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
