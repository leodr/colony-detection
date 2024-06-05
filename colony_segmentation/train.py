# Training Faster R-CNN
# ---
#
# The code in here is originally based on this tutorial
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# %%
from typing import Optional

# Fixed hyperparameters
BATCH_SIZE = 8


class ExperimentParameters:
    def __init__(
        self,
        learning_rate: float,
        use_coco_pretrained: bool,
        use_agar_pretraining: bool,
        patience_agar: int,
        patience_photobox: int,
        use_rotation_augmentation: bool,
        use_patch_augmentation: bool,
        max_norm: Optional[float],
    ):
        self.learning_rate = learning_rate
        self.use_coco_pretrained = use_coco_pretrained
        self.use_agar_pretraining = use_agar_pretraining
        self.patience_agar = patience_agar
        self.patience_photobox = patience_photobox
        self.use_rotation_augmentation = use_rotation_augmentation
        self.use_patch_augmentation = use_patch_augmentation
        self.max_norm = max_norm


import itertools

# %%
from pathlib import Path
from typing import Optional

import torch
from engine import evaluate, train_one_epoch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def average_losses(loss_dicts: list[dict[str, torch.Tensor]]):
    loss_cls = torch.stack([loss["loss_classifier"] for loss in loss_dicts]).mean()
    loss_box_reg = torch.stack([loss["loss_box_reg"] for loss in loss_dicts]).mean()
    loss_objectness = torch.stack(
        [loss["loss_objectness"] for loss in loss_dicts]
    ).mean()
    loss_rpn_box_reg = torch.stack(
        [loss["loss_rpn_box_reg"] for loss in loss_dicts]
    ).mean()

    summed_loss = loss_cls + loss_box_reg + loss_objectness + loss_rpn_box_reg

    return {
        "classifier": loss_cls,
        "box_reg": loss_box_reg,
        "objectness": loss_objectness,
        "rpn_box_reg": loss_rpn_box_reg,
        "total": summed_loss,
    }


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        gpu_id: int,
        writer: SummaryWriter,
        model_save_path: str,
        exp_params: ExperimentParameters,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer

        self.should_val = val_data is not None

        self.model_save_path = model_save_path
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

        self.exp_params = exp_params

        self.cancel_run = False

    def _run_epoch(self, epoch, scaler, save_hparams=False):
        # train for one epoch, printing every 10 iterations
        train_losses = train_one_epoch(
            self.model,
            self.optimizer,
            self.train_data,
            self.gpu_id,
            epoch,
            scaler=scaler,
            print_freq=10,
            max_norm=self.exp_params.max_norm,
        )

        avg_train_loss_dict = average_losses(train_losses)
        self.writer.add_scalars("train_loss", avg_train_loss_dict, epoch)

        # check loss on validation set
        if self.should_val:
            with torch.no_grad():
                val_losses = train_one_epoch(
                    self.model,
                    self.optimizer,
                    self.val_data,
                    self.gpu_id,
                    epoch,
                    print_freq=10,
                    optimize=False,
                )

                avg_val_loss_dict = average_losses(val_losses)
                self.writer.add_scalars("val_loss", avg_val_loss_dict, epoch)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(avg_val_loss_dict["total"])

                    last_lr = self.lr_scheduler.get_last_lr()[0]
                    # Let the learning rate decay two times (10%, 1%), then stop.
                    min_lr = self.exp_params.learning_rate * 0.005

                    self.writer.add_scalar("lr", last_lr, epoch)

                    print(f"Last LR: {last_lr} | Min LR: {min_lr}")

                    if last_lr < min_lr and epoch > 10:
                        self.cancel_run = True

                # evaluate on the validation dataset
                coco_evaluator, mae, smape = evaluate(
                    self.model, self.val_data, device=self.gpu_id
                )

                stats = coco_evaluator.coco_eval["bbox"].stats

                avg_precision_iou_50 = stats[0]
                avg_recall_iou_50 = stats[8]

                val_metrics = {
                    "avg_precision_iou_50": float(avg_precision_iou_50),
                    "avg_recall_iou_50": float(avg_recall_iou_50),
                    "mae": float(mae),
                    "smape": float(smape),
                }

                self.writer.add_scalars("val_metrics", val_metrics, epoch)

                if self.cancel_run:
                    hparams_dict = self.exp_params.__dict__
                    hparams_dict = {"hparam/" + k: v for k, v in hparams_dict.items()}

                    self.writer.add_hparams(hparams_dict, val_metrics)

        self.writer.flush()

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        torch.save(ckp, self.model_save_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {self.model_save_path}")

    def train(self):
        scaler = torch.cuda.amp.GradScaler()

        for epoch in itertools.count():
            self._run_epoch(epoch, scaler)

            self.writer.add_scalar(
                "free_gpu_memory", torch.cuda.mem_get_info(self.gpu_id)[0], epoch
            )

            if epoch % 10 == 0:
                self._save_checkpoint(epoch)

            if self.cancel_run:
                break

        self._save_checkpoint(-1)


# %% Data helpers
import json

import utils
from dataset_agar import AgarBBDataset
from torch.utils.data import Dataset

# The box data for these IDs of the AGAR dataset have negative x values for the bounding boxes.
faulty_ids = ["5794", "11988"]


def load_id_list(path: str):
    with open(path) as f:
        list = json.load(f)
        list = [int(id) for id in list if id.isdigit() and id not in faulty_ids]

    return list


def prepare_agar_datasets():
    train_list = load_id_list(
        "../AGAR_dataset/training_lists/higher_resolution_train.txt"
    )
    val_list = load_id_list("../AGAR_dataset/training_lists/higher_resolution_val.txt")

    print(f"Training on {len(train_list)} images.")
    print(f"Validating on {len(val_list)} images.")

    train_dataset = AgarBBDataset("../AGAR_dataset/dataset", train_list, is_train=True)
    val_dataset = AgarBBDataset("../AGAR_dataset/dataset", val_list, is_train=False)

    return train_dataset, val_dataset


def prepare_dataloader(dataset: Dataset, is_train: bool):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=is_train,
        collate_fn=utils.collate_fn,
    )


import json

# %% Discord notification
import requests


def send_discord_notification(message: str, embed_data: dict):
    url = "REDACTED_DISCORD_WEBHOOK_URL"

    data = {"content": message, "username": "Notification Bot"}

    if data is not None:
        data["embeds"] = [
            {
                "title": "Data",
                "description": json.dumps(embed_data, indent=4),
            }
        ]

    requests.post(url, json=data)


# %% Main function
import os
from datetime import datetime

from dataset_photobox import PhotoboxBBDataset
from model import get_model_instance_segmentation
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main(device: str, exp_params: ExperimentParameters):
    start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

    if exp_params.use_agar_pretraining:
        print("Training on AGAR dataset\n" + "-" * 10 + "\n")

        agar_train_dataset, agar_val_dataset = prepare_agar_datasets()

        agar_train_loader = prepare_dataloader(agar_train_dataset, is_train=True)
        agar_val_loader = prepare_dataloader(agar_val_dataset, is_train=False)

        agar_model = get_model_instance_segmentation(
            AgarBBDataset.num_classes,
            pretrained=exp_params.use_coco_pretrained,
        )

        params = [p for p in agar_model.parameters() if p.requires_grad]
        optimizer = SGD(
            params, lr=exp_params.learning_rate, weight_decay=0.0005, momentum=0.9
        )
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=exp_params.patience_agar)

        writer = SummaryWriter(f"runs/{start_time_str}/agar")
        agar_model_path = f"artifacts/{start_time_str}/agar.pth"

        trainer = Trainer(
            agar_model,
            agar_train_loader,
            agar_val_loader,
            optimizer,
            lr_scheduler,
            device,
            writer,
            agar_model_path,
            exp_params,
        )

        trainer.train()

    print("Training on Photobox dataset\n" + "-" * 10 + "\n")

    photobox_train_dataset = PhotoboxBBDataset(
        "../train_data",
        is_train=True,
        rotate_augmentation=exp_params.use_rotation_augmentation,
        patch_augmentation=exp_params.use_patch_augmentation,
    )

    photobox_val_dataset = PhotoboxBBDataset("../val_data", is_train=False)

    photobox_model = get_model_instance_segmentation(
        PhotoboxBBDataset.num_classes,
        pretrained=(
            exp_params.use_coco_pretrained
            if not exp_params.use_agar_pretraining
            else False
        ),
    )
    if exp_params.use_agar_pretraining:
        photobox_model.load_state_dict(torch.load(agar_model_path))

    photobox_train_loader = prepare_dataloader(photobox_train_dataset, is_train=True)
    photobox_val_loader = prepare_dataloader(photobox_val_dataset, is_train=False)

    params = [p for p in photobox_model.parameters() if p.requires_grad]
    optimizer = SGD(
        params, lr=exp_params.learning_rate, weight_decay=0.0005, momentum=0.9
    )
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=exp_params.patience_photobox)

    writer = SummaryWriter(f"runs/{start_time_str}/final")

    trainer = Trainer(
        photobox_model,
        photobox_train_loader,
        photobox_val_loader,
        optimizer,
        lr_scheduler,
        device,
        writer,
        f"artifacts/{start_time_str}/final.pth",
        exp_params,
    )

    with open(f"artifacts/{start_time_str}/params.json", "w") as f:
        json.dump(exp_params.__dict__, f, indent=4)

    trainer.train()

    send_discord_notification(
        f"{os.uname().nodename} {device} finished training.",
        {
            "start_time": start_time_str,
            "params": exp_params.__dict__,
        },
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    # fmt: off
    parser = ArgumentParser(description="Experiment Parameters")
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to run the experiment on')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the experiment')
    parser.add_argument('--use_coco_pretrained', type=int, default=False, help='Use COCO pretrained model')
    parser.add_argument('--use_agar_pretraining', type=int, default=False, help='Use AGAR pretraining')
    parser.add_argument('--patience_agar', type=int, default=3, help='Patience for AGAR training')
    parser.add_argument('--patience_photobox', type=int, default=10, help='Patience for Photobox training')
    parser.add_argument('--use_rotation_augmentation', type=int, default=False, help='Use rotation augmentation')
    parser.add_argument('--use_patch_augmentation', type=int, default=False, help='Use patch augmentation')
    parser.add_argument('--max_norm', type=float, default=-1, help='Max norm for gradients')
    # fmt: on

    parsed_args = parser.parse_args()

    exp_params = ExperimentParameters(
        learning_rate=parsed_args.learning_rate,
        use_coco_pretrained=bool(parsed_args.use_coco_pretrained),
        use_agar_pretraining=bool(parsed_args.use_agar_pretraining),
        patience_agar=parsed_args.patience_agar,
        patience_photobox=parsed_args.patience_photobox,
        use_rotation_augmentation=bool(parsed_args.use_rotation_augmentation),
        use_patch_augmentation=bool(parsed_args.use_patch_augmentation),
        max_norm=None if parsed_args.max_norm < 0 else parsed_args.max_norm,
    )

    print(exp_params.__dict__)

    main(parsed_args.device, exp_params)
