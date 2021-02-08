import argparse
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import functional as F, BatchNorm2d
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from backbone.build import build_model
from pns import SlimPruner
from pns.functional import update_bn_grad


class LitModel(pl.LightningModule):
    def __init__(self, args):

        super().__init__()

        self.data_dir = "datasets"
        self.dataset = args.dataset
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.sparsity_train = args.sparsity_train
        self.s = args.s
        self.bn_weight_vis_period = args.bn_weight_vis_period
        self.net = args.net
        self.epochs = args.epochs
        self.save_dir = args.save_dir
        self.prune_ratio = args.prune_ratio

        self.model = build_model(self.net, num_classes=10)
        self.is_pruned = False

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)

        if (
            self.global_step != 0
            and self.global_step % self.bn_weight_vis_period == 0
            and self.sparsity_train
        ):
            for name, m in self.model.named_modules():
                if isinstance(m, BatchNorm2d):
                    self.logger.experiment.add_histogram(
                        f"{name}_weights", m.weight.data.cpu().numpy(), self.global_step
                    )

        return loss

    def backward(self, loss, optimizer, optimizer_idx):
        super(LitModel, self).backward(loss, optimizer, optimizer_idx)
        if self.sparsity_train:
            update_bn_grad(self.model, self.s)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return logits, y

    def validation_epoch_end(self, outputs) -> None:
        logits = torch.cat([it[0] for it in outputs], dim=0)
        y = torch.cat([it[1] for it in outputs], dim=0)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=True)
        # dump metric
        metric = {
            "test_acc": acc.item(),
            "net": self.net,
            "params": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "s": self.s,
        }
        if self.is_pruned:
            metric["prune_ratio"] = self.prune_ratio
        else:
            metric["prune_ratio"] = 0.0

        with open(
            os.path.join(self.save_dir, "metric.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metric, f, indent=2, ensure_ascii=False)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def setup(self, stage=None):
        if self.dataset == "mnist":
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                ]
            )

            if stage == "fit" or stage is None:
                self.train_dataset = MNIST(
                    self.data_dir, train=True, transform=transform, download=True
                )

            if stage == "test" or stage is None:
                self.test_dataset = MNIST(
                    self.data_dir, train=False, transform=transform, download=True
                )
        elif self.dataset == "cifar10":
            self.train_dataset = CIFAR10(
                self.data_dir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
            )
            self.test_dataset = CIFAR10(
                self.data_dir,
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="resnet18")
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--save_dir", default="output")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="if load an exist model: do test on model; pruning model; fine tune ",
    )
    parser.add_argument(
        "--ckpt_pruned",
        action="store_true",
        help="Try to load pruning result in checkpoint",
    )

    parser.add_argument("--sparsity_train", action="store_true")
    parser.add_argument("--s", type=float, default=0.001)

    parser.add_argument(
        "--fine_tune", action="store_true", help="是否在训练结束或者加载 ckpt 之后 fine_tune"
    )
    parser.add_argument("--prune_schema", type=str, default="./configs/resnet18.json")
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=20,
        help="fine tune epoch after apply slimming prune",
    )
    parser.add_argument("--fine_tune_learning_rate", type=float, default=1e-4)
    parser.add_argument("--prune_ratio", type=float, default=0.75)
    parser.add_argument("--bn_weight_vis_period", type=int, default=30)

    return parser.parse_args()


class TFLogger(TensorBoardLogger):
    def __init__(self, save_dir: str):
        super().__init__(save_dir, name="")

    @property
    def log_dir(self) -> str:
        return self.save_dir

    @property
    def root_dir(self) -> str:
        return self.save_dir

    @rank_zero_only
    def save(self) -> None:
        super(TensorBoardLogger, self).save()


if __name__ == "__main__":
    args = parse_args()

    model = LitModel(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=args.save_dir,
        filename="{epoch:02d}-{train_loss:.2f}-{test_acc:.3f}",
        save_top_k=1,
        save_last=True,
        mode="min",
        # save_weights_only=True,
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
        # limit_train_batches=10,
        benchmark=True,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=TFLogger(args.save_dir),
    )

    if args.ckpt is None:
        trainer.fit(model)
        trainer.test()
        last_model_path = checkpoint_callback.last_model_path
    else:
        model = LitModel.load_from_checkpoint(args.ckpt, args=args)
        if args.ckpt_pruned:
            pruner = SlimPruner(model)
            print("Load pruning result from model state dict")
            checkpoint = torch.load(args.ckpt)
            pruner.apply_pruning_result(checkpoint[SlimPruner.PRUNING_RESULT_KEY])
            model = pruner.pruned_model
        trainer.test(model)
        last_model_path = args.ckpt

    if not args.fine_tune:
        exit(0)

    origin_model_save_dir = args.save_dir

    # start fine tune
    args.learning_rate = args.fine_tune_learning_rate
    args.sparsity_train = False
    args.save_dir = os.path.join(args.save_dir, f"pruned_{args.prune_ratio}")

    restored_model = LitModel.load_from_checkpoint(last_model_path, args=args)

    pruner = SlimPruner(restored_model, args.prune_schema)
    pruning_result = pruner.run(args.prune_ratio)

    print(
        f"Save pruning result to model state_dict with {SlimPruner.PRUNING_RESULT_KEY} key"
    )

    if args.ckpt is None:

        def save_pruning_result(checkpoint):
            checkpoint[SlimPruner.PRUNING_RESULT_KEY] = pruning_result

        model.on_save_checkpoint = save_pruning_result
        trainer.save_checkpoint(
            os.path.join(origin_model_save_dir, "model_with_pruning_result.ckpt")
        )

    pruned_model = pruner.pruned_model
    pruned_model.is_pruned = True

    fine_tune_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=args.save_dir,
        filename="{epoch:02d}-{train_loss:.2f}-{test_acc:.3f}",
        save_top_k=1,
        save_last=True,
        mode="min",
        # save_weights_only=True,
    )
    fine_tune_trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.fine_tune_epochs,
        check_val_every_n_epoch=5,
        num_sanity_val_steps=0,
        # limit_train_batches=10,
        benchmark=True,
        callbacks=[
            fine_tune_checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=TFLogger(args.save_dir),
    )
    fine_tune_trainer.fit(pruned_model)
    fine_tune_trainer.test()
