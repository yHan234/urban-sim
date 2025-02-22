from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        seed: int,
        device: torch.device,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        grad_scaler: torch.amp.GradScaler,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        dataset: data.Dataset,
        val_set_ratio: float,
        batch_size: int,
        num_workers: int,
        collate_fn: callable,
        total_epochs: int,
        teacher_forcing_epochs: bool,
        metric: callable,
        checkpoints_dir: Path,
        summary_dir: Path,
        checkpoint: Path | None = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.lr_scheduler = lr_scheduler
        self.teacher_forcing_epochs = teacher_forcing_epochs
        self.metric = metric

        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.batch_size = batch_size
        self._get_data_loader(
            dataset, val_set_ratio, batch_size, num_workers, collate_fn
        )

        self.epoch = 0
        self.total_epochs = total_epochs

        self.checkpoints_dir = checkpoints_dir

        self.train_sw = SummaryWriter(Path(summary_dir) / "train")
        self.val_sw = SummaryWriter(Path(summary_dir) / "val")

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def run(self):
        for epoch in tqdm(
            range(self.epoch, self.total_epochs), desc="Epoch", position=0
        ):
            self.epoch = epoch

            self.train_sw.add_scalar("Learning Rate", self.lr, epoch)

            avg_loss, avg_metric = self._train()
            self.train_sw.add_scalar("AvgLoss", avg_loss, epoch)
            self.train_sw.add_scalar("AvgMetric", avg_metric, epoch)

            avg_loss, avg_metric = self._val()
            self.val_sw.add_scalar("AvgLoss", avg_loss, epoch)
            self.val_sw.add_scalar("AvgMetric", avg_metric, epoch)

            self.lr_scheduler.step()
            self._save_checkpoint()

        self.train_sw.close()
        self.val_sw.close()

    def _train(self) -> float:
        tot_loss = 0
        tot_metric = 0
        valid_metric = 0
        self.model.train()

        for x, y, s, m, _ in tqdm(
            self.train_data_loader, desc="Train", position=1, leave=False
        ):
            self.optimizer.zero_grad()
            with torch.amp.autocast(self.device.type):
                x, y, s, m = (
                    x.to(self.device),
                    y.to(self.device),
                    s.to(self.device),
                    m.to(self.device),
                )
                if self.epoch < self.teacher_forcing_epochs:
                    mask = self._gen_teacher_mask(y)
                    y_hat = self.model(x, y.size(0), s, y, mask)
                else:
                    y_hat = self.model(x, y.size(0), s)
                loss = self.criterion(y_hat, y, m)
                metric, _ = self.metric(y_hat, y, x[-1], m)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            tot_loss += loss.item()
            if metric is not None:
                tot_metric += metric
                valid_metric += 1

        return tot_loss / self.data_size_train, tot_metric / valid_metric

    def _val(self) -> float:
        tot_loss = 0
        tot_metric = 0
        valid_metric = 0
        self.model.eval()

        with torch.no_grad():
            for x, y, s, m, _ in tqdm(
                self.val_data_loader, desc="Val", position=1, leave=False
            ):
                x, y, s, m = (
                    x.to(self.device),
                    y.to(self.device),
                    s.to(self.device),
                    m.to(self.device),
                )
                y_hat = self.model(x, y.size(0), s)
                loss = self.criterion(y_hat, y, m)
                metric, _ = self.metric(y_hat, y, x[-1], m)
                tot_loss += loss.item()
                if metric is not None:
                    tot_metric += metric
                    valid_metric += 1

        return tot_loss / self.data_size_val, tot_metric / valid_metric

    def _gen_teacher_mask(self, teacher: torch.Tensor) -> list[torch.Tensor]:

        base = max(1 - (self.epoch / self.teacher_forcing_epochs), 0)

        mask = torch.tensor(
            np.random.uniform(low=base - 0.1, high=base + 0.1, size=teacher.size())
        ).to(self.device, dtype=torch.float16)

        mask = torch.clamp(mask, 0, 1)

        return mask

    def _get_data_loader(
        self, dataset, val_set_ratio, batch_size, num_workers, collate_fn
    ):
        num_val_samples = int(len(dataset) * val_set_ratio)
        num_train_samples = len(dataset) - num_val_samples

        g = torch.Generator()
        g.manual_seed(self.seed)

        train_set, val_set = data.random_split(
            dataset, [num_train_samples, num_val_samples], g
        )

        self.data_size_train = len(train_set)
        self.data_size_val = len(val_set)

        self.train_data_loader = data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        self.val_data_loader = data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def _save_checkpoint(self):
        checkpoint = {
            "epoch": self.epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "grad_scaler": self.grad_scaler.state_dict(),
            "seed": self.seed,
        }
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        filename = self.checkpoints_dir / f"checkpoint_{self.epoch}.pth"
        torch.save(checkpoint, filename)

    def _load_checkpoint(self, checkpoint_file: Path):
        if checkpoint_file.is_file():
            print(f"loading checkpoint '{checkpoint_file}'")
            checkpoint = torch.load(
                checkpoint_file, map_location=self.device, weights_only=True
            )
            self.epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler"])
            self.seed = checkpoint["seed"]
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            print(
                f"loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})"
            )
        else:
            raise FileNotFoundError
