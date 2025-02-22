from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


# TODO: optimize criterion
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
        start_epoch: int,
        total_epochs: int,
        teacher_forcing_epochs: bool,
        checkpoints_dir: str,
        summary_dir: str,
    ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.lr_scheduler = lr_scheduler
        self.teacher_forcing_epochs = teacher_forcing_epochs

        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.get_data_loader(
            dataset, val_set_ratio, batch_size, num_workers, collate_fn
        )

        self.start_epoch = start_epoch
        self.total_epochs = total_epochs

        self.checkpoints_dir = checkpoints_dir

        self.train_summary_writer = SummaryWriter(Path(summary_dir) / "train")
        self.val_summary_writer = SummaryWriter(Path(summary_dir) / "val")

    def get_data_loader(
        self, dataset, val_set_ratio, batch_size, num_workers, collate_fn
    ):
        num_val_samples = int(len(dataset) * val_set_ratio)
        num_train_samples = len(dataset) - num_val_samples

        g = torch.Generator()
        g.manual_seed(self.seed)

        train_set, val_set = data.random_split(
            dataset,
            [num_train_samples, num_val_samples],
            g,
        )

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

    def run(self):
        for epoch in tqdm(
            range(self.start_epoch, self.total_epochs), desc="Epoch", position=0
        ):
            self.epoch = epoch

            avg_loss = self._train()
            self.train_summary_writer.add_scalar("AvgLoss", avg_loss, epoch)

            avg_loss = self._val()
            self.val_summary_writer.add_scalar("AvgLoss", avg_loss, epoch)

            self.lr_scheduler.step()
            self.save_checkpoint(epoch)

        self.train_summary_writer.close()
        self.val_summary_writer.close()

    def _train(self) -> float:
        tot_loss = 0
        self.model.train()

        for i, (x, y, s, m, _) in enumerate(
            tqdm(
                self.train_data_loader,
                desc="Train",
                position=1,
                leave=False,
            )
        ):
            self.optimizer.zero_grad()
            with torch.amp.autocast(self.device.type):
                x, y, s = x.to(self.device), y.to(self.device), s.to(self.device)
                if self.epoch < self.teacher_forcing_epochs:
                    mask = self._gen_teacher_mask(y)
                    y_hat = self.model(x, y.size(0), s, y, mask)
                else:
                    y_hat = self.model(x, y.size(0), s)
                loss = self.criterion(y_hat, y).squeeze(2)
                loss *= torch.from_numpy(m).to(self.device)
                loss = loss.sum()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            tot_loss += loss.item()

        return tot_loss / (i + 1)

    def _val(self) -> float:
        tot_loss = 0
        self.model.eval()

        with torch.no_grad():
            for i, (x, y, s, m, _) in enumerate(
                tqdm(self.val_data_loader, desc="Val", position=1, leave=False)
            ):
                x, y, s = x.to(self.device), y.to(self.device), s.to(self.device)
                y_hat = self.model(x, y.size(0), s)
                loss = self.criterion(y_hat, y).squeeze(2)
                loss *= torch.from_numpy(m).to(self.device)
                loss = loss.sum()
                tot_loss += loss.item()

        return tot_loss / (i + 1)

    def _gen_teacher_mask(self, teacher: torch.Tensor) -> list[torch.Tensor]:

        base = max(1 - (self.epoch / self.teacher_forcing_epochs), 0)

        mask = torch.tensor(
            np.random.uniform(low=base - 0.1, high=base + 0.1, size=teacher.size())
        ).to(self.device, dtype=torch.float16)

        mask = torch.clamp(mask, 0, 1)

        return mask

    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "grad_scaler": self.grad_scaler.state_dict(),
            "start_epoch": self.start_epoch,
            "seed": self.seed,
        }
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        filename = self.checkpoints_dir / f"checkpoint_{epoch}.pth"
        torch.save(checkpoint, filename)

    def load_checkpoint(self, checkpoint_file: Path):
        if checkpoint_file.is_file():
            print(f"=> loading checkpoint '{checkpoint_file}'")
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler"])
            self.seed = checkpoint["seed"]
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            print(
                f"=> loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})"
            )
        else:
            raise FileNotFoundError

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
