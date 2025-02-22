from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from datetime import datetime
from model.CbamConvLSTM import CbamConvLSTM
from utils.data import UrbanDataset
from utils.Trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    d_cfg = cfg["data"]
    m_cfg = cfg["model"]
    t_cfg = cfg["train"]

    model = CbamConvLSTM(
        in_channels=len(d_cfg["spatial_vars"]) + 1,
        hidden_channels=list(m_cfg["hidden_channels"]),
        num_layers=m_cfg["num_layers"],
        kernel_size=list(m_cfg["kernel_size"]),
        dropout=m_cfg["dropout"],
        batch_norm=m_cfg["batch_norm"],
        cbam_sa_kernel_size=m_cfg["cbam_sa_kernel_size"],
    )

    criterion = nn.MSELoss(reduction="none")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )

    grad_scaler = torch.amp.GradScaler()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, t_cfg["epochs"]
    )

    dataset = UrbanDataset(
        d_cfg["train"]["block_size"],
        Path(d_cfg["split_dir"]) / "train",
        d_cfg["spatial_vars"],
        t_cfg["sample_count"],
        range(t_cfg["year_begin"], t_cfg["year_end"]),
        t_cfg["prefix_len"],
        t_cfg["predict_len"],
        True,
    )

    trainer = Trainer(
        seed=t_cfg["seed"],
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        lr_scheduler=lr_scheduler,
        dataset=dataset,
        val_set_ratio=t_cfg["val_set_ratio"],
        batch_size=t_cfg["batch_size"],
        num_workers=t_cfg["num_workers"],
        collate_fn=UrbanDataset.collate_fn,
        start_epoch=t_cfg["start_epoch"],
        total_epochs=t_cfg["epochs"],
        teacher_forcing_epochs=t_cfg["teacher_forcing_epochs"],
        checkpoints_dir=Path(t_cfg["checkpoints_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S"),
        summary_dir=Path(t_cfg["summary_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    trainer.run()


if __name__ == "__main__":
    train()
