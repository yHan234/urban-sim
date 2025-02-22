from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
from datetime import datetime
from model.CbamConvLSTM import CbamConvLSTM
from model.MaskedBCEWithLogitsLoss import MaskedBCEWithLogitsLoss
from utils.UrbanDataset import UrbanDataset
from utils.FoM import FoM_Prob
from utils.Trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    d_cfg = cfg["data"]
    m_cfg = cfg["model"]
    t_cfg = cfg["train"]

    model = CbamConvLSTM(
        in_channels=len(d_cfg["spatial_vars"]) + 1,
        num_layers=m_cfg["num_layers"],
        hidden_channels=list(m_cfg["hidden_channels"]),
        kernel_sizes=list(m_cfg["kernel_sizes"]),
        cbam_ca_reduction=m_cfg["cbam_ca_reduction"],
        cbam_sa_kernel_size=m_cfg["cbam_sa_kernel_size"],
        dropout=m_cfg["dropout"],
        layer_norm=m_cfg["layer_norm"],
        input_shape=(d_cfg["train"]["block_size"], d_cfg["train"]["block_size"]),
    )

    criterion = MaskedBCEWithLogitsLoss(reduction="sum")

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
        None,
        range(t_cfg["year_begin"], t_cfg["year_end"]),
        t_cfg["prefix_len"],
        t_cfg["predict_len"],
        True,
    )

    if t_cfg["resume"]:
        checkpoints_dir = Path(t_cfg["checkpoints_dir"]) / t_cfg["timestamp"]
        summary_dir = Path(t_cfg["summary_dir"]) / t_cfg["timestamp"]
        checkpoint_file = checkpoints_dir / t_cfg["checkpoint_file"]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoints_dir = Path(t_cfg["checkpoints_dir"]) / timestamp
        summary_dir = Path(t_cfg["summary_dir"]) / timestamp

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
        total_epochs=t_cfg["epochs"],
        teacher_forcing_epochs=t_cfg["teacher_forcing_epochs"],
        metric=FoM_Prob,
        checkpoints_dir=checkpoints_dir,
        summary_dir=summary_dir,
        checkpoint=checkpoint_file if t_cfg["resume"] else None,
    )

    trainer.run()


if __name__ == "__main__":
    train()
