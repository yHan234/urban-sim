from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.CbamConvLSTM import CbamConvLSTM
from utils.data import GDALLoader, UrbanDataset, BlockCollector
from model.CellularAutomaton import CellularAutomaton

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model: nn.Module, checkpoint_file: Path):
    if checkpoint_file.is_file():
        print(f"=> loading checkpoint '{checkpoint_file}'")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"=> loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})")
    else:
        raise FileNotFoundError


def load(cfg: DictConfig):
    d_cfg = cfg["data"]
    m_cfg = cfg["model"]
    t_cfg = cfg["test"]

    # rasters
    year_begin = t_cfg["year_end"] - t_cfg["predict_len"]
    raster_paths = GDALLoader.get_raster_paths(
        data_dir=Path(d_cfg["data_dir"]),
        spa_vars=d_cfg["spatial_vars"],
        year_range=range(year_begin - 1, t_cfg["year_end"]),
    )

    rasters = GDALLoader.load_rasters(raster_paths)

    # model
    model = CbamConvLSTM(
        in_channels=len(d_cfg["spatial_vars"]) + 1,
        hidden_channels=list(m_cfg["hidden_channels"]),
        num_layers=m_cfg["num_layers"],
        kernel_size=list(m_cfg["kernel_size"]),
        cbam_sa_kernel_size=m_cfg["cbam_sa_kernel_size"],
    ).to(device)

    load_checkpoint(model, Path(t_cfg["checkpoint_file"]))

    # dataset
    dataset = UrbanDataset(
        d_cfg["test"]["block_size"],
        Path(d_cfg["split_dir"]) / "test",
        d_cfg["spatial_vars"],
        None,
        range(t_cfg["year_begin"], t_cfg["year_end"]),
        t_cfg["prefix_len"],
        t_cfg["predict_len"],
        False,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=t_cfg["num_workers"],
        collate_fn=UrbanDataset.collate_fn,
    )

    return rasters, model, data_loader


def gen_prob_mat(
    model: nn.Module, data_loader: DataLoader, tmpl_raster: np.ndarray, cfg: DictConfig
):
    collector = BlockCollector(
        (
            cfg["test"]["predict_len"],
            tmpl_raster.shape[0],
            tmpl_raster.shape[1],
        )
    )

    model.eval()
    with torch.no_grad():
        for x, y, s, m, c in tqdm(data_loader, desc="gen prob"):
            x, y, s = x.to(device), y.to(device), s.to(device)

            # y: (t, b, c, h, w)
            # y_hat: (b, t, h, w)
            y_hat = (
                model(x, y.size(0), s).squeeze(2).transpose(0, 1).cpu().detach().numpy()
            )
            y_hat *= m[:, np.newaxis, :, :]

            for block in range(y_hat.shape[0]):
                collector.put(y_hat[block], c[block], cfg["test"]["drop_edge_width"])

    return collector.get()


def simulate(rasters: dict, prob: np.ndarray, cfg: DictConfig):
    year_begin = cfg["test"]["year_end"] - cfg["test"]["predict_len"]
    year_end = cfg["test"]["year_end"]

    ca = CellularAutomaton(
        rasters[year_begin - 1],
        rasters["region"],
    )

    for i, year in enumerate(range(year_begin, year_end)):
        _, fom = ca.step(prob[i], rasters[year])
        print(f"year: {year}, fom: {fom}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg: DictConfig):
    rasters, model, data_loader = load(cfg)
    prob = gen_prob_mat(model, data_loader, rasters["region"], cfg)
    simulate(rasters, prob, cfg)


if __name__ == "__main__":
    test()
