from datetime import datetime
from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.CbamConvLSTM import CbamConvLSTM
from model.CellularAutomaton import CellularAutomaton
from utils.GDAL import GDALLoader
from utils.UrbanDataset import UrbanDataset
from utils.BlockCollector import BlockCollector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_rasters(cfg: DictConfig):
    d_cfg = cfg["data"]
    t_cfg = cfg["test"]

    year_begin = t_cfg["year_end"] - t_cfg["predict_len"]
    raster_paths = GDALLoader.get_raster_paths(
        data_dir=Path(d_cfg["data_dir"]),
        spa_vars=d_cfg["spatial_vars"],
        year_range=range(year_begin - 1, t_cfg["year_end"]),
    )

    return GDALLoader.load_rasters(raster_paths)


def load_model(cfg: DictConfig):
    def load_checkpoint(model: nn.Module, checkpoint_file: Path):
        if checkpoint_file.is_file():
            print(f"=> loading checkpoint '{checkpoint_file}'")
            checkpoint = torch.load(
                checkpoint_file, map_location=device, weights_only=True
            )
            model.load_state_dict(checkpoint["state_dict"])
            print(
                f"=> loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})"
            )
        else:
            raise FileNotFoundError

    d_cfg = cfg["data"]
    m_cfg = cfg["model"]
    t_cfg = cfg["test"]

    # model
    model = CbamConvLSTM(
        in_channels=len(d_cfg["spatial_vars"]) + 1,
        hidden_channels=list(m_cfg["hidden_channels"]),
        num_layers=m_cfg["num_layers"],
        kernel_sizes=list(m_cfg["kernel_sizes"]),
        cbam_ca_reduction=m_cfg["cbam_ca_reduction"],
        cbam_sa_kernel_size=m_cfg["cbam_sa_kernel_size"],
        layer_norm=m_cfg["layer_norm"],
        input_shape=(d_cfg["train"]["block_size"], d_cfg["train"]["block_size"]),
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

    return model, data_loader


def gen_prob_mat(
    model: nn.Module, data_loader: DataLoader, shape: tuple[int, int], cfg: DictConfig
):
    drop_edge_width = cfg["test"]["drop_edge_width"]

    collector = BlockCollector((cfg["test"]["predict_len"],) + shape)

    model.eval()
    with torch.no_grad():
        for x, y, s, m, c in tqdm(data_loader, desc="gen prob"):
            x, y, s, m = x.to(device), y.to(device), s.to(device), m.to(device)

            # y: (t, b, c, h, w)
            y_hat = torch.sigmoid(model(x, y.size(0), s) * m)

            # y_hat: (b, t, h, w)
            y_hat = y_hat.squeeze(2).transpose(0, 1).cpu().detach().numpy()

            for block in range(y_hat.shape[0]):
                collector.put(y_hat[block], c[block], drop_edge_width)

    return collector.get()


def simulate(rasters: dict, prob: np.ndarray, cfg: DictConfig):
    year_begin = cfg["test"]["year_end"] - cfg["test"]["predict_len"]
    year_end = cfg["test"]["year_end"]

    ca = CellularAutomaton(rasters[year_begin - 1])

    for i, year in enumerate(range(year_begin, year_end)):
        _, fom = ca.step(prob[i], rasters[year])
        print(f"Year: {year}, FoM: {fom}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg: DictConfig):
    if cfg["test"]["use_prob_npy"]:
        prob = np.load(cfg["test"]["prob_npy_file"])

        if cfg["test"]["show_images"]:
            plt.imshow(prob[0])
            plt.colorbar()
            plt.show()

        rasters = load_rasters(cfg)
        simulate(rasters, prob, cfg)
    else:
        result_dir = Path(cfg["test"]["result_dir"]) / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        rasters = load_rasters(cfg)

        model, data_loader = load_model(cfg)

        prob = gen_prob_mat(
            model, data_loader, rasters[cfg["test"]["year_begin"]].shape, cfg
        )
        np.save(result_dir / "prob.npy", prob)

        simulate(rasters, prob, cfg)


if __name__ == "__main__":
    test()
