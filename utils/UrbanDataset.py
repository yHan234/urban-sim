import os
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from .GDAL import GDALLoader, ILLEGAL_AREA, URBAN_AREA, NON_URBAN_AREA


class UrbanDataset(Dataset):
    def __init__(
        self,
        block_size: int,
        data_dir: Path,
        spa_vars: list[str] | None,
        sample_count: int | None,
        year_range: range,
        prefix_len: int,
        pred_len: int,
        augment: bool,
    ):
        self.block_size = block_size
        self.spa_vars = spa_vars

        assert (
            len(year_range) == prefix_len + pred_len
        ), "prefix_len + pred_len != year_range"
        self.year_range = year_range
        self.prefix_len = prefix_len
        self.pred_len = pred_len

        block_dirs = os.listdir(data_dir)

        if sample_count is not None:
            block_dirs = random.sample(block_dirs, sample_count)

        self.block_dirs = [data_dir / item for item in block_dirs]

        self.augment = augment

    def __getitem__(self, idx):
        block_dir = self.block_dirs[idx]
        base_dir = os.path.basename(block_dir)

        coord = tuple(map(int, base_dir.split("_")[:2]))

        # spa_tensor (c, h, w)
        if self.spa_vars is not None:
            spa_tensor = torch.empty(
                len(self.spa_vars), self.block_size, self.block_size
            )
            for i, spa in enumerate(self.spa_vars):
                spa_data = GDALLoader.load(block_dir / f"{spa}.tif")
                spa_tensor[i, :, :] = torch.as_tensor(spa_data).float()
        else:
            spa_tensor = None

        # land_tensor (t, c=1, h, w)
        land_tensor = torch.empty(
            len(self.year_range),
            1,
            self.block_size,
            self.block_size,
        ).float()
        for i, year in enumerate(self.year_range):
            land_data = GDALLoader.load(block_dir / f"land_{year}.tif")
            land_data = torch.as_tensor(land_data).float()
            land_tensor[i, 0, :, :] = land_data

        if self.augment:
            land_tensor, spa_tensor = self._augment(land_tensor, spa_tensor)

        x, y = torch.split(land_tensor, [self.prefix_len, self.pred_len])

        mask = torch.empty(1, self.block_size, self.block_size).float()
        mask[x[-1] == ILLEGAL_AREA] = 0
        mask[x[-1] == URBAN_AREA] = 1
        mask[x[-1] == NON_URBAN_AREA] = 1

        x[x == ILLEGAL_AREA] = NON_URBAN_AREA
        y[y == ILLEGAL_AREA] = NON_URBAN_AREA

        return x, y, spa_tensor, mask, coord

    def __len__(self):
        return len(self.block_dirs)

    def _augment(self, land: torch.Tensor, spa: torch.Tensor):
        def transform(tensor, rot_times, h_flip, v_flip):
            tensor = torch.rot90(tensor, rot_times, dims=(-2, -1))
            if h_flip:
                tensor = torch.flip(tensor, dims=(-1,))
            if v_flip:
                tensor = torch.flip(tensor, dims=(-2,))
            return tensor

        rot_times = random.randint(0, 3)
        h_flip = random.random() < 0.5
        v_flip = random.random() < 0.5

        land = transform(land, rot_times, h_flip, v_flip)
        spa = transform(spa, rot_times, h_flip, v_flip)

        return land, spa

    @staticmethod
    def collate_fn(batch):
        """
        x, y (t, b, c, h, w)
        s, m (   b, c, h, w)
        """

        x, y, s, m, c = zip(*batch)

        x = torch.stack(x, dim=1)
        y = torch.stack(y, dim=1)

        s = torch.stack(s)
        m = torch.stack(m)

        return x, y, s, m, c
