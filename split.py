import os
from pathlib import Path
import hydra
import numpy as np
import random as rd
import shutil
from omegaconf import DictConfig
from tqdm import tqdm
from utils.GDAL import (
    GDALSaver,
    GDALLoader,
    LAND_NO_DATA_VAL,
    SPA_NO_DATA_VAL,
    ILLEGAL_AREA,
)


def split_blocks(
    rasters: dict,
    block_size: int,
    block_step: int,
    crop: bool,
    crop_required_block_count: int,
    save_path: Path,
    spa_vars: list[str],
    year_range: range,
    gdal_saver: GDALSaver,
) -> None:
    def is_legal(land: np.ndarray):
        return not np.all(land == ILLEGAL_AREA)

    def pad(block: np.ndarray, pad_value: float):
        h, w = block.shape
        padding_rows = block_size - h
        padding_cols = block_size - w

        if padding_rows > 0 or padding_cols > 0:
            return np.pad(
                block,
                ((0, padding_rows), (0, padding_cols)),
                "constant",
                constant_values=pad_value,
            )
        else:
            return block

    def save_blocks(
        save_path: Path,
        var_name: str,
        arr: np.ndarray,
        start_row: int,
        start_col: int,
    ) -> None:
        if not save_path.exists():
            os.makedirs(save_path)
        save_path = save_path / var_name
        gdal_saver.save(
            arr,
            save_path,
            start_row=start_row,
            start_col=start_col,
        )

    if crop:
        assert block_size == block_step, "step != size"

    if save_path.exists():
        print(f'Removing "{save_path}"')
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)

    tmpl_land = rasters[year_range[0]]
    blocks_to_save = []

    num_rows, num_cols = tmpl_land.shape
    if crop:
        existing_block = set()
        pbar = tqdm(total=crop_required_block_count, desc="Cropping")
        while len(blocks_to_save) < crop_required_block_count:
            row_start, col_start = rd.randint(block_size, num_rows), rd.randint(
                block_size, num_cols
            )
            row_end, col_end = row_start + block_size, col_start + block_size

            if (row_start, col_start) in existing_block:
                continue
            else:
                existing_block.add((row_start, col_start))

            block = tmpl_land[row_start:row_end, col_start:col_end]
            if is_legal(block):
                blocks_to_save.append([row_start, col_start])
                pbar.update(1)
        pbar.close()
    else:
        for row_start in tqdm(
            range(0, num_rows, block_step),
            desc="Row",
            position=0,
        ):
            for col_start in tqdm(
                range(0, num_cols, block_step),
                desc="Col",
                position=1,
                leave=False,
            ):
                row_end, col_end = row_start + block_size, col_start + block_size
                block = tmpl_land[row_start:row_end, col_start:col_end]
                if is_legal(block):
                    blocks_to_save.append((row_start, col_start))

    for row_start, col_start in tqdm(blocks_to_save, desc="Saving blocks"):
        row_end, col_end = row_start + block_size, col_start + block_size
        block_save_path = save_path / f"{row_start}_{col_start}"

        for var in spa_vars:
            var_block = pad(
                rasters[var][row_start:row_end, col_start:col_end], SPA_NO_DATA_VAL
            )
            save_blocks(
                block_save_path,
                f"{var}.tif",
                var_block,
                row_start,
                col_start,
            )

        for year in year_range:
            land_block = pad(
                rasters[year][row_start:row_end, col_start:col_end], LAND_NO_DATA_VAL
            )
            save_blocks(
                block_save_path,
                f"land_{year}.tif",
                land_block,
                row_start,
                col_start,
            )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def split(cfg: DictConfig):
    cfg = cfg["data"]

    spatial_vars = cfg["spatial_vars"]

    # train
    print("Splitting train set")

    raster_paths = GDALLoader.get_raster_paths(
        Path(cfg["data_dir"]),
        spatial_vars,
        range(cfg["train"]["year_begin"], cfg["train"]["year_end"]),
    )

    rasters = GDALLoader.load_rasters(raster_paths)

    gdal_saver = GDALSaver(raster_paths[cfg["train"]["year_begin"]])
    split_blocks(
        rasters,
        block_size=cfg["train"]["block_size"],
        block_step=cfg["train"]["block_step"],
        crop=cfg["train"]["crop"],
        crop_required_block_count=cfg["train"]["crop_required_block_count"],
        save_path=Path(cfg["split_dir"]) / "train",
        spa_vars=spatial_vars,
        year_range=range(cfg["train"]["year_begin"], cfg["train"]["year_end"]),
        gdal_saver=gdal_saver,
    )

    # test
    print("Splitting test set")

    raster_paths = GDALLoader.get_raster_paths(
        data_dir=Path(cfg["data_dir"]),
        spa_vars=spatial_vars,
        year_range=range(cfg["test"]["year_begin"], cfg["test"]["year_end"]),
    )

    rasters = GDALLoader.load_rasters(raster_paths)

    split_blocks(
        rasters,
        block_size=cfg["test"]["block_size"],
        block_step=cfg["test"]["block_step"],
        crop=cfg["test"]["crop"],
        crop_required_block_count=cfg["test"]["crop_required_block_count"],
        save_path=Path(cfg["split_dir"]) / "test",
        spa_vars=spatial_vars,
        year_range=range(cfg["test"]["year_begin"], cfg["test"]["year_end"]),
        gdal_saver=gdal_saver,
    )


if __name__ == "__main__":
    split()
