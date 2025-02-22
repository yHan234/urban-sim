from pathlib import Path
import numpy as np
from osgeo import gdal
from tqdm import tqdm

gdal.UseExceptions()

URBAN_AREA = 1
NON_URBAN_AREA = 0
ILLEGAL_AREA = -1

LAND_NO_DATA_VAL = float(-1)
SPA_NO_DATA_VAL = float(0)


def get_no_data_value(path: Path):
    return LAND_NO_DATA_VAL if path.name.startswith("land") else SPA_NO_DATA_VAL


class GDALSaver:
    def __init__(self, template_image: Path) -> None:
        tmpl_dataset = gdal.Open(str(template_image))
        self.geo_trans = tmpl_dataset.GetGeoTransform()
        self.prj = tmpl_dataset.GetProjection()
        self.driver = gdal.GetDriverByName("GTiff")

    def _save_multi_band_block(
        self, tif_array, tif_path, tif_type, no_data_value, geo_trans
    ):
        band_count = tif_array.shape[0]
        dataset = self.driver.Create(
            str(tif_path),
            tif_array.shape[2],
            tif_array.shape[1],
            band_count,
            tif_type,
            options=["COMPRESS=LZW"],
        )
        dataset.SetGeoTransform(geo_trans)
        dataset.SetProjection(self.prj)
        for i in range(1, band_count + 1):
            band = dataset.GetRasterBand(i)
            band.SetNoDataValue(no_data_value)
            band.WriteArray(tif_array[i - 1, :, :])
        dataset.FlushCache()

    def _save_single_band_block(
        self, tif_array, tif_path, tif_type, no_data_value, geo_trans
    ):
        dataset = self.driver.Create(
            str(tif_path),
            tif_array.shape[1],
            tif_array.shape[0],
            1,
            tif_type,
            options=["COMPRESS=LZW"],
        )
        dataset.SetGeoTransform(geo_trans)
        dataset.SetProjection(self.prj)
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(no_data_value)
        band.WriteArray(tif_array)
        dataset.FlushCache()

    def save(
        self,
        tif_array: np.ndarray,
        tif_path: Path,
        start_row: int | None = None,
        start_col: int | None = None,
        tif_type=gdal.GDT_Float32,
        no_data_value=None,
    ):
        if tif_type == gdal.GDT_Int16:
            tif_array[tif_array >= 2**16] = 2**16 - 1

        if start_row is not None and start_col is not None:
            geo_trans = (
                self.geo_trans[0] + start_col * self.geo_trans[1],
                self.geo_trans[1],
                self.geo_trans[2],
                self.geo_trans[3] + start_row * self.geo_trans[5],
                self.geo_trans[4],
                self.geo_trans[5],
            )
        else:
            geo_trans = self.geo_trans

        if no_data_value is None:
            no_data_value = get_no_data_value(tif_path)

        if len(tif_array.shape) == 3:
            self._save_multi_band_block(
                tif_array, tif_path, tif_type, no_data_value, geo_trans
            )
        elif len(tif_array.shape) == 2:
            self._save_single_band_block(
                tif_array, tif_path, tif_type, no_data_value, geo_trans
            )
        else:
            raise NotImplementedError()


class GDALLoader:
    @staticmethod
    def get_raster_paths(
        data_dir: Path,
        spa_vars: list[str],
        year_range: range,
    ) -> dict:
        return {var: data_dir / f"{var}.tif" for var in spa_vars} | {
            year: data_dir / "year" / f"land_{year}.tif" for year in year_range
        }

    @staticmethod
    def load_rasters(raster_paths: dict) -> dict:
        return {
            name: GDALLoader.load(path)
            for name, path in tqdm(raster_paths.items(), desc="Loading rasters")
        }

    @staticmethod
    def load(ras: Path) -> np.ndarray:
        no_data_value = get_no_data_value(ras)

        dataset = gdal.Open(str(ras), gdal.GA_ReadOnly)
        band_count = dataset.RasterCount

        bands = []
        for i in range(1, band_count + 1):
            band = dataset.GetRasterBand(i).ReadAsArray().astype(np.float32)
            band_no_data = dataset.GetRasterBand(i).GetNoDataValue()
            band[band == band_no_data] = no_data_value
            bands.append(band)

        if len(bands) > 1:
            bands = np.stack(bands)
        else:
            bands = bands[0]

        return bands
