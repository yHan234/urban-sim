import numpy as np


class BlockCollector:
    def __init__(self, shape: tuple[int, int, int]):
        self.acc = np.zeros(shape, dtype=np.float32)
        self.overlap = np.zeros(shape, dtype=np.int32)

    def put(self, block: np.ndarray, coord: tuple[int, int], drop_edge_width: int = 0):
        assert drop_edge_width >= 0 and drop_edge_width * 2 < block.shape[-1]

        block = block[
            :, drop_edge_width:-drop_edge_width, drop_edge_width:-drop_edge_width
        ].astype(np.float32)
        block_size = block.shape[-1]

        x, y = coord[0] + drop_edge_width, coord[1] + drop_edge_width

        slice_x = slice(x, min(x + block_size, self.acc.shape[1]))
        slice_y = slice(y, min(y + block_size, self.acc.shape[2]))

        to_acc = self.acc[:, slice_x, slice_y]

        try:
            to_acc += block[:, : to_acc.shape[1], : to_acc.shape[2]]
        except ValueError:
            print(
                f"\033[91m\033[1m\nError: block shape {block.shape} does not fit at coord {coord}.\n"
                f"Accumulated blocks shape: {self.acc.shape}, slice_x: {slice_x}, slice_y: {slice_y}\n"
                f"Sliced Accumulated blocks shape: {self.acc[:, slice_x, slice_y].shape}\033[0m"
            )
            raise

        self.overlap[:, slice_x, slice_y] += 1

    def get(self) -> np.ndarray[np.float32]:
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(self.overlap != 0, self.acc / self.overlap, 0)
        return res
