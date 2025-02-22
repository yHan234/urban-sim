import numpy as np
from utils.data import URBAN_AREA


class CellularAutomaton:
    def __init__(self, init_state: np.ndarray):
        self.init_state = init_state
        self.state = init_state
        self.last_gt = init_state

    def step(self, prob: np.ndarray, gt: np.ndarray):
        prob = self._add_noise(prob.astype(np.float32))

        land_demand = np.count_nonzero(gt != self.last_gt)

        sim = self._simulate(prob, land_demand)
        fom = self.FoM(sim, gt, self.init_state)

        self.last_gt = gt
        self.state = sim

        return sim, fom

    def _simulate(self, prob: np.ndarray, land_demand: int):
        sim = self.state.copy()
        prob[(self.state == URBAN_AREA)] = 0
        indices = np.unravel_index(np.argsort(-prob, axis=None), prob.shape)
        indices = np.column_stack(indices)[:land_demand]

        sim[indices[:, 0], indices[:, 1]] = URBAN_AREA
        return sim

    @staticmethod
    def FoM(pred: np.ndarray, gt: np.ndarray, prev: np.ndarray):
        true_changes = gt != prev
        pred_changes = pred != prev

        num_true_changes = np.count_nonzero(true_changes)
        num_matched_changes = np.count_nonzero(true_changes & pred_changes)

        print(f"true: {num_true_changes}, matched: {num_matched_changes}")

        fom = num_matched_changes / (
            2 * (num_true_changes - num_matched_changes) + num_matched_changes
        )

        return fom

    @staticmethod
    def _add_noise(prob_map: np.ndarray, noise_level: float = 0.01):

        noise = np.random.normal(0, noise_level, size=prob_map.shape).astype(np.float32)

        noise_map = prob_map + noise

        noise_map = np.clip(noise_map, 0, 1)

        assert noise_map.dtype == np.float32

        return noise_map
