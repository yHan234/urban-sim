import numpy as np
from utils.GDAL import URBAN_AREA, NON_URBAN_AREA
from utils.FoM import FoM


class CellularAutomaton:
    def __init__(self, init_state: np.ndarray):
        self.init_state = init_state
        self.state = init_state
        self.last_gt = init_state

    def step(self, prob: np.ndarray, gt: np.ndarray):
        prob = self._add_noise(prob.astype(np.float32))

        demand_pos = ((gt == URBAN_AREA) & (self.last_gt == NON_URBAN_AREA)).sum()
        demand_neg = ((gt == NON_URBAN_AREA) & (self.last_gt == URBAN_AREA)).sum()

        print("demand", "+", demand_pos, "-", demand_neg)

        sim = self._simulate(prob, demand_pos, demand_neg)
        fom = FoM(sim, gt, self.init_state)

        self.last_gt = gt
        self.state = sim

        return sim, fom

    def _simulate(self, prob: np.ndarray, demand_pos: int, demand_neg: int):
        sim = self.state.copy()

        # expand
        expand_probs = prob[self.state == NON_URBAN_AREA]
        to_expand = np.argsort(-expand_probs)[:demand_pos]
        non_urban_area = np.argwhere(self.state == NON_URBAN_AREA)
        indices = non_urban_area[to_expand]
        sim[indices[:, 0], indices[:, 1]] = URBAN_AREA

        # shrink
        shrink_probs = prob[self.state == URBAN_AREA]
        to_shrink = np.argsort(shrink_probs)[:demand_neg]
        urban_area = np.argwhere(self.state == URBAN_AREA)
        indices = urban_area[to_shrink]
        sim[indices[:, 0], indices[:, 1]] = NON_URBAN_AREA

        return sim

    @staticmethod
    def _add_noise(prob_map: np.ndarray, noise_level: float = 0.01):

        noise = np.random.normal(0, noise_level, size=prob_map.shape).astype(np.float32)

        noise_map = prob_map + noise

        noise_map = np.clip(noise_map, 0, 1)

        assert noise_map.dtype == np.float32

        return noise_map
