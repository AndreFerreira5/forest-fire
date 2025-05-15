import random

import numpy as np
from noise import pnoise2
from scipy.signal import convolve2d
import plotly.express as px


rng = np.random.default_rng()


class Forest:
    def __init__(self, dimensions, p_tree=0.5, density=6.0, noise_octaves=4, seed=0):
        self.H, self.W = dimensions
        self.board = np.zeros(dimensions)
        self.p_tree = p_tree
        self.scale = density
        self.noise_octaves = noise_octaves
        self.seed = seed
        self.kernel = np.ones((3, 3), dtype=np.uint8)
        self.kernel[1,1] = 0
        self.wind_vectors = [

        ]
        self.init_perlin()


    def init_perlin(self):
        for i in range(self.H):
            for j in range(self.W):
                n = pnoise2(i/self.scale, j/self.scale, octaves=self.noise_octaves, base=self.seed)
                self.board[i, j] = (1 if random.random()<self.p_tree else 2) if (n + 1)/2 < self.p_tree else 0


    def count_neighbors(self):
        empty_neighbors = convolve2d(self.board == 0, self.kernel, mode='same', boundary='wrap')
        good_neighbors = convolve2d(self.board == 1, self.kernel, mode='same', boundary='wrap')
        burning_neighbors = convolve2d(self.board == 2, self.kernel, mode='same', boundary='wrap')
        burned_neighbors = convolve2d(self.board == 3, self.kernel, mode='same', boundary='wrap')
        return empty_neighbors, good_neighbors, burning_neighbors, burned_neighbors

    def step(self):
        # unpack neighbor counts for each of the four states
        c0, c1, c2, c3 = self.count_neighbors()

        # prepare an empty board for next generation
        next_board = np.empty_like(self.board, dtype=np.uint8)

        # 0 (empty) → stays 0
        next_board[self.board == 0] = 0

        # 1 (alive) → if any burning neighbor, become 2; otherwise stay 1
        alive = (self.board == 1)
        next_board[alive & (c2 > 0)] = 2
        next_board[alive & (c2 == 0)] = 1

        # 2 (burning) → always become 3 (burned)
        next_board[self.board == 2] = 3

        # 3 (burned) → stays 3
        next_board[self.board == 3] = 3

        # swap in the new board
        self.board = next_board


    def render(self):
        colorscale = [
            [0.0, 'black'], [1/4, 'black'],
            [1/4, "forestgreen"],   [2/4, "forestgreen"],
            [2/4, "orangered"], [3/4, "orangered"],
            [3/4, "lightgray"], [1.0, "lightgray"],
        ]

        fig = px.imshow(
            self.board,
            color_continuous_scale=colorscale,
            zmin=0, zmax=3,
            origin='lower',
            aspect='equal',
            labels={"color": "State"}
        )
        fig.update_layout(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["Empty", "Good Tree", "Burning Tree", "Burned Tree"],
            )
        )
        fig.update_traces(hovertemplate="State: %{z}<extra></extra>")
        fig.show()


def main():
    FOREST_DIMENSIONS = (20, 40)
    forest = Forest(FOREST_DIMENSIONS, density=2, noise_octaves=30, p_tree=0.4)
    forest.render()

    for _ in range(5):
        forest.step()
        forest.render()


if __name__ == "__main__":
    main()
