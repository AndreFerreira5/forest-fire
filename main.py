import numpy as np
from noise import pnoise2
from scipy.signal import convolve2d
import plotly.express as px


class Forest:
    def __init__(self, dimensions, p_tree=0.5, density=6.0, noise_octaves=4, seed=0):
        self.H, self.W = dimensions
        self.board = np.zeros(dimensions)
        self.p_tree = p_tree
        self.scale = density
        self.noise_octaves = noise_octaves
        self.seed = seed
        self.init_perlin()


    def init_perlin(self):
        for i in range(self.H):
            for j in range(self.W):
                n = pnoise2(i/self.scale, j/self.scale, octaves=self.noise_octaves, base=self.seed)
                self.board[i, j] = 1 if (n + 1)/2 < self.p_tree else 0


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
    forest = Forest(FOREST_DIMENSIONS, density=3.5, noise_octaves=1, p_tree=0.4)
    forest.render()


if __name__ == "__main__":
    main()
