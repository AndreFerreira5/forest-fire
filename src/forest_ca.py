import random

import numpy as np
from noise import pnoise2
from numpy.f2py.crackfortran import dimensionpattern
from scipy.signal import convolve2d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Seeds for repeatability
rng = np.random.default_rng(seed=0)
random.seed(0)

INT32_MAX = 2_147_483_647
INT32_MIN = -2_147_483_648

class Forest:
    def __init__(
            self,
            dimensions,
            p_tree=0.5,
            p_grow=1e-3,
            f_lightning=1e-5,
            density=6.0,
            noise_octaves=4,
            seed=0,
            wind_dir=(0, 0),
            wind_speed=2.0,
            radiant_decay=0.4,
            ignition_base_prob=0.8,
            max_ignition_distance=3,
            spotting_prob=0.01,
            spotting_range=15
    ):
        self.n_steps = 0 # number of steps

        # board
        self.H, self.W = dimensions
        self.board = np.zeros(dimensions)

        # probabilities
        self.p_tree = p_tree
        self.p_grow = float(p_grow)
        self.f_lightning = float(f_lightning)

        # noise
        self.scale = density
        self.noise_octaves = noise_octaves
        self.seed = int(seed)
        if self.seed > INT32_MAX:
            self.seed -= 2 * INT32_MAX

        # kernel
        self.kernel = np.ones((3, 3), dtype=np.uint8)
        self.kernel[1,1] = 0

        # trees stats history
        self.alive_trees_hist = []
        self.burning_trees_hist = []
        self.burned_trees_hist = []

        # wind
        self.wind_dir = np.asarray(wind_dir, dtype=float)
        self.wind_dir /= np.hypot(*self.wind_dir) + 1e-9 # normalise so that ||wind_dir|| == 1 even if a diagonal is supplied
        self.wind_speed = float(wind_speed)

        # spread parameters
        self.radiant_decay = radiant_decay # radiant heat decay (nearby cells) - crown
        self.ignition_base_prob = ignition_base_prob # ignition chance for adjacent cells
        self.max_ignition_distance = max_ignition_distance # for direct/radiant heat

        # fire spotting parameters
        self.spotting_prob = spotting_prob
        self.spotting_range = spotting_range

        self.heat_kernel = self.create_heat_kernel()
        self.init_perlin()


    def init_perlin(self):
        for i in range(self.H):
            for j in range(self.W):
                n = pnoise2(i/self.scale, j/self.scale, octaves=self.noise_octaves, base=self.seed)
                self.board[i, j] = 1 if (n + 1)/2 < self.p_tree else 0

        # Set a single random tree on fire
        trees = np.argwhere(self.board == 1)  # Coordinates of all good trees
        if len(trees) > 0:
            idx = rng.integers(0, len(trees))  # Random index
            i, j = trees[idx]
            self.board[i, j] = 2


    def create_heat_kernel(self):
        """Create a Gaussian-like kernel for fire spread"""
        size = 2 * self.max_ignition_distance + 1
        kernel = np.zeros((size, size))
        center = self.max_ignition_distance

        for i in range(size):
            for j in range(size):
                distance = np.hypot(i - center, j - center)
                if distance <= 1:  # Direct neighbors (Moore neighborhood)
                    kernel[i, j] = self.ignition_base_prob
                elif 1 < distance <= self.max_ignition_distance:
                    # Sharp decay for non-adjacent cells
                    kernel[i, j] = self.ignition_base_prob * np.exp(-self.radiant_decay * (distance - 1))
        return kernel


    def fire_spotting(self, board):
        """Simulates fire spotting (embers or leaves flying away and starting secondary fires)"""
        burning_cells = np.argwhere(self.board == 2)
        mask = np.zeros(board.shape, dtype=bool)
        for i, j in burning_cells:
            if rng.random() < self.spotting_prob:
                base_distance = rng.triangular(left=2, mode=3, right=self.spotting_range)
                angle_variation = rng.uniform(-np.pi / 4, np.pi / 4)  # ±45° around main flow

                # Wind direction in polar form
                wind_angle = np.arctan2(self.wind_dir[0], self.wind_dir[1])

                # Drift: farther & more aligned with stronger wind
                distance = base_distance * (1.0 + 0.7 * self.wind_speed)
                angle = wind_angle + angle_variation
                #distance = rng.triangular(left=2, mode=3, right=self.spotting_range) # Random distance with triangular distribution
                #angle = rng.uniform(0, 2 * np.pi) # Random direction for ember travel

                # Convert angle to cell coordinates
                di = int(np.round(distance * np.cos(angle)))
                dj = int(np.round(distance * np.sin(angle)))

                # Check if there is a tree at the landing location
                if (i + di) < self.H and (j + dj) < self.W:
                    ni, nj = (i + di), (j + dj)
                    if mask[ni, nj] == 1:
                        mask[ni, nj] = 2
        return mask


    def count_neighbors(self):
        empty_neighbors = convolve2d(self.board == 0, self.kernel, mode='same', boundary='wrap')
        good_neighbors = convolve2d(self.board == 1, self.kernel, mode='same', boundary='wrap')
        burning_neighbors = convolve2d(self.board == 2, self.kernel, mode='same', boundary='wrap')
        burned_neighbors = convolve2d(self.board == 3, self.kernel, mode='same', boundary='wrap')
        return empty_neighbors, good_neighbors, burning_neighbors, burned_neighbors

    def step(self):
        self.n_steps += 1

        burning_cells = np.argwhere(self.board == 2)
        self.burning_trees_hist.append(len(burning_cells))
        self.alive_trees_hist.append(len(np.argwhere(self.board == 1)))
        self.burned_trees_hist.append(len(np.argwhere(self.board == 3)))

        # Prepare next board
        next_board = np.copy(self.board)

        # Rule 1: Burning trees become burned
        next_board[self.board == 2] = 3

        # Rule 2: Direct and radiant heat spread
        # Build per-cell ignition probabilities (highest influence rule)
        prob_map = np.zeros_like(self.board)

        for i, j in burning_cells:
            for di in range(-self.max_ignition_distance,
                            self.max_ignition_distance + 1):
                for dj in range(-self.max_ignition_distance,
                                self.max_ignition_distance + 1):

                    if di == 0 and dj == 0:
                        continue  # the burning cell itself

                    # Current cell coordinates
                    ni = i + di
                    nj = j + dj

                    # Ignore if out of bounds
                    if ni >= self.H or nj >= self.W:
                        continue

                    # Euclidean distance from the burning cell
                    distance = np.hypot(di, dj)

                    # Skip cells beyond the configured range
                    if distance > self.max_ignition_distance:
                        continue

                    # Direct neighbours
                    if distance <= 1:
                        prob = self.ignition_base_prob

                    # Radiant heat (sharp exponential decay)
                    else:
                        prob = (self.ignition_base_prob *
                                np.exp(-self.radiant_decay * (distance - 1)))

                    # wind amplification
                    # Dot-product gives cosθ between wind and [di, dj] (range −1…1)
                    cos_theta = (np.array([di, dj], dtype=float) @ self.wind_dir) / (distance + 1e-9)

                    # Cells exactly down-wind get prob × (1 + wind_speed)
                    # Cells exactly up-wind get prob × (1 – wind_speed)   (floored at zero)
                    wind_factor = 1.0 + self.wind_speed * cos_theta
                    wind_factor = np.clip(wind_factor, 0.0, None)  # never negative

                    prob *= wind_factor

                    # Store ONLY the strongest influence so far
                    if prob > prob_map[ni, nj]:
                        prob_map[ni, nj] = prob

        # Perform ignitions on currently good trees
        good_trees = self.board == 1
        rand_vals = rng.random(self.board.shape)
        ignite_mask = good_trees & (rand_vals < prob_map)
        next_board[ignite_mask] = 2

        # Rule 3: Apply fire spotting
        spotting_mask = self.fire_spotting(self.board)
        next_board[spotting_mask] = 2

        # Rule 4: Regrowth
        grow_mask = (self.board == 3) & (rng.random(self.board.shape) < self.p_grow)
        next_board[grow_mask] = 1

        # Rule 4: Lightning
        lightning_mask = (self.board == 1) & (rng.random(self.board.shape) < self.f_lightning)
        next_board[lightning_mask] = 2

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
    FOREST_DIMENSIONS = (100, 100)
    forest = Forest(FOREST_DIMENSIONS, density=30, noise_octaves=30, p_tree=0.4, wind_dir=(1, 1))
    forest.render()

    STEPS = 200
    for _ in range(STEPS):
        forest.step()
        #forest.render()

    n_steps = list(range(1, STEPS+1))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Alive Trees Over Time", "Burning Trees Over Time", "Burned Trees Over Time"),
        horizontal_spacing=0.08,
        vertical_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(x=n_steps, y=forest.alive_trees_hist, name="Alive Trees Over Time"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=n_steps, y=forest.burning_trees_hist, name="Burning Trees Over Time"),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=n_steps, y=forest.burned_trees_hist, name="Burned Trees Over Time"),
        row=2, col=1
    )

    fig.update_layout(
        title_text="Forest Statistics Over Time",
        showlegend=False,
        height=600, width=650
    )

    fig.show()


if __name__ == "__main__":
    main()
