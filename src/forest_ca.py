import random
import os
import matplotlib.pyplot as plt
import numpy as np
from noise import pnoise2
from numpy.f2py.crackfortran import dimensionpattern
from scipy.signal import convolve2d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import imageio
from scipy.ndimage import label
from scipy.stats import linregress

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
        self._prev_burning = 1
        self.fire_sizes = []
        self.entropy_hist = []
        self.R0_hist = []
        self.largest_cluster_hist = []
        self.mean_cluster_hist = []
        self.tau_hist = []
        self.equil_hist = []

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
                    if 0 <= ni < self.H and 0 <= nj < self.W:
                        if mask[ni, nj] == 1:
                            mask[ni, nj] = 2
        return mask


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
                    if ni < 0 or nj < 0 or ni >= self.H or nj >= self.W:
                        continue

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

        # ── fire-event accounting ──────────────────────────────────────────────
        burning_now = self.burning_trees_hist[-1]

        if self._prev_burning > 0 and burning_now == 0:  # fire just ended
            burned_this_event = self.burned_trees_hist[-1] - self.burned_trees_hist[-2]
            if burned_this_event:  # ignore zero-size blips
                self.fire_sizes.append(int(burned_this_event))

        self._prev_burning = burning_now

        self.entropy_hist.append(self.spatial_entropy())
        self.R0_hist.append(self.fire_R0())
        _, cstats = self.cluster_stats()
        self.largest_cluster_hist.append(cstats["largest_cluster"])
        self.mean_cluster_hist.append(cstats["mean_cluster"])
        tau, *_ = self.powerlaw_exponent()
        self.tau_hist.append(tau)
        self.equil_hist.append(int(self.is_equilibrated()))


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


    def render_frame(self, save_dir="results"):
        os.makedirs(save_dir, exist_ok=True)

        cmap = plt.cm.get_cmap("gist_heat", 4)
        plt.imshow(self.board, cmap=cmap, vmin=0, vmax=3)
        plt.axis("off")

        plt.title(f"Step {self.n_steps}", fontsize=10)

        plt.savefig(f"{save_dir}/frame_{self.n_steps:04d}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def cluster_stats(self):
        """
        Returns:
            sizes (np.ndarray) – 1-D array of all tree-cluster sizes
            stats (dict)       – {'n_clusters', 'largest', 'mean'}
        """
        labels, n_clust = label(self.board == 1, structure=np.ones((3, 3)))
        if n_clust:
            sizes = np.bincount(labels.ravel())[1:]  # skip background 0
            stats = dict(
                n_clusters=int(n_clust),
                largest_cluster=int(sizes.max()),
                mean_cluster=float(sizes.mean()),
            )
        else:
            sizes = np.empty(0, dtype=int)
            stats = dict(n_clusters=0, largest_cluster=0, mean_cluster=0.0)
        return sizes, stats


    def spatial_entropy(self):
        """
        Shannon entropy of the four-state field, in bits (0 = fully ordered).
        """
        counts = np.bincount(self.board.ravel().astype(int), minlength=4)
        probs  = counts / counts.sum()
        nz     = probs > 0
        return float(-(probs[nz] * np.log2(probs[nz])).sum())


    def fire_R0(self):
        """
        R_t = (new burning cells) / (previous burning cells).
        Returns 0 when no trees are burning at t–1 to avoid division by zero.
        """
        if len(self.burning_trees_hist) < 2:
            return 0.0
        prev = self.burning_trees_hist[-2]
        curr = self.burning_trees_hist[-1]
        return float(curr / prev) if prev else 0.0


    def powerlaw_exponent(self, xmin=5):
        """
        Ordinary-least-squares estimate of τ in P(s) ~ s^-τ (s ≥ xmin).
        Returns (tau, r_value, stderr).  Needs at least 5 events.
        """
        data = np.array(self.fire_sizes, dtype=float)
        data = data[data >= xmin]
        if len(data) < 5:
            return np.nan, np.nan, np.nan
        y = np.log10(data)
        hist, edges = np.histogram(y, bins='auto')  # log-binned
        cdf = np.cumsum(hist[::-1])[::-1] / hist.sum()
        x = edges[:-1]  # log10(s)
        m, _, r, _, se = linregress(x, np.log10(cdf + 1e-12))
        tau = -m  # slope = –τ
        return tau, r, se


    def is_equilibrated(self, window=300, eps=1e-3):
        """
        Returns True when the alive-tree count has stabilised:
           (max – min) / mean  <  ε  over the last `window` steps.
        """
        if len(self.alive_trees_hist) < window:
            return False
        data = np.array(self.alive_trees_hist[-window:], dtype=float)
        return (np.ptp(data) / data.mean()) < eps



def main():
    RENDER = False
    FOREST_DIMENSIONS = (100, 100)
    forest = Forest(FOREST_DIMENSIONS, density=30, noise_octaves=30, p_tree=0.4, wind_dir=(1, 1))
    if RENDER: forest.render_frame()

    STEPS = 10000
    for _ in range(STEPS):
        forest.step()
        #forest.render()
        if RENDER: forest.render_frame()

    if RENDER:
        frame_dir = "results"
        with imageio.get_writer("forest_fire.gif", mode="I", duration=0.1) as writer:
            for step in range(STEPS + 1):
                filename = f"{frame_dir}/frame_{step:04d}.png"
                image = imageio.imread(filename)
                writer.append_data(image)


    n_steps = np.arange(1, STEPS+1)
    dash = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            "Alive Trees", "Burning Trees", "Burned Trees",
            "Spatial Entropy (bits)", "Fire $R_0$", "Largest Cluster Size",
            "Mean Cluster Size", "Power-law τ̂", "Equilibrated?"  # last row partly blank
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.12,
    )

    def add(line, r, c):
        dash.add_trace(line, row=r, col=c)

    add(go.Scatter(x=n_steps, y=forest.alive_trees_hist), 1, 1)
    add(go.Scatter(x=n_steps, y=forest.burning_trees_hist), 1, 2)
    add(go.Scatter(x=n_steps, y=forest.burned_trees_hist), 1, 3)

    add(go.Scatter(x=n_steps, y=forest.entropy_hist), 2, 1)
    add(go.Scatter(x=n_steps, y=forest.R0_hist), 2, 2)
    add(go.Scatter(x=n_steps, y=forest.largest_cluster_hist), 2, 3)

    add(go.Scatter(x=n_steps, y=forest.mean_cluster_hist), 3, 1)
    add(go.Scatter(x=n_steps, y=forest.tau_hist),                     3, 2)
    add(go.Scatter(x=n_steps, y=forest.equil_hist, mode="lines"),     3, 3)
    dash.update_yaxes(range=[-0.05, 1.05], row=3, col=3, nticks=2)

    dash.update_layout(
        title_text="Forest-Fire CA: Time-series Dashboard",
        height=900, width=1000, showlegend=False,
    )
    dash.show()

    # cumulative fire-size distribution
    if forest.fire_sizes:
        counts, edges = np.histogram(forest.fire_sizes, bins="auto")
        cdf = np.cumsum(counts[::-1])[::-1]
        x = edges[:-1]  # left bin edges

        fig_fire = go.Figure(
            go.Scatter(x=x, y=cdf, mode="lines+markers")
        )
        fig_fire.update_layout(
            title="Cumulative fire-size distribution  (log–log)",
            xaxis=dict(title="fire size (cells)", type="log"),
            yaxis=dict(title="P(size ≥ s)", type="log"),
        )
        fig_fire.show()



if __name__ == "__main__":
    main()
