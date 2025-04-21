import numpy as np
from noise import pnoise2
from scipy.signal import convolve2d


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
                self.board[i, j] = 1 if (n + 1)/2 > self.p_tree else 0




def main():
    FOREST_DIMENSIONS = (20, 40)
    forest = Forest(FOREST_DIMENSIONS)


if __name__ == "__main__":
    main()
