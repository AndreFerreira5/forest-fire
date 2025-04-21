import numpy as np
from noise import pnoise2


def init_perlin(
        shape,
        p_tree=0.5,
        scale=50.0,
        octaves=4,
        seed=0
):
    H, W = shape
    board = np.zeros(shape, dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            n = pnoise2(i/scale, j/scale, octaves=octaves, base=seed)
            board[i, j] = 1 if (n + 1)/2 > p_tree else 0

    return board
    #return np.array([1 if (pnoise2(i/scale, j/scale, octaves=octaves, base=seed)+1)/2 > p_tree else 0 for j in range(W) for i in range(H)], dtype=np.uint8)


def main():
    FOREST_DIMENSIONS = (20, 40)
    print(init_perlin(FOREST_DIMENSIONS, scale=6))


if __name__ == "__main__":
    main()
