import csv, itertools, multiprocessing as mp
from forest_ca import Forest
import numpy as np, itertools, random
import lhs
from scipy.stats import qmc

def run_one(param_tuple):
    p, f, p_tree, dens, run_seed = param_tuple
    forest = Forest((100,100), p_tree=p_tree,
                    p_grow=p, f_lightning=f,
                    density=dens, seed=run_seed)
    #while not forest.is_equilibrated():
    for _ in range(100):
        forest.step()
    print("FINISHED")
    metrics = dict(
        p=p, f=f, p_tree=p_tree, density=dens, seed=run_seed,
        rho=np.mean(forest.alive_trees_hist[-300:]),
        tau=forest.powerlaw_exponent()[0],
        entropy=forest.entropy_hist[-1],
        Teq=forest.n_steps,
        Pmega = np.mean(np.array(forest.fire_sizes) > 0.1*forest.H*forest.W),
    )
    return metrics

N_RUNS = 800
dim    = 4  # p, f, p_tree, density

sampler = qmc.LatinHypercube(d=dim, seed=0)
lhs_points = sampler.random(N_RUNS)

# Map unit-cube → real ranges
def scale(x, lo, hi, log=False):
    if log:        return lo * (hi/lo)**x
    else:          return lo + (hi-lo)*x

if __name__ == "__main__":
    param_space = []
    for u in lhs_points:
        p = scale(u[0], 1e-5, 1e-2, log=True)
        f = scale(u[1], 1e-8, 1e-5, log=True)
        p_tree = scale(u[2], 0.25, 0.6)
        density = int(scale(u[3], 5, 80))
        seed = random.randrange(2 ** 31)
        param_space.append((p, f, p_tree, density, seed))

    print(f"{param_space}")
    print(f"{len(param_space)=}")  # 4×4×3×3×5 = 720 runs


    with mp.Pool() as pool, open("results.csv","w",newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["p","f","p_tree","density","seed",
                                           "rho","tau","entropy","Teq","Pmega"])
        w.writeheader()
        for row in pool.imap_unordered(run_one, param_space):
            w.writerow(row)
