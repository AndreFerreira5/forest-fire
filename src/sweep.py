import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp, random, time
from tqdm import tqdm
from forest_ca import Forest
import secrets

# ------------------------------------------------------------------ #
# 0. quick-test switch                                               #
# ------------------------------------------------------------------ #
QUICK_TEST = False        # set False for the full 3 000-step sweep
STEPS   = 400 if QUICK_TEST else 3_000
N_SEEDS = 1  if QUICK_TEST else 5

# ------------------------------------------------------------------ #
# 1. parameter ranges                                                #
# ------------------------------------------------------------------ #
p_vals = np.logspace(-5, -2, 5 if QUICK_TEST else 9)
f_vals = np.logspace(-8, -5, 4 if QUICK_TEST else 7)

FIXED_PARAMS = dict(p_tree=0.4, density=30, noise_octaves=30)

# ------------------------------------------------------------------ #
def run_one(p, f, seed, timeout=90 if QUICK_TEST else 600):
    """Return (p, f, metric) or None on timeout."""
    t0 = time.time()
    forest = Forest((100, 100), p_grow=p, f_lightning=f,
                    seed=seed, **FIXED_PARAMS)
    for _ in range(STEPS):
        forest.step()
        if time.time() - t0 > timeout:          # safeguard
            return None
    metric = np.mean(forest.alive_trees_hist[-STEPS//2:])
    return (p, f, metric)

# job list ----------------------------------------------------------
jobs = [(p, f, secrets.randbelow(1_000_000)) for p in p_vals for f in f_vals
        for _ in range(N_SEEDS)]

heat   = np.zeros((len(p_vals), len(f_vals)))
counts = np.zeros_like(heat)

# ------------------------------------------------------------------ #
# 2. multiprocessing with progress bar                               #
# ------------------------------------------------------------------ #

def collect(result):
    if result is None:        # timed-out worker
        return
    p, f, m = result
    i = np.where(p_vals == p)[0][0]
    j = np.where(f_vals == f)[0][0]
    heat[i, j] += m
    counts[i, j] += 1
    pbar.update()


with mp.Pool(mp.cpu_count(), maxtasksperchild=10) as pool:
    with tqdm(total=len(jobs), desc="Simulations") as pbar:
        for p, f, seed in jobs:
            pool.apply_async(run_one, (p, f, seed), callback=collect)
        pool.close()
        pool.join()

if counts.min() == 0:
    print("Warning: some grid cells have no completed runs.")
heat = np.divide(heat, counts, out=np.zeros_like(heat), where=counts>0)

# ------------------------------------------------------------------ #
# 3. plot                                                            #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(6,4))
im = ax.imshow(heat, origin='lower',
               extent=[f_vals[0], f_vals[-1], p_vals[0], p_vals[-1]],
               aspect='auto', cmap='viridis')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('f_lightning'); ax.set_ylabel('p_grow')
fig.colorbar(im, ax=ax, label='mean ρ̄ (last half of run)')
title = "Quick-test heat-map" if QUICK_TEST else "ρ̄ vs (p_grow, f_lightning)"
ax.set_title(title)
plt.tight_layout(); plt.savefig('pf_heatmap.png', dpi=300); plt.show()
