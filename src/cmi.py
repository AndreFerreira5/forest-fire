import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from forest_ca import Forest

# ---------------- utility: conditional mutual information ---------------
def cond_mutual_info(x, y, z, bins=256):
    """
    Estimate I(x; y | z) for 1-D integer arrays x, y, z of equal length.
    Histogram-based (naïve) estimator; good enough for a sanity check.
    """
    # joint histograms (counts)
    h_xyz, _ = np.histogramdd(np.stack([x, y, z],  axis=1), bins=bins)
    h_xz,  _ = np.histogramdd(np.stack([x, z],     axis=1), bins=bins)
    h_yz,  _ = np.histogramdd(np.stack([y, z],     axis=1), bins=bins)
    h_z,   _ = np.histogramdd(z[:, None],          bins=bins)

    # normalise to probabilities (+ tiny constant to avoid log0)
    p_xyz = h_xyz / h_xyz.sum() + 1e-12
    p_xz  = h_xz  / h_xz.sum()  + 1e-12
    p_yz  = h_yz  / h_yz.sum()  + 1e-12
    p_z   = h_z   / h_z.sum()   + 1e-12

    # entropies
    H_xyz = entropy(p_xyz.ravel(), base=2)
    H_xz  = entropy(p_xz.ravel(),  base=2)
    H_yz  = entropy(p_yz.ravel(),  base=2)
    H_z   = entropy(p_z.ravel(),   base=2)

    # I(X;Y|Z) = H(X,Z)+H(Y,Z)−H(Z)−H(X,Y,Z)
    return H_xz + H_yz - H_z - H_xyz

# ---------------- simulate ----------------------------------------------
STEPS   = 10_000          # after burn-in
BURNIN  = 500

forest = Forest((100, 100), p_tree=0.4, p_grow=1e-3, f_lightning=1e-5)
states = []

for _ in range(STEPS + BURNIN):
    forest.step()
    if forest.n_steps > BURNIN:
        # quick 64-bit hash of the board
        states.append(np.frombuffer(forest.board.tobytes()[:8], dtype='>u8')[0])

states = np.array(states, dtype=np.uint64)

# ---------------- compute CMI for lags 1–3 ------------------------------
lags = [1, 2, 3]
cmi  = []

for k in lags:
    x = states[:-k-1]
    y = states[k:-1]
    z = states[1:-k]
    cmi.append(cond_mutual_info(x, y, z))

# ---------------- plot ---------------------------------------------------
plt.figure(figsize=(4,3))
plt.bar(lags, cmi, width=0.6)
plt.xlabel("lag k")
plt.ylabel("I(X$_{t-k}$; X$_t$ | X$_{t-1}$)  [bits]")
plt.title("Conditional mutual information")
plt.tight_layout()
plt.savefig("cmi_vs_lag.pdf", dpi=300)
plt.show()

print("CMI values (bits):", dict(zip(lags, cmi)))
