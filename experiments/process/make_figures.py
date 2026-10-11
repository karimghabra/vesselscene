"""Figures for experiments/process/burst_to_graph.tex: python experiments/process/make_figures.py experiments/process"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

OUT = sys.argv[1]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif"})

# Figure 1: scale selection on 1-D profiles
x = np.arange(-600, 601, dtype=float)
sig = np.exp(np.linspace(np.log(0.7), np.log(30), 300))
def response(p):
    return np.array([-s**2 * gaussian_filter1d(p, s, order=2)[600] for s in sig])
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for s0, c in ((1.5, S1), (3.0, S2), (6.0, S3)):
    R = response(np.exp(-x**2 / (2 * s0**2)))
    i = int(np.argmax(R))
    ax.plot(sig, R, color=c, lw=1.6, label=fr"Gaussian, $s_0={s0:g}$ px ($\sigma^*={sig[i]:.2f}$)")
    ax.plot(sig[i], R[i], "o", ms=4, color=c)
r = 8.0
p = 2 * np.sqrt(np.clip(r**2 - x**2, 0, None)); p /= p.max()
R = response(p); i = int(np.argmax(R))
ax.plot(sig, R, color=INK2, lw=1.4, ls="--", label=fr"round lumen, $r={r:g}$ px ($\sigma^*={sig[i]:.2f}$)")
ax.plot(sig[i], R[i], "o", ms=4, color=INK2)
ax.legend(loc="upper left", frameon=False, fontsize=7, labelcolor=INK)
ax.axhline(2 / 3**1.5, color=GRID, lw=0.8, zorder=0)
ax.set_xscale("log"); ax.set_xlabel(r"scale $\sigma$ (px)"); ax.set_ylabel(r"$-\sigma^2 L_{xx}$ at the centre")
ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
ax.set_ylim(0, 0.66)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_scale_selection.pdf"); plt.close(fig)

# Figure 2: Frangi vesselness on a synthetic OD image with a crossing and a fork
H, W = 220, 340
yy, xx = np.mgrid[0:H, 0:W].astype(float)
pts = np.stack([xx.ravel(), yy.ravel()], 1)
def tube(path, r, mu=0.03):
    t = np.linspace(0, 1, 2000)[:, None]
    P = np.concatenate([path[k] + t * (path[k + 1] - path[k]) for k in range(len(path) - 1)])
    d, _ = cKDTree(P).query(pts)
    return (mu * 2 * np.sqrt(np.clip(r**2 - d**2, 0, None))).reshape(H, W)
A = np.array
wide = tube(A([[0, 55], [170, 48], [340, 62]]), 8)
thin = tube(A([[95, 125], [165, 0]]), 2)                      # thin over wide, about 60 deg
x1 = tube(A([[20, 205], [130, 105]]), 4)                      # equal widths, crossing at about 85 deg
x2 = tube(A([[25, 110], [125, 210]]), 4)
trunk = tube(A([[170, 160], [250, 160]]), 4)
b1, b2 = tube(A([[250, 160], [340, 120]]), 3), tube(A([[250, 160], [340, 200]]), 3)
fork = np.maximum.reduce([trunk, b1, b2])                     # one lumen at the fork
CENTRES = {"thin over wide": (137, 49), "equal crossing": (72, 157), "fork": (250, 160)}
rng = np.random.default_rng(0)
OD = ndi.gaussian_filter(wide + thin + x1 + x2 + fork, 1.2) + rng.normal(0, 0.01, (H, W))

sigmas = [2 ** (k / 2) for k in range(0, 8)]                  # 1 ... 11.3 px
def hessian_eigs(img, s):
    Lxx = ndi.gaussian_filter(img, s, order=(0, 2)); Lyy = ndi.gaussian_filter(img, s, order=(2, 0))
    Lxy = ndi.gaussian_filter(img, s, order=(1, 1))
    Lxx, Lyy, Lxy = s**2 * Lxx, s**2 * Lyy, s**2 * Lxy
    m, q = (Lxx + Lyy) / 2, np.sqrt(((Lxx - Lyy) / 2) ** 2 + Lxy**2)
    a, b = m - q, m + q
    small = np.abs(a) < np.abs(b)
    return np.where(small, a, b), np.where(small, b, a)       # lambda1 (small), lambda2 (large)
noise = rng.normal(0, 0.01, (H, W))
V = np.zeros((len(sigmas), H, W))
for k, s in enumerate(sigmas):
    l1, l2 = hessian_eigs(OD, s)
    n1, n2 = hessian_eigs(noise, s)
    c = 3 * np.sqrt(np.mean(n1**2 + n2**2))                    # c in noise units
    RB, S = l1 / np.where(l2 == 0, 1e-12, l2), np.sqrt(l1**2 + l2**2)
    V[k] = np.where(l2 > 0, 0, np.exp(-RB**2 / (2 * 0.5**2)) * (1 - np.exp(-S**2 / (2 * c**2))))
Vmax, kstar = V.max(0), V.argmax(0)
sstar = np.where(Vmax > 0.2, np.array(sigmas)[kstar], np.nan)

fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.95))
for a in axs: a.set_xticks([]); a.set_yticks([])
im = axs[0].imshow(OD, cmap="Greys", vmin=0, vmax=0.5); axs[0].set_title("(a) optical density", fontsize=8)
fig.colorbar(im, ax=axs[0], fraction=0.035, pad=0.02)
im = axs[1].imshow(Vmax, cmap="Blues", vmin=0, vmax=1); axs[1].set_title(r"(b) vesselness $V$", fontsize=8)
fig.colorbar(im, ax=axs[1], fraction=0.035, pad=0.02)
from skimage.morphology import skeletonize
skel = skeletonize(Vmax > 0.5)
sline = np.where(ndi.binary_dilation(skel, iterations=1), ndi.grey_dilation(np.where(skel, np.array(sigmas)[kstar], 0), size=3), np.nan)
cm = plt.get_cmap("Oranges").copy(); cm.set_bad("white")
im = axs[2].imshow(sline, cmap=cm, vmin=1, vmax=11.3)
axs[2].set_title(r"(c) $\sigma^*$ (px) on the ridge of $V$", fontsize=8)
fig.colorbar(im, ax=axs[2], fraction=0.035, pad=0.02)
for lab, (cx, cy) in CENTRES.items():
    for a in axs[1:]:
        a.add_patch(plt.Circle((cx, cy), 17, fill=False, ec=S2, lw=1.1))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_frangi_example.pdf"); fig.savefig(f"{OUT}/fig_frangi_example.png", dpi=160)
plt.close(fig)
for lab, (cx, cy) in CENTRES.items():
    print(lab, "V at the centre", round(float(Vmax[cy, cx]), 3), "| skeleton pixels within 6 px:", int(skel[cy-6:cy+7, cx-6:cx+7].sum()))
print("median V on plain stretches:", round(float(np.median(Vmax[skel])), 3))
# skeleton topology near each junction: arms crossing an annulus, branch pixels, loops
from skimage.measure import euler_number, label
nb = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
for lab_, (cx, cy) in CENTRES.items():
    d = np.hypot(xx - cx, yy - cy)
    arms = label(skel & (d > 12) & (d < 17), connectivity=2).max()
    disc = skel & (d <= 14)
    branch = int((disc & (nb >= 3)).sum())
    holes = int(label(~disc[cy-15:cy+16, cx-15:cx+16], connectivity=1).max() - 1)
    print(f"{lab_}: arms leaving {arms}, branch pixels {branch}, enclosed loops {holes}")
# the thin vessel's own centreline: V along its path inside the wide vessel
t = np.linspace(0, 1, 400)
px, py = 95 + t * (165 - 95), 125 + t * (0 - 125)
vals = Vmax[np.round(py).astype(int), np.round(px).astype(int)]
sel = np.hypot(px - 137, py - 49) < 10
print("thin path: min V within 10 px of the crossing", round(float(vals[sel].min()), 3))
by, bx = np.nonzero(skel & (nb >= 3) & (np.hypot(xx - 250, yy - 160) < 60))
if len(bx):
    k = int(np.argmin(np.hypot(bx - 250, by - 160)))
    print("fork: nearest skeleton branch point at", (int(bx[k]), int(by[k])), "distance", round(float(np.hypot(bx[k] - 250, by[k] - 160)), 1), "px")
