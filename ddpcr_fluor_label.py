"""
ddPCR fluorescence droplet auto-labeling script v13.

Label definition:
    0 = Negative
    1 = Positive
    2 = Bubble / empty dark well

v13 focuses on two failure modes:
    1. Weak positives missed by a single high SNR threshold.
    2. Bright speckles/noise being labeled as positive droplets.

The classifier treats fluorescence speckles as negative unless the bright region
is large enough, compact enough, and close enough to the expected droplet scale.
"""

import glob
import os
import random
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

# ============================================================
# 0) Path config
# ============================================================
dataset_root = r"E:\qiujiuer_data\pycharm_file\patchs\dataset_dark"
fluor_dir = os.path.join(dataset_root, "images_fluor")
output_dir = os.path.join(dataset_root, "labels")
max_images = None

# ============================================================
# 1) Hyper-parameters
# ============================================================
GRID_N = 20
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 35
HOUGH_P1 = 50
HOUGH_P2 = 15
HOUGH_MIN_R = 10
HOUGH_MAX_R = 22
FALLBACK_FRAC = 0.38

# Feature masks. Median features are more robust than max/mean for speckles.
CENTER_R = 0.62
INNER_R = 0.95
BG_INNER = 1.25
BG_OUTER = 1.75

# Bubble: only nearly black wells count as bubbles.
BUBBLE_GMAX_DELTA = 4.0
BUBBLE_P95_DELTA = 5.0
BUBBLE_MEDIAN_DELTA = 2.0

# Score thresholding: score = robust droplet median - local background median.
NEG_TAIL_FRAC = 0.35
SCORE_STRONG_SIGMA = 4.0
SCORE_STRONG_MIN_DELTA = 10.0
SCORE_RESCUE_SIGMA = 2.3
SCORE_RESCUE_MIN_DELTA = 5.0
GMM_SEP_MIN = 0.85

# True positives are droplet-scale blobs; speckles are smaller and irregular.
MIN_LOW_AREA_PX_STRONG = 45
MIN_LOW_AREA_PX_RESCUE = 45
MIN_LOW_AREA_RATIO_STRONG = 0.055
MIN_LOW_AREA_RATIO_RESCUE = 0.050
MIN_LOW_EQ_RADIUS_FRAC_STRONG = 0.22
MIN_LOW_EQ_RADIUS_FRAC_RESCUE = 0.21
MIN_CORE_AREA_RATIO_STRONG = 0.018
MIN_CIRC_STRONG = 0.38
MIN_CIRC_RESCUE = 0.35
MIN_ASPECT_STRONG = 0.48
MIN_ASPECT_RESCUE = 0.45
MAX_CENTER_OFFSET_STRONG = 0.62
MAX_CENTER_OFFSET_RESCUE = 0.65
MAX_SMALL_NOISE_AREA_RATIO = 0.030
MAX_SMALL_NOISE_AREA_PX = 24

# Guard against a weak rule turning diffuse background haze into positives.
OVERCALL_MAX_FRAC = 0.58
OVERCALL_RESCUE_MAX_FRAC = 0.35

# Extra cleanup for images like 00101: one/few true saturated droplets plus many
# tiny fluorescence speckles that passed weak-positive rescue.
DOMINANT_CLEANUP_MIN_POS = 25
DOMINANT_CLEANUP_TOP_SCORE = 120.0
DOMINANT_CLEANUP_MEDIAN_FRAC = 0.22
DOMINANT_CLEANUP_P95_FRAC = 0.45
DOMINANT_CLEANUP_MAX_FULL_CORE_FRAC = 0.06
DOMINANT_CLEANUP_MIN_LOW_CORE_FRAC = 0.55
DOMINANT_KEEP_SCORE_FRAC = 0.78
DOMINANT_KEEP_CORE_RATIO = 0.70
DOMINANT_KEEP_AREA_RATIO = 0.65

# Strong positives can be very bright without matching the adaptive score split.
# Keep them if they are full droplet-scale, not tiny speckles.
BRIGHT_KEEP_CORE_RATIO = 0.45
BRIGHT_KEEP_AREA_RATIO = 0.55
BRIGHT_KEEP_GMAX = 220.0
BRIGHT_KEEP_P95 = 200.0
BRIGHT_KEEP_CIRC = 0.62
BRIGHT_KEEP_ASPECT = 0.70
BRIGHT_KEEP_MAX_OFFSET = 0.45

VIS_INTERVAL = 1


# ============================================================
# 2) Helpers
# ============================================================
def robust_std(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(1.4826 * mad, float(np.std(x)) * 0.35, 1.0)


def circle_mask(shape, cx, cy, radius):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), max(1, int(radius)), 255, -1)
    return mask


def ring_mask(shape, cx, cy, r0, r1):
    mask = np.zeros(shape, np.uint8)
    if r1 > 0:
        cv2.circle(mask, (int(cx), int(cy)), max(1, int(r1)), 255, -1)
    if r0 > 0:
        cv2.circle(mask, (int(cx), int(cy)), max(1, int(r0)), 0, -1)
    return mask


def regular_grid(shape):
    h, w = shape
    step_y = h / GRID_N
    step_x = w / GRID_N
    r = int(min(step_x, step_y) * FALLBACK_FRAC)
    wcx, wcy, wcr = [], [], []
    for ri in range(GRID_N):
        for ci in range(GRID_N):
            wcx.append(int((ci + 0.5) * step_x))
            wcy.append(int((ri + 0.5) * step_y))
            wcr.append(max(5, r))
    return wcx, wcy, wcr


def read_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)


def measure_largest_blob(mask, pcx, pcy, cr, inner_count):
    mask = (mask > 0).astype(np.uint8)
    n_lab, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n_lab <= 1:
        return {
            "area_px": 0.0,
            "area_ratio": 0.0,
            "eq_radius_frac": 0.0,
            "circularity": 0.0,
            "aspect": 0.0,
            "center_offset": 9.0,
            "component_count": 0,
        }, np.zeros_like(mask, dtype=np.uint8)

    comp_ids = [i for i in range(1, n_lab) if stats[i, cv2.CC_STAT_AREA] >= 3]
    if not comp_ids:
        return {
            "area_px": 0.0,
            "area_ratio": 0.0,
            "eq_radius_frac": 0.0,
            "circularity": 0.0,
            "aspect": 0.0,
            "center_offset": 9.0,
            "component_count": 0,
        }, np.zeros_like(mask, dtype=np.uint8)

    best = max(comp_ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    comp = (lab == best).astype(np.uint8)
    area_px = float(stats[best, cv2.CC_STAT_AREA])
    x = float(stats[best, cv2.CC_STAT_LEFT])
    y = float(stats[best, cv2.CC_STAT_TOP])
    bw = float(stats[best, cv2.CC_STAT_WIDTH])
    bh = float(stats[best, cv2.CC_STAT_HEIGHT])
    aspect = min(bw, bh) / max(bw, bh, 1.0)

    cnts, _ = cv2.findContours(comp * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        contour_area = max(float(cv2.contourArea(cnt)), 1.0)
        perim = cv2.arcLength(cnt, True)
        circularity = float(4.0 * np.pi * contour_area / (perim * perim + 1e-6))

    bx, by = cents[best]
    center_offset = float(np.sqrt((bx - pcx) ** 2 + (by - pcy) ** 2) / max(cr, 1))
    eq_radius_frac = float(np.sqrt(area_px / np.pi) / max(cr, 1))
    return {
        "area_px": area_px,
        "area_ratio": float(area_px / max(inner_count, 1)),
        "eq_radius_frac": eq_radius_frac,
        "circularity": circularity,
        "aspect": float(aspect),
        "center_offset": center_offset,
        "component_count": len(comp_ids),
    }, comp


# ============================================================
# 3) Grid detection
# ============================================================
def detect_grid(g):
    h, w = g.shape
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    blur = cv2.GaussianBlur(clahe.apply(g), (5, 5), 1.2)
    wc = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_P1,
        param2=HOUGH_P2,
        minRadius=HOUGH_MIN_R,
        maxRadius=HOUGH_MAX_R,
    )
    if wc is None:
        return regular_grid(g.shape)

    circles = np.squeeze(wc, 0)
    if circles.ndim == 1:
        circles = circles.reshape(1, -1)
    if len(circles) < GRID_N:
        return regular_grid(g.shape)

    km_x = KMeans(GRID_N, n_init=10, random_state=0).fit(circles[:, 0].reshape(-1, 1))
    km_y = KMeans(GRID_N, n_init=10, random_state=0).fit(circles[:, 1].reshape(-1, 1))
    col_c = np.sort(km_x.cluster_centers_.flatten())
    row_c = np.sort(km_y.cluster_centers_.flatten())
    pr = float(np.median(np.diff(row_c)))
    pc = float(np.median(np.diff(col_c)))
    fb = int(min(pr, pc) * FALLBACK_FRAC)

    rb = (
        [max(0, int(row_c[0] - pr / 2))]
        + [int((row_c[i] + row_c[i + 1]) / 2) for i in range(GRID_N - 1)]
        + [min(h - 1, int(row_c[-1] + pr / 2))]
    )
    cb = (
        [max(0, int(col_c[0] - pc / 2))]
        + [int((col_c[i] + col_c[i + 1]) / 2) for i in range(GRID_N - 1)]
        + [min(w - 1, int(col_c[-1] + pc / 2))]
    )

    cx_a, cy_a, cr_a = circles[:, 0], circles[:, 1], circles[:, 2]
    wcx, wcy, wcr = [], [], []
    for ri in range(GRID_N):
        for ci in range(GRID_N):
            gx = (cb[ci] + cb[ci + 1]) / 2
            gy = (rb[ri] + rb[ri + 1]) / 2
            d = np.sqrt((cx_a - gx) ** 2 + (cy_a - gy) ** 2)
            bi = int(np.argmin(d))
            if d[bi] < min(pr, pc) * 0.65:
                wcx.append(int(cx_a[bi]))
                wcy.append(int(cy_a[bi]))
                wcr.append(int(cr_a[bi]))
            else:
                wcx.append(int(gx))
                wcy.append(int(gy))
                wcr.append(max(5, fb))
    return wcx, wcy, wcr


# ============================================================
# 4) Feature extraction
# ============================================================
def extract_features(g, wcx, wcy, wcr):
    h, w = g.shape
    nf = float(np.percentile(g, 1))
    global_bg_std = robust_std(g.reshape(-1))
    rows = []

    for cx, cy, cr in zip(wcx, wcy, wcr):
        cr = max(int(cr), 5)
        pad = int(cr * BG_OUTER + 4)
        x1, x2 = max(0, cx - pad), min(w, cx + pad + 1)
        y1, y2 = max(0, cy - pad), min(h, cy + pad + 1)
        pg = g[y1:y2, x1:x2].astype(float)
        ph, pw = pg.shape
        pcx, pcy = cx - x1, cy - y1

        center_m = circle_mask((ph, pw), pcx, pcy, cr * CENTER_R)
        inner_m = circle_mask((ph, pw), pcx, pcy, cr * INNER_R)
        bg_m = ring_mask((ph, pw), pcx, pcy, cr * BG_INNER, cr * BG_OUTER)

        center_px = pg[center_m > 0]
        inner_px = pg[inner_m > 0]
        bg_px = pg[bg_m > 0]
        if len(bg_px) < 8:
            bg_px = pg.reshape(-1)

        center_mean = float(center_px.mean()) if len(center_px) else 0.0
        center_median = float(np.median(center_px)) if len(center_px) else 0.0
        inner_mean = float(inner_px.mean()) if len(inner_px) else 0.0
        inner_median = float(np.median(inner_px)) if len(inner_px) else 0.0
        inner_p90 = float(np.percentile(inner_px, 90)) if len(inner_px) else 0.0
        inner_p95 = float(np.percentile(inner_px, 95)) if len(inner_px) else 0.0
        inner_std = float(np.std(inner_px)) if len(inner_px) else 0.0
        bg_mean = float(bg_px.mean()) if len(bg_px) else 0.0
        bg_median = float(np.median(bg_px)) if len(bg_px) else nf
        bg_sigma = max(robust_std(bg_px), global_bg_std * 0.12, 1.0)
        gmax = float(inner_px.max()) if len(inner_px) else 0.0

        score = max(center_median, inner_median) - bg_median
        score_p90 = inner_p90 - bg_median
        robust_z = score / max(bg_sigma, 1.0)
        snr_center = center_mean / max(bg_mean, bg_median, nf + 1.0, 1.0)

        low_thr = max(bg_median + max(4.0, 1.7 * bg_sigma), nf + 4.0)
        core_thr = max(bg_median + max(8.0, 3.0 * bg_sigma), nf + 9.0)
        low_mask = ((pg > low_thr) & (inner_m > 0)).astype(np.uint8)
        core_mask = ((pg > core_thr) & (inner_m > 0)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        low_mask = cv2.morphologyEx(low_mask, cv2.MORPH_OPEN, kernel)
        core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel)

        inner_count = int((inner_m > 0).sum())
        low_blob, low_comp = measure_largest_blob(low_mask, pcx, pcy, cr, inner_count)
        core_blob, core_comp = measure_largest_blob(core_mask, pcx, pcy, cr, inner_count)

        low_px = pg[low_comp > 0]
        core_px = pg[core_comp > 0]
        low_blob_mean = float(low_px.mean()) if len(low_px) else 0.0
        low_blob_median = float(np.median(low_px)) if len(low_px) else 0.0
        core_blob_mean = float(core_px.mean()) if len(core_px) else 0.0
        core_blob_median = float(np.median(core_px)) if len(core_px) else 0.0

        fill_ratio = float((inner_px > core_thr).mean()) if len(inner_px) else 0.0
        support_ratio = float((inner_px > low_thr).mean()) if len(inner_px) else 0.0
        peak_ratio = (gmax - bg_median) / max(score_p90, score, 1.0)
        uniformity = inner_std / max(abs(score), 1.0)
        blob_score = max(low_blob_median, core_blob_median, inner_p90) - bg_median
        blob_z = blob_score / max(bg_sigma, 1.0)

        rows.append(
            {
                "snr_center": snr_center,
                "score": score,
                "score_p90": score_p90,
                "blob_score": blob_score,
                "blob_z": blob_z,
                "robust_z": robust_z,
                "inner_mean": inner_mean,
                "inner_median": inner_median,
                "inner_p95": inner_p95,
                "inner_std": inner_std,
                "bg_mean": bg_mean,
                "bg_median": bg_median,
                "bg_sigma": bg_sigma,
                "gmax": gmax,
                "low_thr": low_thr,
                "core_thr": core_thr,
                "low_blob_mean": low_blob_mean,
                "low_blob_median": low_blob_median,
                "core_blob_mean": core_blob_mean,
                "core_blob_median": core_blob_median,
                "fill_ratio": fill_ratio,
                "support_ratio": support_ratio,
                "peak_ratio": peak_ratio,
                "uniformity": uniformity,
                "area_ratio": low_blob["area_ratio"],
                "area_px": low_blob["area_px"],
                "eq_radius_frac": low_blob["eq_radius_frac"],
                "circularity": low_blob["circularity"],
                "aspect": low_blob["aspect"],
                "center_offset": low_blob["center_offset"],
                "component_count": low_blob["component_count"],
                "core_area_ratio": core_blob["area_ratio"],
                "core_area_px": core_blob["area_px"],
                "core_eq_radius_frac": core_blob["eq_radius_frac"],
                "core_circularity": core_blob["circularity"],
            }
        )

    return pd.DataFrame(rows), nf


# ============================================================
# 5) Adaptive thresholds
# ============================================================
def otsu_threshold(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return None
    lo, hi = np.percentile(values, [1, 99])
    if hi - lo < 1e-6:
        return None
    scaled = np.clip((values - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    t8, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(lo + (t8 / 255.0) * (hi - lo))


def gmm_threshold(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return None, 0.0
    try:
        x = values.reshape(-1, 1)
        gm = GaussianMixture(2, n_init=8, covariance_type="full", random_state=0).fit(x)
        means = gm.means_.flatten()
        sigmas = np.sqrt(gm.covariances_.flatten())
        weights = gm.weights_.flatten()
        order = np.argsort(means)
        ni, pi = int(order[0]), int(order[1])
        sep = float((means[pi] - means[ni]) / max(sigmas[ni] + sigmas[pi], 1e-3))
        if sep < GMM_SEP_MIN:
            return None, sep
        xs = np.linspace(means[ni], means[pi], 2048)
        d0 = weights[ni] * np.exp(-0.5 * ((xs - means[ni]) / sigmas[ni]) ** 2) / max(sigmas[ni], 1e-3)
        d1 = weights[pi] * np.exp(-0.5 * ((xs - means[pi]) / sigmas[pi]) ** 2) / max(sigmas[pi], 1e-3)
        cross = np.where(np.diff(np.sign(d1 - d0)))[0]
        if len(cross):
            return float(xs[cross[0]]), sep
        return float((means[ni] + means[pi]) / 2.0), sep
    except Exception:
        return None, 0.0


def score_thresholds(scores, valid_mask):
    scores = np.asarray(scores, dtype=float)
    valid_scores = scores[valid_mask & np.isfinite(scores)]
    if len(valid_scores) < 20:
        valid_scores = scores[np.isfinite(scores)]

    sorted_scores = np.sort(valid_scores)
    tail_n = max(20, int(len(sorted_scores) * NEG_TAIL_FRAC))
    neg_tail = sorted_scores[:tail_n]
    neg_mu = float(np.median(neg_tail))
    neg_std = robust_std(neg_tail)

    base_strong = max(
        SCORE_STRONG_MIN_DELTA,
        neg_mu + max(SCORE_STRONG_MIN_DELTA, SCORE_STRONG_SIGMA * neg_std),
    )
    base_rescue = max(
        SCORE_RESCUE_MIN_DELTA,
        neg_mu + max(SCORE_RESCUE_MIN_DELTA, SCORE_RESCUE_SIGMA * neg_std),
    )

    candidates = []
    otsu_t = otsu_threshold(valid_scores)
    if otsu_t is not None and otsu_t > base_rescue:
        candidates.append(("otsu", otsu_t))
    gmm_t, gmm_sep = gmm_threshold(valid_scores)
    if gmm_t is not None and gmm_t > base_rescue:
        candidates.append((f"gmm{gmm_sep:.2f}", gmm_t))

    if candidates:
        source, data_thr = min(candidates, key=lambda item: item[1])
        strong_thr = max(base_strong, float(data_thr))
    else:
        source = "robust"
        strong_thr = base_strong

    return {
        "strong_thr": float(strong_thr),
        "rescue_thr": float(base_rescue),
        "neg_mu": neg_mu,
        "neg_std": neg_std,
        "method": f"{source}:strong={strong_thr:.2f},rescue={base_rescue:.2f}",
    }


# ============================================================
# 6) Classification
# ============================================================
def classify(feat, nf):
    n = len(feat)
    labels = np.zeros(n, dtype=int)

    gmax = feat["gmax"].to_numpy(float)
    score = feat["score"].to_numpy(float)
    blob_score = feat["blob_score"].to_numpy(float)
    blob_z = feat["blob_z"].to_numpy(float)
    robust_z = feat["robust_z"].to_numpy(float)
    inner_median = feat["inner_median"].to_numpy(float)
    inner_p95 = feat["inner_p95"].to_numpy(float)
    fill = feat["fill_ratio"].to_numpy(float)
    support = feat["support_ratio"].to_numpy(float)
    area = feat["area_ratio"].to_numpy(float)
    area_px = feat["area_px"].to_numpy(float)
    eqr = feat["eq_radius_frac"].to_numpy(float)
    circ = feat["circularity"].to_numpy(float)
    aspect = feat["aspect"].to_numpy(float)
    offset = feat["center_offset"].to_numpy(float)
    snr = feat["snr_center"].to_numpy(float)
    core_area = feat["core_area_ratio"].to_numpy(float)
    core_area_px = feat["core_area_px"].to_numpy(float)

    bubble = (
        (gmax <= nf + BUBBLE_GMAX_DELTA)
        & (inner_p95 <= nf + BUBBLE_P95_DELTA)
        & (inner_median <= nf + BUBBLE_MEDIAN_DELTA)
    )
    valid = ~bubble
    th = score_thresholds(blob_score, valid)

    has_blob = area_px > 0
    tiny_noise = has_blob & (
        (area <= MAX_SMALL_NOISE_AREA_RATIO)
        | (area_px <= MAX_SMALL_NOISE_AREA_PX)
        | (eqr < MIN_LOW_EQ_RADIUS_FRAC_RESCUE)
    )
    irregular_noise = (
        has_blob
        & ((circ < 0.16) | (aspect < 0.28) | (offset > 0.90))
    )
    noise_like = valid & (tiny_noise | irregular_noise)

    shape_strong = (
        (area_px >= MIN_LOW_AREA_PX_STRONG)
        & (area >= MIN_LOW_AREA_RATIO_STRONG)
        & (eqr >= MIN_LOW_EQ_RADIUS_FRAC_STRONG)
        & (circ >= MIN_CIRC_STRONG)
        & (aspect >= MIN_ASPECT_STRONG)
        & (offset <= MAX_CENTER_OFFSET_STRONG)
    )

    shape_rescue = (
        (area_px >= MIN_LOW_AREA_PX_RESCUE)
        & (area >= MIN_LOW_AREA_RATIO_RESCUE)
        & (eqr >= MIN_LOW_EQ_RADIUS_FRAC_RESCUE)
        & (circ >= MIN_CIRC_RESCUE)
        & (aspect >= MIN_ASPECT_RESCUE)
        & (offset <= MAX_CENTER_OFFSET_RESCUE)
    )

    signal_strong = (
        (blob_score >= th["strong_thr"])
        | ((blob_z >= 4.0) & (core_area >= MIN_CORE_AREA_RATIO_STRONG))
        | ((snr >= 1.35) & (core_area >= MIN_CORE_AREA_RATIO_STRONG))
    )
    signal_rescue = (
        (blob_score >= th["rescue_thr"])
        | ((blob_z >= 2.0) & (support >= 0.10))
        | ((robust_z >= 2.2) & (score >= SCORE_RESCUE_MIN_DELTA))
    )

    strong_pos = valid & shape_strong & signal_strong
    rescued = valid & ~strong_pos & shape_rescue & signal_rescue

    # Avoid a weak rescue wave turning background haze into hundreds of positives.
    if rescued.mean() > OVERCALL_RESCUE_MAX_FRAC:
        rescued &= (blob_score >= np.percentile(blob_score[valid], 75)) & (
            core_area >= MIN_CORE_AREA_RATIO_STRONG
        )

    labels[strong_pos | rescued] = 1
    labels[bubble] = 2

    bright_full_droplet = valid & (
        (core_area >= BRIGHT_KEEP_CORE_RATIO)
        & (area >= BRIGHT_KEEP_AREA_RATIO)
        & (gmax >= BRIGHT_KEEP_GMAX)
        & (inner_p95 >= BRIGHT_KEEP_P95)
        & (circ >= BRIGHT_KEEP_CIRC)
        & (aspect >= BRIGHT_KEEP_ASPECT)
        & (offset <= BRIGHT_KEEP_MAX_OFFSET)
    )
    labels[bright_full_droplet] = 1
    rescued[bright_full_droplet] = False
    noise_like[bright_full_droplet] = False

    pos_frac = float((labels == 1).mean())
    if pos_frac > OVERCALL_MAX_FRAC:
        high_conf = bright_full_droplet | (valid & shape_strong & (
            (blob_score >= max(th["strong_thr"], np.percentile(blob_score[valid], 84)))
            | ((blob_z >= 5.0) & (core_area >= MIN_CORE_AREA_RATIO_STRONG))
        ))
        labels[:] = 0
        labels[high_conf] = 1
        labels[bubble] = 2
        th["method"] += f"+overcall_guard({pos_frac:.2f})"
        rescued = np.zeros(n, dtype=bool)
    elif rescued.any():
        th["method"] += f"+rescue({int(rescued.sum())})"

    # Speckles are not a fourth class; they remain negative.
    labels[(labels == 1) & noise_like & (core_area_px == 0)] = 0

    pos_now = labels == 1
    if pos_now.sum() >= DOMINANT_CLEANUP_MIN_POS:
        pos_scores = blob_score[pos_now]
        top_score = float(pos_scores.max())
        median_score = float(np.median(pos_scores))
        p95_score = float(np.percentile(pos_scores, 95))
        full_core_count = int(
            (
                pos_now
                & (core_area >= DOMINANT_KEEP_CORE_RATIO)
                & (area >= DOMINANT_KEEP_AREA_RATIO)
            ).sum()
        )
        low_core_frac = float(((pos_now) & (core_area < 0.15)).sum() / max(pos_now.sum(), 1))
        dominant_keep = (pos_now & bright_full_droplet) | (pos_now & (
            (blob_score >= max(DOMINANT_CLEANUP_TOP_SCORE, top_score * DOMINANT_KEEP_SCORE_FRAC))
            & (core_area >= DOMINANT_KEEP_CORE_RATIO)
            & (area >= DOMINANT_KEEP_AREA_RATIO)
            & (circ >= MIN_CIRC_STRONG)
            & (aspect >= MIN_ASPECT_STRONG)
            & (offset <= MAX_CENTER_OFFSET_STRONG)
        ))
        if (
            top_score >= DOMINANT_CLEANUP_TOP_SCORE
            and median_score <= top_score * DOMINANT_CLEANUP_MEDIAN_FRAC
            and p95_score <= top_score * DOMINANT_CLEANUP_P95_FRAC
            and full_core_count <= max(4, int(pos_now.sum() * DOMINANT_CLEANUP_MAX_FULL_CORE_FRAC))
            and low_core_frac >= DOMINANT_CLEANUP_MIN_LOW_CORE_FRAC
            and 1 <= dominant_keep.sum() <= max(3, int(pos_now.sum() * 0.08))
        ):
            removed = pos_now & ~dominant_keep
            labels[removed] = 0
            rescued[removed] = False
            noise_like |= removed
            th["method"] += f"+speckle_cleanup({int(removed.sum())})"

    return labels, th, noise_like, rescued


# ============================================================
# 7) Visualization
# ============================================================
C_POS = (50, 205, 80)
C_NEG = (65, 130, 220)
C_BUB = (215, 55, 55)
C_GRID = (180, 80, 220)
LBL_C = {1: C_POS, 0: C_NEG, 2: C_BUB}


def make_overlay_vis(img_rgb, wcx, wcy, wcr, labels, rescued, th, nf, img_id=""):
    h, w = img_rgb.shape[:2]
    out = img_rgb.copy()

    wcx_arr = np.array(wcx)
    wcy_arr = np.array(wcy)
    km_x = KMeans(GRID_N, n_init=10, random_state=0).fit(wcx_arr.reshape(-1, 1))
    km_y = KMeans(GRID_N, n_init=10, random_state=0).fit(wcy_arr.reshape(-1, 1))
    col_c = np.sort(km_x.cluster_centers_.flatten())
    row_c = np.sort(km_y.cluster_centers_.flatten())
    pr = float(np.median(np.diff(row_c)))
    pc = float(np.median(np.diff(col_c)))
    h_lines = (
        [max(0, int(row_c[0] - pr / 2))]
        + [int((row_c[i] + row_c[i + 1]) / 2) for i in range(GRID_N - 1)]
        + [min(h - 1, int(row_c[-1] + pr / 2))]
    )
    v_lines = (
        [max(0, int(col_c[0] - pc / 2))]
        + [int((col_c[i] + col_c[i + 1]) / 2) for i in range(GRID_N - 1)]
        + [min(w - 1, int(col_c[-1] + pc / 2))]
    )
    for y in h_lines:
        cv2.line(out, (v_lines[0], y), (v_lines[-1], y), C_GRID, 2, cv2.LINE_AA)
    for x in v_lines:
        cv2.line(out, (x, h_lines[0]), (x, h_lines[-1]), C_GRID, 2, cv2.LINE_AA)

    draw_r = int(np.median(wcr) * 0.82)
    cell_cx = [int((v_lines[ci] + v_lines[ci + 1]) / 2) for ci in range(GRID_N)]
    cell_cy = [int((h_lines[ri] + h_lines[ri + 1]) / 2) for ri in range(GRID_N)]

    for idx, lbl in enumerate(labels):
        ri, ci = idx // GRID_N, idx % GRID_N
        cx, cy = cell_cx[ci], cell_cy[ri]
        color = LBL_C[int(lbl)]
        if lbl == 1:
            overlay = out.copy()
            cv2.circle(overlay, (cx, cy), draw_r, color, -1)
            cv2.addWeighted(out, 0.70, overlay, 0.30, 0, out)
            cv2.circle(out, (cx, cy), draw_r, color, 3)
        elif lbl == 0:
            cv2.circle(out, (cx, cy), draw_r, color, 2)
        else:
            cv2.circle(out, (cx, cy), draw_r, color, 3)

    bar_h = 78
    bar = np.full((bar_h, w, 3), (22, 22, 30), dtype=np.uint8)
    np_ = int((labels == 1).sum())
    nn = int((labels == 0).sum())
    nb = int((labels == 2).sum())
    nr = int(rescued.sum())

    cv2.putText(
        bar,
        f"ddPCR Fluorescence v13 | {img_id}",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (220, 220, 225),
        1,
        cv2.LINE_AA,
    )
    info = f"{th['method']}  nf={nf:.0f}"
    (iw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(bar, info, (max(14, w - iw - 14), 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (145, 145, 155), 1, cv2.LINE_AA)

    legend = [
        (C_POS, f"Positive {np_:3d}"),
        (C_NEG, f"Negative {nn:3d}"),
        (C_BUB, f"Bubble {nb:2d}"),
    ]
    for i, (col, txt) in enumerate(legend):
        xo = 14 + i * 220
        cv2.circle(bar, (xo + 10, 57), 9, col, -1)
        cv2.putText(bar, txt, (xo + 26, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 225), 1, cv2.LINE_AA)
    cv2.line(bar, (0, bar_h - 1), (w, bar_h - 1), (55, 55, 70), 1)
    return np.vstack([bar, out])


# ============================================================
# 8) Main loop
# ============================================================
def process_all():
    for sub in ("state", "vis"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    fluor_paths = sorted(glob.glob(os.path.join(fluor_dir, "*.tif")))
    if max_images:
        fluor_paths = fluor_paths[:max_images]
    n_total = len(fluor_paths)

    vis_indices = set()
    for b in range(0, n_total, VIS_INTERVAL):
        vis_indices.add(b + random.randint(0, min(VIS_INTERVAL - 1, n_total - b - 1)))

    summary = []
    print(f"Processing {n_total} images -> {output_dir}")
    print("-" * 88)

    for idx, fpath in enumerate(fluor_paths):
        img_id = os.path.splitext(os.path.basename(fpath))[0]
        try:
            img_np = read_image_rgb(fpath)
            g = img_np[:, :, 1]

            wcx, wcy, wcr = detect_grid(g)
            feat, nf = extract_features(g, wcx, wcy, wcr)
            labels, th, noise_like, rescued = classify(feat, nf)

            np_ = int((labels == 1).sum())
            nn = int((labels == 0).sum())
            nb = int((labels == 2).sum())
            assert np_ + nn + nb == 400, f"Label count error: {np_}+{nn}+{nb}"

            rows = []
            for j in range(400):
                row = {
                    "well_idx": j,
                    "row": chr(65 + j // 20),
                    "col": j % 20 + 1,
                    "cx": wcx[j],
                    "cy": wcy[j],
                    "cr": wcr[j],
                    "nf": round(float(nf), 3),
                    "threshold_strong": round(float(th["strong_thr"]), 5),
                    "threshold_rescue": round(float(th["rescue_thr"]), 5),
                    "neg_mu": round(float(th["neg_mu"]), 5),
                    "neg_std": round(float(th["neg_std"]), 5),
                    "method": th["method"],
                    "noise_like": int(noise_like[j]),
                    "rescued": int(rescued[j]),
                    "label": int(labels[j]),
                }
                for col in feat.columns:
                    row[col] = round(float(feat.iloc[j][col]), 5)
                rows.append(row)

            pd.DataFrame(rows).to_csv(
                os.path.join(output_dir, "state", f"{img_id}_fluor_labels.csv"),
                index=False,
            )

            if idx in vis_indices:
                vis = make_overlay_vis(img_np, wcx, wcy, wcr, labels, rescued, th, nf, img_id=img_id)
                cv2.imwrite(
                    os.path.join(output_dir, "vis", f"{img_id}_grid.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                )

            summary.append(
                {
                    "img_id": img_id,
                    "n_pos": np_,
                    "n_neg": nn,
                    "n_bub": nb,
                    "n_rescued": int(rescued.sum()),
                    "n_noise_like": int(noise_like.sum()),
                    "threshold_strong": round(float(th["strong_thr"]), 4),
                    "threshold_rescue": round(float(th["rescue_thr"]), 4),
                    "method": th["method"],
                    "noise_floor": round(float(nf), 1),
                    "score_min": round(float(feat["score"].min()), 4),
                    "score_max": round(float(feat["score"].max()), 4),
                }
            )

            if idx % 100 == 0 or idx < 5:
                print(
                    f"[{idx + 1:5d}/{n_total}] {img_id:20s} "
                    f"P={np_:3d} N={nn:3d} B={nb:2d} R={int(rescued.sum()):2d} "
                    f"[{th['method']}]"
                )

        except Exception as e:
            import traceback

            print(f"[ERR] {img_id}: {e}")
            traceback.print_exc()
            summary.append({"img_id": img_id, "error": str(e)})

    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "fluor_batch_summary.csv"), index=False)
    print(f"\nDone. Summary -> {output_dir}/fluor_batch_summary.csv")


if __name__ == "__main__":
    process_all()
