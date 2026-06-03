"""
ddPCR 明场图像自动打标签脚本 v3
基于原有版本改进，主要修复：
  1. 低对比度/偏暗图像 Hough 失败 → CLAHE 预处理 + 多参数自适应回退
  2. Hough 圆数量不足 → 投影法网格作为兜底 fallback
  3. 新增气泡(Bubble)类别与合并液滴区分（原来都归Artifact）
  4. 透视变换鲁棒化（角点缺失时改用仿射变换）
  5. 失败原因详细记录，便于后续排查

标签：
  0 = Droplet  正常液滴
  1 = Artifact 合并液滴（多液滴在同一微井）
  2 = Empty    空/无效微井
  3 = Bubble   气泡（新增）

输出与原版相同：geom CSV / state CSV / debug CSV / overlay PNG / AE patches
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import cv2
import tifffile
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=UserWarning)

# ===================== 0) 路径配置 =====================
dataset_root  = r"E:\qiujiuer_data\pycharm_file\patchs\dataset_bright"
images_dir    = os.path.join(dataset_root, "images")
geom_dir      = os.path.join(dataset_root, "labels", "geom")
state_dir     = os.path.join(dataset_root, "labels", "state")
vis_dir       = os.path.join(dataset_root, "labels", "vis")
debug_dir     = os.path.join(dataset_root, "labels", "debug")
ae_patch_root = os.path.join(dataset_root, "labels", "AE_patches")

for d in [geom_dir, state_dir, vis_dir, debug_dir, ae_patch_root]:
    os.makedirs(d, exist_ok=True)

max_images = None               # 调试时设 100；正式跑设 None
save_overlay_every  = 50
save_first_overlay  = True
save_overlay_if_AE  = True      # 有 A/E/B 时强制保存 overlay

# ===================== 1) 全局参数 =====================
N_ROWS, N_COLS = 20, 20
patch_size = 42

well_radius_um       = 50.0
droplet_min_radius_um = 28.0

# ── 微井 Hough（主参数）──
H_PRIMARY = dict(dp=1.2, minDist=35, param1=120, param2=20,
                 minRadius=10, maxRadius=22)

# ── Hough 多级回退（依次宽松）──
H_FALLBACKS = [
    dict(dp=1.2, minDist=33, param1=100, param2=17, minRadius=8,  maxRadius=24),
    dict(dp=1.2, minDist=30, param1=80,  param2=14, minRadius=8,  maxRadius=25),
    dict(dp=1.5, minDist=30, param1=70,  param2=12, minRadius=7,  maxRadius=26),
]
HOUGH_MIN_ACCEPT   = 300   # 少于此数量视为检测失败，尝试下一级参数
HOUGH_TARGET       = 400   # 理想数量

# ── 标签 ──
LABELS = {
    0: ("Droplet",  ""),
    1: ("Artifact", "A"),
    2: ("Empty",    "E"),
    3: ("Bubble",   "B"),
}

# ── Empty/Invalid 判断 ──
E_mean_th        = 45.0
E_center_mean_th = 35.0
E_std_th         = 10.0
E_edge_ratio_th  = 0.010

# ── 气泡判断（与合并液滴区分）──
# 气泡特征：整体偏暗 + 高对比度（中心亮斑+暗环）
BUBBLE_MEAN_MAX = 130.0
BUBBLE_STD_MIN  = 50.0

# ── Artifact（合并液滴，核心连通域>=2）──
center_mask_factor = 0.75
core_factor        = 0.85
min_core_area      = 10

# ── 可视化 ──
purple      = (180, 0, 180)
blue        = (255, 0, 0)
orange      = (0, 140, 255)
text_white  = (255, 255, 255)
text_black  = (0, 0, 0)
circle_r_factor_of_pitch = 0.30
circle_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX


# ===================== 2) 工具函数 =====================

def to_uint8(x):
    if x.dtype == np.uint8:
        return x
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def read_tif_bgr(path):
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        return cv2.cvtColor(to_uint8(arr), cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(to_uint8(arr), cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(to_uint8(arr[..., :3]), cv2.COLOR_RGB2BGR)


def enhance_contrast(gray8):
    """
    CLAHE 自适应对比度增强，解决低对比度/偏暗图像导致 Hough 失败的问题。
    对正常图像几乎无影响，对暗图效果显著。
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray8)


def try_hough_circles(blur_gray):
    """
    多级参数尝试 Hough 圆检测。
    优先原始参数；若检测数量不足，依次回退到宽松参数。
    返回：(circles_array | None, 使用的参数级别 0=primary, 1/2/3=fallback)
    """
    for level, params in enumerate([H_PRIMARY] + H_FALLBACKS):
        wc = cv2.HoughCircles(blur_gray, cv2.HOUGH_GRADIENT, **params)
        if wc is not None:
            wc = np.squeeze(wc, axis=0)
            if wc.ndim == 1:
                wc = wc.reshape(1, 3)
            if len(wc) >= HOUGH_MIN_ACCEPT:
                return wc, level
    return None, -1


def fallback_grid_from_projection(gray8, n_rows=N_ROWS, n_cols=N_COLS):
    """
    当 Hough 完全失败时，用行列投影法（找谷值）生成网格中心坐标。
    这是一个更鲁棒的兜底方案，不依赖圆形检测。
    返回：centers (N_ROWS, N_COLS, 2) 或 None（如果投影也失败）
    """
    h, w = gray8.shape
    row_proj = gray8.mean(axis=1)
    col_proj = gray8.mean(axis=0)

    valley_dist = max(25, int(min(h, w) / (max(n_rows, n_cols) * 1.5)))
    valleys_row, _ = find_peaks(-row_proj, distance=valley_dist, prominence=2)
    valleys_col, _ = find_peaks(-col_proj, distance=valley_dist, prominence=2)

    if len(valleys_row) < 3 or len(valleys_col) < 3:
        return None, None, None

    def fit_uniform_bounds(valleys, img_size, n_cells):
        idx = np.arange(len(valleys))
        coeffs = np.polyfit(idx, valleys, 1)
        step = coeffs[0]
        first_v = coeffs[1]
        start = first_v - step / 2
        bounds = [int(round(start + i * step)) for i in range(n_cells + 1)]
        bounds[0]  = max(0, bounds[0])
        bounds[-1] = min(img_size, bounds[-1])
        return bounds, step

    row_bounds, step_r = fit_uniform_bounds(valleys_row, h, n_rows)
    col_bounds, step_c = fit_uniform_bounds(valleys_col, w, n_cols)

    centers = np.zeros((n_rows, n_cols, 2), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            cy = (row_bounds[r] + row_bounds[r + 1]) / 2
            cx = (col_bounds[c] + col_bounds[c + 1]) / 2
            centers[r, c] = [cx, cy]

    pitch = float((step_r + step_c) / 2)
    return centers, pitch, None  # 投影法无法估算半径，返回 None


def assign_grid_20x20(pts_xy):
    """KMeans 将检测到的圆分配到 20×20 网格。（原版逻辑保持不变）"""
    y = pts_xy[:, 1].reshape(-1, 1)
    km = KMeans(n_clusters=N_ROWS, n_init=10, random_state=0).fit(y)
    row_labels  = km.labels_
    row_centers = km.cluster_centers_.flatten()

    order = np.argsort(row_centers)
    label_to_row = {int(old): int(new) for new, old in enumerate(order)}
    row_idx = np.array([label_to_row[int(l)] for l in row_labels])

    grid = [[None] * N_COLS for _ in range(N_ROWS)]
    used = np.zeros(len(pts_xy), dtype=bool)

    for r in range(N_ROWS):
        idxs = np.where(row_idx == r)[0]
        cy   = row_centers[order[r]]
        if len(idxs) > N_COLS:
            idxs = idxs[np.argsort(np.abs(pts_xy[idxs, 1] - cy))[:N_COLS]]
        idxs = idxs[np.argsort(pts_xy[idxs, 0])]
        for c, idx in enumerate(idxs[:N_COLS]):
            grid[r][c] = int(idx)
            used[idx]  = True

    remaining = np.where(~used)[0]
    for r in range(N_ROWS):
        filled = [i for i in grid[r] if i is not None]
        if len(filled) == N_COLS:
            continue
        cy   = row_centers[order[r]]
        cand = remaining[np.argsort(np.abs(pts_xy[remaining, 1] - cy))]
        take = cand[:N_COLS - len(filled)]
        merged = np.array(filled + take.tolist(), dtype=int)
        merged = merged[np.argsort(pts_xy[merged, 0])][:N_COLS]
        for c in range(N_COLS):
            grid[r][c] = int(merged[c])
        used[merged]  = True
        remaining     = np.where(~used)[0]

    centers = np.zeros((N_ROWS, N_COLS, 2), dtype=np.float32)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            centers[r, c] = pts_xy[grid[r][c]]
    return centers


def crop_with_padding(arr, x0, y0, w, h, fill=0):
    H, W = arr.shape[:2]
    out = np.full((h, w) if arr.ndim == 2 else (h, w, arr.shape[2]),
                  fill, dtype=arr.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + w), min(H, y0 + h)
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return out


def build_global_distance_transform(gray8):
    blur  = cv2.GaussianBlur(gray8, (5, 5), 1.2)
    edges = cv2.Canny(blur, 60, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    non_edge = (edges == 0).astype(np.uint8) * 255
    return cv2.distanceTransform(non_edge, cv2.DIST_L2, 5).astype(np.float32)


def empty_score(patch_gray8):
    """判断微井是否为 Empty/Invalid（逻辑与原版一致）"""
    blur       = cv2.GaussianBlur(patch_gray8, (5, 5), 1.2)
    mean       = float(patch_gray8.mean())
    std        = float(patch_gray8.std())
    edges      = cv2.Canny(blur, 60, 150)
    edge_ratio = float((edges > 0).mean())

    mask = np.zeros_like(patch_gray8, dtype=np.uint8)
    rr   = int(round((patch_size / 2) * 0.45))
    cv2.circle(mask, (patch_size // 2, patch_size // 2), rr, 1, -1)
    center_mean = float(patch_gray8[mask == 1].mean()) if np.any(mask == 1) else mean

    is_E = ((mean <= E_mean_th)
            or (center_mean <= E_center_mean_th)
            or (std <= E_std_th and edge_ratio <= E_edge_ratio_th))
    return is_E, {"mean": mean, "std": std,
                  "edge_ratio": edge_ratio, "center_mean": center_mean}


def bubble_score(patch_gray8):
    """
    判断是否为气泡。
    气泡特征：整体偏暗（mean < 130）+ 高对比度（std > 50）
    这与正常液滴（mean>150, std<35）和空微井（mean<45）均不同。
    """
    mean = float(patch_gray8.mean())
    std  = float(patch_gray8.std())
    is_B = (mean < BUBBLE_MEAN_MAX) and (std > BUBBLE_STD_MIN)
    return is_B


def count_core_components(dist_patch, r_min_px):
    """统计核心连通域数量，>=2 表示合并液滴（原版逻辑保持不变）"""
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    rr   = int(round((patch_size / 2.0) * center_mask_factor))
    cv2.circle(mask, (patch_size // 2, patch_size // 2), rr, 1, -1)

    d    = dist_patch * mask.astype(np.float32)
    thr  = float(r_min_px) * core_factor
    core = (d >= thr).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8), iterations=1)

    num, _, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([])
    return int(np.sum(areas >= min_core_area)) if areas.size else 0


# ===================== 3) 单张处理函数 =====================

def process_one_tif(tif_path, save_overlay_requested: bool):
    image_id   = os.path.splitext(os.path.basename(tif_path))[0]
    out_geom   = os.path.join(geom_dir,  f"{image_id}_well_geometry.csv")
    out_labels = os.path.join(state_dir, f"{image_id}_well_labels.csv")
    out_debug  = os.path.join(debug_dir, f"{image_id}_well_debug.csv")
    out_overlay = os.path.join(vis_dir,  f"{image_id}_overlay.png")
    this_ae_dir = os.path.join(ae_patch_root, image_id)
    os.makedirs(this_ae_dir, exist_ok=True)

    img_bgr  = read_tif_bgr(tif_path)
    gray8    = to_uint8(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    # ── 改进点1：CLAHE 增强后再做 Hough（解决暗图失败问题）──
    gray_enhanced = enhance_contrast(gray8)
    blur_enhanced = cv2.GaussianBlur(gray_enhanced, (5, 5), 1.2)

    well_circles, hough_level = try_hough_circles(blur_enhanced)
    grid_method = "hough"
    well_r_px   = None

    # ── 改进点2：Hough 失败时用投影法兜底 ──
    if well_circles is None or len(well_circles) < HOUGH_MIN_ACCEPT:
        centers_proj, pitch_proj, _ = fallback_grid_from_projection(gray_enhanced)
        if centers_proj is None:
            raise RuntimeError(
                f"Both Hough (found {len(well_circles) if well_circles is not None else 0}) "
                f"and projection grid failed for {image_id}"
            )
        centers     = centers_proj
        pitch       = pitch_proj
        margin      = int(round(pitch * 0.6))
        grid_method = "projection_fallback"
        # 投影法无半径信息，用理论值估算
        well_r_px   = (pitch / 2) * 0.65

    else:
        pts   = well_circles[:, :2].astype(np.float32)
        rads  = well_circles[:, 2].astype(np.float32)
        well_r_px = float(np.median(rads))

        # ── 改进点3：KMeans 分配失败时的容错 ──
        try:
            centers = assign_grid_20x20(pts)
        except Exception as e:
            # KMeans 偶尔在某些极端分布下会失败，回退到投影法
            centers_proj, pitch_proj, _ = fallback_grid_from_projection(gray_enhanced)
            if centers_proj is None:
                raise RuntimeError(f"KMeans failed ({e}) and projection fallback also failed")
            centers     = centers_proj
            grid_method = "projection_fallback(kmeans_err)"

        pitch_x = float(np.median(np.diff(np.sort(centers[N_ROWS//2, :, 0]))))
        pitch_y = float(np.median(np.diff(np.sort(centers[:, N_COLS//2, 1]))))
        pitch   = float((pitch_x + pitch_y) / 2.0)
        margin  = int(round(pitch * 0.6))

    um_per_px = well_radius_um / well_r_px
    r_min_px  = max(3.0, droplet_min_radius_um / um_per_px)

    # ── 透视变换（对齐网格）──
    out_w = int(round(margin * 2 + pitch * (N_COLS - 1)))
    out_h = int(round(margin * 2 + pitch * (N_ROWS - 1)))

    src = np.array([
        centers[0, 0], centers[0, N_COLS - 1],
        centers[N_ROWS - 1, 0], centers[N_ROWS - 1, N_COLS - 1]
    ], dtype=np.float32)
    dst = np.array([
        [margin, margin],
        [margin + pitch * (N_COLS - 1), margin],
        [margin, margin + pitch * (N_ROWS - 1)],
        [margin + pitch * (N_COLS - 1), margin + pitch * (N_ROWS - 1)]
    ], dtype=np.float32)

    # ── 改进点4：透视变换异常时改用仿射变换（3点）──
    try:
        Hmat    = cv2.getPerspectiveTransform(src, dst)
        aligned = cv2.warpPerspective(img_bgr, Hmat, (out_w, out_h),
                                       flags=cv2.INTER_LINEAR)
    except cv2.error:
        # 角点坐标异常，用前3点做仿射变换
        Amat    = cv2.getAffineTransform(src[:3], dst[:3])
        aligned = cv2.warpAffine(img_bgr, Amat, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR)

    aligned_gray8 = to_uint8(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
    dist_global   = build_global_distance_transform(aligned_gray8)

    overlay    = aligned.copy() if save_overlay_requested else None
    half       = patch_size // 2
    circle_r   = max(3, int(round(pitch * circle_r_factor_of_pitch)))
    font_scale = max(0.35, float(pitch) / 120.0)
    font_th    = 2

    geom_rows, label_rows, debug_rows = [], [], []
    A_wells, E_wells, B_wells = [], [], []

    for r in range(N_ROWS):
        for c in range(N_COLS):
            cx  = int(round(margin + pitch * c))
            cy  = int(round(margin + pitch * r))
            wid = f"r{r:02d}c{c:02d}"

            if overlay is not None:
                cv2.rectangle(overlay,
                               (cx - half, cy - half),
                               (cx + half, cy + half),
                               purple, 2)

            x0, y0    = cx - half, cy - half
            patch_g   = crop_with_padding(aligned_gray8, x0, y0,
                                           patch_size, patch_size, fill=0)
            patch_d   = crop_with_padding(dist_global,   x0, y0,
                                           patch_size, patch_size, fill=0.0)

            is_E, E_feats = empty_score(patch_g)
            core_cnt = 0

            if is_E:
                # ── Empty/Invalid ──
                label_id = 2
                label_name, letter = LABELS[label_id]
                note = "empty"
                E_wells.append(wid)

            else:
                # ── 改进点5：气泡检测（在合并液滴判断之前）──
                is_B = bubble_score(patch_g)
                if is_B:
                    label_id = 3
                    label_name, letter = LABELS[label_id]
                    note = "bubble"
                    B_wells.append(wid)

                else:
                    core_cnt = count_core_components(patch_d, r_min_px)
                    if core_cnt >= 2:
                        # ── 合并液滴 ──
                        label_id = 1
                        label_name, letter = LABELS[label_id]
                        note = "merged"
                        A_wells.append(wid)
                    else:
                        # ── 正常液滴 ──
                        label_id = 0
                        label_name, _ = LABELS[label_id]
                        letter = ""
                        note   = ""

            # 可视化
            if overlay is not None:
                if label_id == 0:
                    cv2.circle(overlay, (cx, cy), circle_r, blue, circle_thickness)
                elif label_id == 3:
                    # 气泡用橙色圆圈
                    cv2.circle(overlay, (cx, cy), circle_r, orange, circle_thickness)
                    (tw, thh), _ = cv2.getTextSize(letter, font, font_scale, font_th)
                    org = (cx - tw // 2, cy + thh // 2)
                    cv2.putText(overlay, letter, org, font, font_scale,
                                text_black, font_th + 2, cv2.LINE_AA)
                    cv2.putText(overlay, letter, org, font, font_scale,
                                text_white, font_th, cv2.LINE_AA)
                else:
                    (tw, thh), _ = cv2.getTextSize(letter, font, font_scale, font_th)
                    org = (cx - tw // 2, cy + thh // 2)
                    cv2.putText(overlay, letter, org, font, font_scale,
                                text_black, font_th + 2, cv2.LINE_AA)
                    cv2.putText(overlay, letter, org, font, font_scale,
                                text_white, font_th, cv2.LINE_AA)

            geom_rows.append({
                "well_id": wid, "row": r, "col": c,
                "center_x_px_aligned": float(cx),
                "center_y_px_aligned": float(cy),
                "pitch_px": float(pitch),
                "margin_px": int(margin),
                "patch_size": int(patch_size),
                "well_r_px_median": float(well_r_px),
                "um_per_px": float(um_per_px),
                "droplet_min_r_px": float(r_min_px),
                "grid_method": grid_method,
                "hough_level": int(hough_level),
            })

            label_rows.append({
                "well_id": wid,
                "label_id": int(label_id),
                "label_name": label_name,
                "review_flag": 0,
                "note": note,
            })

            debug_rows.append({
                "well_id": wid,
                "label_id": int(label_id),
                "core_cnt": int(core_cnt),
                **E_feats,
            })

    # 保存 CSV
    pd.DataFrame(geom_rows).to_csv(out_geom,   index=False, encoding="utf-8-sig")
    pd.DataFrame(label_rows).to_csv(out_labels, index=False, encoding="utf-8-sig")
    pd.DataFrame(debug_rows).to_csv(out_debug,  index=False, encoding="utf-8-sig")

    # 强制保存 overlay（当有 A/E/B 时）
    need_force = save_overlay_if_AE and (len(A_wells) + len(E_wells) + len(B_wells) > 0)
    if overlay is None and need_force:
        overlay = aligned.copy()
        for r in range(N_ROWS):
            for c in range(N_COLS):
                cx  = int(round(margin + pitch * c))
                cy  = int(round(margin + pitch * r))
                wid = f"r{r:02d}c{c:02d}"
                cv2.rectangle(overlay, (cx - half, cy - half),
                               (cx + half, cy + half), purple, 2)
                lid    = label_rows[r * N_COLS + c]["label_id"]
                letter = LABELS[lid][1]
                if lid == 0:
                    cv2.circle(overlay, (cx, cy), circle_r, blue, circle_thickness)
                elif lid == 3:
                    cv2.circle(overlay, (cx, cy), circle_r, orange, circle_thickness)
                    if letter:
                        (tw, thh), _ = cv2.getTextSize(letter, font, font_scale, font_th)
                        org = (cx - tw // 2, cy + thh // 2)
                        cv2.putText(overlay, letter, org, font, font_scale,
                                    text_black, font_th + 2, cv2.LINE_AA)
                        cv2.putText(overlay, letter, org, font, font_scale,
                                    text_white, font_th, cv2.LINE_AA)
                elif letter:
                    (tw, thh), _ = cv2.getTextSize(letter, font, font_scale, font_th)
                    org = (cx - tw // 2, cy + thh // 2)
                    cv2.putText(overlay, letter, org, font, font_scale,
                                text_black, font_th + 2, cv2.LINE_AA)
                    cv2.putText(overlay, letter, org, font, font_scale,
                                text_white, font_th, cv2.LINE_AA)

    if overlay is not None:
        cv2.imwrite(out_overlay, overlay)

    # 导出 A/E/B patches 供人工复核
    for wid in A_wells + E_wells + B_wells:
        rr = int(wid[1:3]); cc = int(wid[4:6])
        cx = int(round(margin + pitch * cc))
        cy = int(round(margin + pitch * rr))
        x0, y0  = cx - half, cy - half
        patch_c = crop_with_padding(aligned, x0, y0, patch_size, patch_size, fill=0)
        cv2.imwrite(os.path.join(this_ae_dir, f"{wid}.png"), patch_c)

    return image_id, len(A_wells), len(E_wells), len(B_wells), grid_method


# ===================== 4) 批量主程序 =====================

tif_files = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
if max_images is not None:
    tif_files = tif_files[:max_images]

print(f"[INFO] 共找到 {len(tif_files)} 张 tif 图像，开始处理...\n")

summary_rows = []
ok, fail = 0, 0
fallback_count = 0

for i, tif_path in enumerate(tif_files, 1):
    image_id = os.path.splitext(os.path.basename(tif_path))[0]
    save_overlay = (save_first_overlay and i == 1) or (save_overlay_every > 0 and i % save_overlay_every == 0)

    try:
        img_id, A_cnt, E_cnt, B_cnt, method = process_one_tif(
            tif_path, save_overlay_requested=save_overlay
        )
        used_fallback = "fallback" in method
        if used_fallback:
            fallback_count += 1

        summary_rows.append({
            "image_id": img_id,
            "A_count": A_cnt, "E_count": E_cnt, "B_count": B_cnt,
            "grid_method": method, "status": "ok"
        })
        ok += 1

        if (i <= 5) or (i % 100 == 0) or used_fallback:
            print(f"[{i:5d}/{len(tif_files)}] {img_id}: "
                  f"A={A_cnt} E={E_cnt} B={B_cnt} "
                  f"method={method} overlay={'Y' if save_overlay else 'n'}")

    except Exception as e:
        summary_rows.append({
            "image_id": image_id,
            "A_count": "", "E_count": "", "B_count": "",
            "grid_method": "fail", "status": f"fail: {e}"
        })
        fail += 1
        print(f"[{i:5d}/{len(tif_files)}] {image_id}: FAIL -> {e}")

summary_csv = os.path.join(dataset_root, "labels", "batch_summary.csv")
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

print("\n" + "=" * 50)
print(f"完成: OK={ok}  FAIL={fail}  使用投影法兜底={fallback_count}")
print(f"汇总报告: {summary_csv}")
