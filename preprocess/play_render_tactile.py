import os
import json
import numpy as np
import cv2
import trimesh
from matplotlib import cm
from pyrenderer import ManoRenderer


def render_tactile_patterns_to_mano(
    input_dir="tactile_pattern",
    output_dir="tactile_pattern_rendered",
    target_size=(960, 960),
):
    """
    Pipeline:

      For all .npy files in `input_dir`:

        1) Load each as/into a 16x16 tactile grid.
        2) PER-FILE normalize each 16x16 (min–max → [0, 1]).
        3) For valid nodes only:
             - compute 16x16 mean over all normalized grids on valid nodes
             - invalid nodes in mean_grid are 0
        4) Save mean_grid to tactile_mean_16x16.npy.
        5) For each file, subtract mean_grid from its normalized grid.
        6) Run MANO rendering pipeline on these mean-reduced grids.
    """

    # --- helpers copied / adapted from export_tactile_mano -------------------
    def _find_first_existing(cands):
        for p in cands:
            if p and os.path.exists(p):
                return p
        return None

    def _load_layout_json():
        layout_json = _find_first_existing([
            "handLayoutNewest_meshid.json",
            os.path.join(os.path.dirname(__file__), "data", "handLayoutNewest_meshid.json"),
            os.path.join(os.path.dirname(__file__), "scratch", "handLayoutNewest_meshid.json"),
            os.path.join(os.path.dirname(__file__), "handLayoutNewest_meshid.json"),
        ])
        if layout_json is None:
            raise FileNotFoundError("Missing handLayoutNewest_meshid.json")
        with open(layout_json, "r") as f:
            d = json.load(f)
        return d["positions"], set(d.get("erasedNodes", []))

    def _build_vertex_graph(verts, faces):
        V = len(verts)
        nbrs = [[] for _ in range(V)]
        dists = [[] for _ in range(V)]
        edges = set()
        for a, b, c in faces.astype(np.int64):
            edges.update({
                (min(a, b), max(a, b)),
                (min(b, c), max(b, c)),
                (min(c, a), max(c, a)),
            })
        for i, j in edges:
            dij = np.linalg.norm(verts[i] - verts[j])
            nbrs[i].append(j)
            dists[i].append(dij)
            nbrs[j].append(i)
            dists[j].append(dij)
        return nbrs, dists

    def _gaussian_smooth_vertex_signal(vals, nbrs, dists, sigma=0.005, iters=2):
        if sigma <= 0 or iters <= 0:
            return vals
        two_sig2 = 2.0 * (sigma * sigma)
        out = vals.astype(np.float32).copy()
        for _ in range(iters):
            new = out.copy()
            for i, (N, D) in enumerate(zip(nbrs, dists)):
                acc = out[i]
                w_sum = 1.0
                for j, dij in zip(N, D):
                    w = np.exp(-(dij * dij) / two_sig2)
                    acc += w * out[j]
                    w_sum += w
                new[i] = acc / max(w_sum, 1e-8)
            out = new
        return out

    def _render_pressure_mano(mano_vertices, mano_faces, renderer,
                              pressure16, layout, erased_nodes,
                              vmin, vmax, nbrs, dists):
        from collections import defaultdict

        # normalize pressure → [0,1] using global vmin/vmax
        norm = ((pressure16 - vmin) / max(vmax - vmin, 1e-6)).clip(0, 1)

        valid_nodes = {
            nid: {"mano_vid": layout[nid].get("mano_vid", [])}
            for nid in layout.keys() if nid not in erased_nodes
        }

        vert_to_vals = defaultdict(list)
        for nid, info in valid_nodes.items():
            r, c = map(int, nid.split('-'))
            val = float(norm[r, c])
            for vid in info["mano_vid"]:
                vert_to_vals[vid].append(val)

        n_verts = mano_vertices.shape[0]
        vert_vals = np.zeros(n_verts, dtype=np.float32)
        if vert_to_vals:
            for vid, arr in vert_to_vals.items():
                vert_vals[vid] = float(np.mean(arr))
            known_mask = np.zeros(n_verts, bool)
            known_mask[list(vert_to_vals.keys())] = True
            vert_max = float(vert_vals[known_mask].max())
            vert_vals[~known_mask] = vert_max

        # connectivity smoothing
        vert_vals = _gaussian_smooth_vertex_signal(
            vert_vals, nbrs, dists, sigma=0.005, iters=2
        )

        # invert + min-max normalize on vertices
        mn, mx = float(vert_vals[known_mask].min()), float(vert_vals[known_mask].max())
        vert_vals[~known_mask] = 0.0
        if mx > mn:
            vert_vals[known_mask] = 1.0 - (vert_vals[known_mask] - mn) / (mx - mn)
        else:
            vert_vals[:] = 1.0
            
        vert_vals[~known_mask] = 0.0
        
        colormap_fn = lambda x: np.array(cm.jet(x))
        vertex_colors = colormap_fn(vert_vals)  # RGBA float in [0,1]
        img_rgb = renderer.render(
            vertex_colors=vertex_colors,
            colormap_fn=colormap_fn,
            smooth=True
        )
        # return BGR, like your original code
        return img_rgb[:, :, ::-1], vertex_colors

    # --- setup ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    layout, erased_nodes = _load_layout_json()

    obj_path = _find_first_existing([
        os.path.join("data", "mano_right_neutral_subdiv.obj"),
        os.path.join(os.path.dirname(__file__), "data", "mano_right_neutral_subdiv.obj"),
        os.path.join(os.path.dirname(__file__), "scratch", "mano_right_neutral_subdiv.obj"),
    ])
    if obj_path is None:
        raise FileNotFoundError("Missing mano_right_neutral_subdiv.obj")
    mesh = trimesh.load(obj_path, process=False)
    mano_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mano_faces = np.asarray(mesh.faces, dtype=np.int32)

    nbrs, dists = _build_vertex_graph(mano_vertices, mano_faces)

    width, height = target_size
    renderer = ManoRenderer(
        image_size=(width, height),
        mano_vertices=mano_vertices,
        mano_faces=mano_faces,
    )

    # --- load and PER-FILE normalize all .npy grids --------------------------
    npy_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".npy")
    ]
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    normalized_grids = []  # list of (fname, 16x16 normalized grid)

    for fname in npy_files:
        path = os.path.join(input_dir, fname)
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float32)

        # expect one 16x16 frame; if shape is different, try to coerce
        if arr.ndim == 2:
            grid = arr
        elif arr.ndim >= 3:
            grid = arr[0]  # e.g., first frame if [T,16,16]
        else:
            raise ValueError(f"Unexpected shape for {fname}: {arr.shape}")

        if grid.shape != (16, 16):
            grid = cv2.resize(
                grid,
                (16, 16),
                interpolation=cv2.INTER_AREA
            ).astype(np.float32)

        # PER-FILE min–max normalization to [0,1]
        g_min = float(grid.min())
        g_max = float(grid.max())
        if g_max > g_min:
            grid_norm = (grid - g_min) / (g_max - g_min)
        else:
            grid_norm = np.zeros_like(grid, dtype=np.float32)

        normalized_grids.append((fname, grid_norm))

    # --- valid mask (same as vmin/vmax & MANO mapping uses) ------------------
    valid_mask = np.zeros((16, 16), dtype=bool)
    for nid in layout.keys():
        if nid in erased_nodes:
            continue
        r, c = map(int, nid.split('-'))
        valid_mask[r, c] = True

    # --- compute 16x16 mean ONLY on valid nodes ------------------------------
    stack_norm = np.stack([g for _, g in normalized_grids], axis=0)  # (N,16,16)

    mean_grid = np.zeros((16, 16), dtype=np.float32)
    # mean over normalized grids, but only for valid_mask positions
    mean_grid[valid_mask] = stack_norm[:, valid_mask].mean(axis=0)
    # invalid nodes remain 0

    mean_path = os.path.join(output_dir, "tactile_mean_16x16.npy")
    np.save(mean_path, mean_grid)
    print(f"Saved 16x16 per-taxel mean (valid nodes only) to {mean_path}")

    # --- subtract mean_grid (reduce the average) -----------------------------
    centered_grids = [
        (fname, grid_norm - mean_grid)
        for fname, grid_norm in normalized_grids
    ]

    # --- vmin/vmax on valid nodes using centered signals ---------------------
    stack_centered = np.stack([g for _, g in centered_grids], axis=0)
    vmin = float(stack_centered[:, valid_mask].min())
    vmax = float(stack_centered[:, valid_mask].max())
    if vmax <= vmin:
        vmax = vmin + 1e-6

    print(f"Global vmin={vmin:.4f}, vmax={vmax:.4f} over {len(centered_grids)} centered frames")

    # --- render each centered grid through MANO ------------------------------
    alpha = 1.2      # contrast
    beta = 0.1 * 255 # brightness

    for fname, grid_centered in centered_grids:
        img_bgr, _ = _render_pressure_mano(
            mano_vertices, mano_faces, renderer,
            grid_centered, layout, erased_nodes,
            vmin, vmax,
            nbrs, dists
        )

        img_adj = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, base + ".png")
        cv2.imwrite(out_path, img_adj)
        print(f"Saved {out_path}")

    print("Done rendering MANO tactile patterns with valid-node mean subtraction.")


if __name__ == "__main__":
    render_tactile_patterns_to_mano(
        input_dir="../tactile_pattern",
        output_dir="../tactile_pattern_rendered",
        target_size=(960, 960),
    )
