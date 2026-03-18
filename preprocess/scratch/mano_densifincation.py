# save as: export_mano_neutral_obj.py
import os, sys, argparse, numpy as np
import torch

# --- compat shims for old MANO deps on Py3.11/NumPy>=2 ---
import inspect, collections as _col
if not hasattr(inspect, "getargspec"):
    from inspect import getfullargspec
    ArgSpec = _col.namedtuple("ArgSpec", "args varargs keywords defaults")
    inspect.getargspec = lambda fn: ArgSpec(*getfullargspec(fn)[:4])  # type: ignore
import numpy as _np
for _n,_v in [('int',int),('float',float),('bool',bool),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if not hasattr(_np,_n): setattr(_np,_n,_v)
# ---------------------------------------------------------
MAX_EDGE = 0.004
DENSE_THRES = 0.005
SPARSE_THRES = 0.007

def export_mano():
    ap = argparse.ArgumentParser("Export MANO neutral mesh → OBJ")
    ap.add_argument("--model", default='preprocess/data/MANO_RIGHT.pkl',
                    help="Path to MANO_RIGHT/LEFT.pkl or the directory that contains them (e.g. preprocess/data/smplx/smplh)")
    ap.add_argument("--side", choices=["right","left"], default="right")
    ap.add_argument("--out", default=None, help="Output OBJ path (default: mano_<side>_neutral.obj)")
    args = ap.parse_args()

    # smplx MANO expects model_path = directory containing MANO_RIGHT/LEFT.pkl
    model_dir = args.model if os.path.isdir(args.model) else os.path.dirname(args.model)
    is_rhand = (args.side == "right")

    # --- build MANO, neutral shape/pose ---
    from smplx import MANO
    mano = MANO(model_path=model_dir, is_rhand=is_rhand, use_pca=False, flat_hand_mean=True).to('cpu')  # axis-angle (45) for hand
    device = torch.device("cpu")
    betas         = torch.zeros(1, 10, dtype=torch.float32, device=device)
    hand_pose     = torch.zeros(1, 45, dtype=torch.float32, device=device)  # neutral fingers
    global_orient = torch.zeros(1, 3,  dtype=torch.float32, device=device)  # no global rot
    transl        = torch.zeros(1, 3,  dtype=torch.float32, device=device)  # no translation

    out = mano(return_verts=True, betas=betas, hand_pose=hand_pose,
               global_orient=global_orient, transl=transl)
    verts = out.vertices[0].detach().cpu().numpy()    # (778,3)
    faces = mano.faces.astype(_np.int32)              # (1538,3)

    # --- write OBJ ---
    out_path = args.out or f"mano_{args.side}_neutral.obj"
    with open(out_path, "w") as f:
        f.write("# MANO neutral mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces + 1:  # OBJ is 1-indexed
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    print(f"Exported: {out_path}")

# save as: subdivide_with_trimesh.py
import argparse, json, numpy as np, trimesh

def subdivide():
    ap = argparse.ArgumentParser("Subdivide mesh with trimesh.remesh.subdivide_to_size")
    ap.add_argument("--in", dest="inp", default="preprocess/data/mano_right_neutral.obj")
    ap.add_argument("--out", dest="out", default="preprocess/data/mano_right_neutral_subdiv.obj")
    ap.add_argument("--map", dest="map_path", default="preprocess/data/mano_right_neutral_subdiv_mapping.json")
    ap.add_argument("--max_edge", type=float, default=MAX_EDGE)
    ap.add_argument("--max_iter", type=int, default=20)
    args = ap.parse_args()

    # Load
    mesh = trimesh.load(args.inp, process=False)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)

    # Subdivide; request index of original face for each new triangle
    V2, F2, idx = trimesh.remesh.subdivide_to_size(
        V, F, max_edge=args.max_edge, max_iter=args.max_iter, return_index=True
    )
    # Some trimesh versions return (q,) or (q,3); normalize to (q,)
    if idx.ndim == 2:
        idx = idx[:, 0]

    # Build mapping: original face id -> list of new face ids
    mapping = {}
    for new_id, orig in enumerate(idx.tolist()):
        mapping.setdefault(str(int(orig)), []).append(int(new_id))

    # Save OBJ (triangle soup)
    with open(args.out, "w") as f:
        for v in V2:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for a, b, c in F2:
            f.write(f"f {a+1} {b+1} {c+1}\n")

    # Save mapping JSON
    with open(args.map_path, "w") as jf:
        json.dump({
            "max_edge": args.max_edge,
            "max_iter": args.max_iter,
            "original_face_count": int(len(F)),
            "new_face_count": int(len(F2)),
            "mapping": mapping
        }, jf, indent=2)

    print(f"Saved OBJ: {args.out}")
    print(f"Saved mapping: {args.map_path}")
    print(f"Faces: {len(F)} -> {len(F2)} | Verts: {len(V)} -> {len(V2)}")

# save as: export_grid_points_obj.py
import json, argparse

def layout2pcd():
    ap = argparse.ArgumentParser("Export 16x16 positions to 3D point-cloud OBJ")
    ap.add_argument("--in-file", default="preprocess/data/handLayoutNewest.json", help="path to JSON with {'positions': {...}, 'erasedNodes': [...]} (erased not required)")
    ap.add_argument("--out", default="preprocess/data/grid_points.obj", help="output OBJ path")
    ap.add_argument("--exclude-erased", action="store_true", help="skip keys listed in erasedNodes")
    args = ap.parse_args()
    
    in_path = args.in_file
    
    with open(in_path, "r") as f:
        data = json.load(f)
        
    pos = data.get("positions", {})
    erased = list(data.get("erasedNodes", []))

    verts = []
    order = []
    for r in range(16):
        for c in range(16):
            k = f"{r}-{c}"
            p = pos.get(k)
            if k in erased: 
                x2 = -float(1)      # space X  = -orig y
                y2 = 0.0                 # space Y  = 0
                z2 =  -float(1)      # space Z  =  orig x
                verts.append((x2, y2, z2))
                order.append(k)
            else:
                x2 = float(p["y"]) * 0.0005  - 0.128    # space X  = -orig y
                y2 = 0.0                 # space Y  = 0
                z2 =  float(p["x"]) * 0.0005  -0.256  # space Z  =  orig x
                verts.append((x2, y2, z2))
                order.append(k)

    with open(args.out, "w") as f:
        f.write("# point cloud from 16x16 grid (X=-y, Y=0, Z=x)\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # no faces: pure point cloud

    print(f"Wrote {len(verts)} vertices to {args.out}")
    # If you want to see the vertex order, uncomment:
    # json.dump(order, open(args.out.replace('.obj', '_order.json'), 'w'), indent=2)

#!/usr/bin/env python3
# snap_points_y_to_mesh_face_mean.py

import argparse
import numpy as np
import trimesh

def load_points_obj(path):
    """Return Nx3 float array from v-lines; preserve order."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, xs, ys, zs = line.strip().split()[:4]
                verts.append((float(xs), float(ys), float(zs)))
    if not verts:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(verts, dtype=np.float64)

def save_points_obj(path, verts):
    with open(path, "w") as f:
        f.write("# snapped points (y possibly increased)\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

def snap_3dgrip_to_mesh():
    ap = argparse.ArgumentParser(
        description="Snap point-cloud y to mean-y of closest mesh face if distance<thresh; only increase y."
    )
    ap.add_argument("--points", default="preprocess/data/grid_points_palm_aligned_merged.obj", help="grid_points_palm_aligned_merged.obj")
    ap.add_argument("--mesh", default="preprocess/data/mano_right_neutral_subdiv.obj", help="mano_right_neutral_subdiv.obj")
    ap.add_argument("--out", default="preprocess/data/grid_points_palm_aligned_merged_snapped.obj", help="output OBJ for snapped points")
    ap.add_argument("--thresh", type=float, default=5.0, help="distance threshold")
    ap.add_argument("--batch",  type=int, default=4096, help="batch size for nearest queries")
    args = ap.parse_args()

    # Load data
    P = load_points_obj(args.points)                   # (N,3)
    mesh = trimesh.load(args.mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        # in case the file contains a Scene, concatenate its geometry
        mesh = trimesh.util.concatenate(tuple(
            g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ))
    V = mesh.vertices
    F = mesh.faces

    # Precompute mean y per face
    face_mean_y = V[F].mean(axis=1)[:, 1]  # (M,)

    # Proximity: get closest point & triangle index for each input point
    # Use vectorized batches to avoid excessive memory usage
    newP = P.copy()
    N = len(P)

    # Use trimesh.proximity.closest_point if available; fallback to on_surface
    use_cp = hasattr(trimesh.proximity, "closest_point")
    for start in range(0, N, args.batch):
        end = min(N, start + args.batch)
        pts = P[start:end]

        if use_cp:
            # Returns (closest_points, distance, triangle_id)
            cpts, dists, tids = trimesh.proximity.closest_point(mesh, pts)
        else:
            # Older API
            cpts, tids = mesh.nearest.on_surface(pts)
            # Euclidean distance to the surface points
            dists = np.linalg.norm(cpts - pts, axis=1)

        # For each point, decide snapping
        for i in range(end - start):
            tid = int(tids[i])
            if tid < 0 or tid >= len(F):
                continue
            if dists[i] >= args.thresh:
                continue
            target_y = float(face_mean_y[tid])
            # only increase y
            if target_y > newP[start + i, 1]:
                newP[start + i, 1] = target_y

    save_points_obj(args.out, newP)

import argparse, json, numpy as np, trimesh

def load_points_obj(path):
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, xs, ys, zs = line.strip().split()[:4]
                verts.append((float(xs), float(ys), float(zs)))
    if not verts:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(verts, dtype=np.float64)

def make_point_names(N=16):
    return [f"{r}-{c}" for r in range(N) for c in range(N)]

def distinct_colors(n, seed=0):
    # Evenly spaced HSV → RGB (0–255)
    rng = np.random.default_rng(seed)
    # Start from evenly spaced hues and jitter a bit for variety
    hues = (np.arange(n) / n + rng.uniform(-0.02, 0.02, n)) % 1.0
    s, v = 0.65, 0.95
    def hsv2rgb(h, s, v):
        i = int(h*6.0); f = h*6.0 - i
        p, q, t = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
        i %= 6
        r,g,b = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]
        return int(r*255), int(g*255), int(b*255)
    return np.array([hsv2rgb(h, s, v) + (255,) for h in hues], dtype=np.uint8)

def batched_nearest(queries, points, batch=8192):
    """
    For each query (M,3), find nearest point in `points` (N,3).
    Returns (idx (M,), dist (M,))
    """
    M, N = len(queries), len(points)
    idx = np.empty(M, dtype=np.int32)
    dmin = np.empty(M, dtype=np.float64)
    for s in range(0, M, batch):
        e = min(M, s+batch)
        q = queries[s:e][:, None, :]            # (B,1,3)
        p = points[None, :, :]                  # (1,N,3)
        d2 = np.sum((q - p)**2, axis=2)         # (B,N)
        I = np.argmin(d2, axis=1)
        idx[s:e] = I
        dmin[s:e] = np.sqrt(d2[np.arange(e-s), I])
    return idx, dmin

def mesh_point_mapping():
    """
    Color mesh *vertices* by nearest snapped point (thresholded);
    output colored PLY (per-vertex colors) and point -> vertex_ids JSON.
    """
    import argparse, json, numpy as np, trimesh

    # --- helpers (same as before) --------------------------------------------
    def load_points_obj(path):
        vs = []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    _, xs, ys, zs = line.strip().split()[:4]
                    vs.append((float(xs), float(ys), float(zs)))
        if not vs:
            raise ValueError(f"No vertices found in {path}")
        return np.asarray(vs, dtype=np.float64)

    def make_point_names(N=16):
        return [f"{r}-{c}" for r in range(N) for c in range(N)]

    def distinct_colors(n, seed=0):
        rng = np.random.default_rng(seed)
        hues = (np.arange(n) / n + rng.uniform(-0.02, 0.02, n)) % 1.0
        s, v = 0.65, 0.95
        def hsv2rgb(h, s, v):
            i = int(h*6.0); f = h*6.0 - i
            p, q, t = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
            i %= 6
            r,g,b = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]
            return int(r*255), int(g*255), int(b*255)
        return np.array([hsv2rgb(h, s, v) + (255,) for h in hues], dtype=np.uint8)

    def batched_nearest(queries, points, batch=8192):
        M = len(queries)
        idx = np.empty(M, dtype=np.int32)
        dmin = np.empty(M, dtype=np.float64)
        for s in range(0, M, batch):
            e = min(M, s+batch)
            q = queries[s:e][:, None, :]        # (B,1,3)
            p = points[None, :, :]              # (1,N,3)
            d2 = np.sum((q - p)**2, axis=2)     # (B,N)
            I = np.argmin(d2, axis=1)
            idx[s:e] = I
            dmin[s:e] = np.sqrt(d2[np.arange(e-s), I])
        return idx, dmin

    # --- args ----------------------------------------------------------------
    ap = argparse.ArgumentParser(
        description="Color mesh vertices by nearest snapped point (thresholded); export colored PLY and point→vertex_ids JSON."
    )
    ap.add_argument("--points", default="preprocess/data/grid_points_palm_aligned_merged_snapped.obj",
                    help="snapped points OBJ (row-major 16x16)")
    ap.add_argument("--mesh", default="preprocess/data/mano_right_neutral_subdiv.obj",
                    help="subdivided mesh OBJ")
    ap.add_argument("--out_mesh", default="preprocess/data/mano_right_neutral_subdiv_colored_vertices.ply",
                    help="output colored mesh PLY (per-vertex colors)")
    ap.add_argument("--out_json", default="preprocess/data/point_to_vertex_ids.json",
                    help="JSON mapping point name→[vertex_ids]")
    ap.add_argument("--seed", type=int, default=0, help="random seed for point colors")
    ap.add_argument("--batch", type=int, default=16384, help="batch size for NN queries")
    args = ap.parse_args()

    # --- load ----------------------------------------------------------------
    P = load_points_obj(args.points)
    if len(P) != 256:
        raise ValueError(f"Expected 256 points; got {len(P)}")
    names = make_point_names(16)

    mesh = trimesh.load(args.mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    V = mesh.vertices.copy()   # (Nv,3)
    F = mesh.faces

    # --- nearest point for each *vertex* -------------------------------------
    nn_idx, nn_dist = batched_nearest(V, P, batch=args.batch)  # nearest snapped point per vertex

    # row/col from point index (row-major 16x16)
    rows = nn_idx // 16
    cols = nn_idx % 16

    # threshold rule: if (row <= 8) OR (col >= 11) -> 0.004 else 0.008
    per_vtx_thresh = np.where((rows <= 8) , DENSE_THRES, SPARSE_THRES)
    valid_mask = nn_dist < per_vtx_thresh

    # --- colors & mapping ----------------------------------------------------
    point_colors = distinct_colors(256, seed=args.seed)             # (256,4)
    vtx_colors = np.tile(np.array([[10,10,10,255]], np.uint8), (len(V), 1))
    vtx_colors[valid_mask] = point_colors[nn_idx[valid_mask]]

    # mapping: point name -> list of vertex ids
    mapping = {n: [] for n in names}
    valid_ids = np.nonzero(valid_mask)[0]
    for vid in valid_ids:
        pid = int(nn_idx[vid])
        mapping[names[pid]].append(int(vid))

    # --- export --------------------------------------------------------------
    colored = trimesh.Trimesh(vertices=V, faces=F, process=False)
    colored.visual.vertex_colors = vtx_colors
    colored.export(args.out_mesh)

    with open(args.out_json, "w") as f:
        json.dump({
            "threshold_row_le_8_or_col_ge_11": DENSE_THRES,
            "threshold_else": SPARSE_THRES,
            "vertices_total": int(len(V)),
            "vertices_colored": int(valid_mask.sum()),
            "point_to_vertex_ids": mapping
        }, f, indent=2)

    print(f"Saved colored mesh (vertices): {args.out_mesh}")
    print(f"Saved mapping JSON: {args.out_json}")
    print(f"Colored vertices: {valid_mask.sum()} / {len(V)}")

def mesh_point_mapping_new():
    """
    Iterate points; for each point, gather all mesh vertices within a row-dependent
    threshold (rows<=8 -> DENSE_THRES else SPARSE_THRES). A vertex may be assigned
    to multiple points. Vertex color = average of owning points' colors.
    Exports vertex-colored PLY and point→vertex_ids JSON.
    """
    import argparse, json, numpy as np, trimesh

    def load_points_obj(path):
        vs = []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    _, xs, ys, zs = line.strip().split()[:4]
                    vs.append((float(xs), float(ys), float(zs)))
        if not vs:
            raise ValueError(f"No vertices found in {path}")
        return np.asarray(vs, dtype=np.float64)

    def make_point_names(N=16):
        return [f"{r}-{c}" for r in range(N) for c in range(N)]

    def distinct_colors(n, seed=0):
        rng = np.random.default_rng(seed)
        hues = (np.arange(n) / n + rng.uniform(-0.02, 0.02, n)) % 1.0
        s, v = 0.65, 0.95
        def hsv2rgb(h, s, v):
            i = int(h*6.0); f = h*6.0 - i
            p, q, t = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
            i %= 6
            r,g,b = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]
            return int(r*255), int(g*255), int(b*255)
        cols = np.array([hsv2rgb(h, s, v) for h in hues], dtype=np.uint8)
        alpha = np.full((n, 1), 255, dtype=np.uint8)
        return np.concatenate([cols, alpha], axis=1)  # RGBA

    # args
    ap = argparse.ArgumentParser(
        description="Iterate points; allow vertices to be assigned to multiple points; average colors per vertex."
    )
    ap.add_argument("--points", default="preprocess/data/grid_points_palm_aligned_merged_snapped.obj")
    ap.add_argument("--mesh",   default="preprocess/data/mano_right_neutral_subdiv.obj")
    ap.add_argument("--out_mesh", default="preprocess/data/mano_right_neutral_subdiv_colored_vertices.ply")
    ap.add_argument("--out_json", default="preprocess/data/point_to_vertex_ids.json")
    ap.add_argument("--dense", type=float, default=0.004, help="threshold for rows <= 8")
    ap.add_argument("--sparse", type=float, default=0.008, help="threshold for rows >= 9")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    DENSE_THRES, SPARSE_THRES = args.dense, args.sparse

    # load
    P = load_points_obj(args.points)
    if len(P) != 256:
        raise ValueError(f"Expected 256 points; got {len(P)}")
    names = make_point_names(16)

    mesh = trimesh.load(args.mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    V, F = mesh.vertices.copy(), mesh.faces
    Nv = len(V)

    # colors + accumulators
    point_colors = distinct_colors(256, seed=args.seed)          # (256,4) uint8
    vtx_sum = np.zeros((Nv, 4), dtype=np.float64)                # sum of RGBA
    vtx_cnt = np.zeros(Nv, dtype=np.int32)                       # owner count
    mapping = {n: [] for n in names}                             # point → [vertex_ids]

    # iterate points, collect neighbors under threshold
    for p_idx, p in enumerate(P):
        r = p_idx // 16
        th = DENSE_THRES if r <= 8 else SPARSE_THRES
        d = np.linalg.norm(V - p, axis=1)
        nbr = np.where(d < th)[0]
        if nbr.size == 0:
            continue
        # update mapping and accumulators
        mapping[names[p_idx]].extend(nbr.tolist())
        vtx_sum[nbr] += point_colors[p_idx].astype(np.float64)
        vtx_cnt[nbr] += 1

    # finalize vertex colors: average of owning points; unowned -> dark gray
    vtx_colors = np.tile(np.array([[10, 10, 10, 255]], dtype=np.uint8), (Nv, 1))
    owned = vtx_cnt > 0
    vtx_colors[owned] = np.clip((vtx_sum[owned] / vtx_cnt[owned, None]).round(), 0, 255).astype(np.uint8)

    # export
    colored = trimesh.Trimesh(vertices=V, faces=F, process=False)
    colored.visual.vertex_colors = vtx_colors
    colored.export(args.out_mesh)

    with open(args.out_json, "w") as f:
        json.dump({
            "dense_threshold_rows_le_8": float(DENSE_THRES),
            "sparse_threshold_rows_ge_9": float(SPARSE_THRES),
            "vertices_total": int(Nv),
            "vertices_colored": int(int(owned.sum())),
            "point_to_vertex_ids": mapping
        }, f, indent=2)

    print(f"Saved colored mesh (vertices): {args.out_mesh}")
    print(f"Saved mapping JSON: {args.out_json}")
    print(f"Vertices with at least one owner: {owned.sum()} / {Nv}")
    
import json, re, argparse

def rewrite_mapjson():
    ap = argparse.ArgumentParser("Replace 'mano_faceid' with vertex ids from point_to_vertex_ids.json")
    ap.add_argument("--layout", default="preprocess/data/handLayoutNewest_meshid.json")
    ap.add_argument("--mapping", default="preprocess/data/point_to_vertex_ids.json")
    ap.add_argument("--out", default="preprocess/data/handLayoutNewest_meshid.json")
    args = ap.parse_args()

    layout = json.load(open(args.layout))
    m = json.load(open(args.mapping))
    pt2v = m.get("point_to_vertex_ids", m)  # allow wrapped or direct
    pat = re.compile(r"^\d+-\d+$")

    def derive_key(d):
        for k in ("sensor_id","id","key","name","sensorKey"):
            v = d.get(k)
            if isinstance(v, str) and pat.match(v): return v
        if {"row","col"} <= d.keys():
            return f"{int(d['row'])}-{int(d['col'])}"
        return None

    def walk(o, hint=None):
        if isinstance(o, dict):
            key = hint or derive_key(o)
            if key and key in pt2v:
                o.pop("mano_faceid", None)
                o["mano_vid"] = pt2v[key]
            for k, v in o.items():
                walk(v, k if isinstance(k, str) and pat.match(k) else key)
        elif isinstance(o, list):
            for v in o: walk(v, hint)

    walk(layout)
    json.dump(layout, open(args.out, "w"), indent=2)
    print(f"Saved: {args.out}")
    
if __name__ == "__main__":
    subdivide() #--max_edge
    mesh_point_mapping_new()
    rewrite_mapjson()
