import io
import math
import os
import re
from collections import deque

import h5py
import numpy as np


def process_pressure_discrete_intervals(
    pressure: np.ndarray,
    max_value: float = 3072.0,
    num_intervals: int = 5,
    output_range: str = "uint8",
    method: str = "binning",
    noise_threshold: float = 500.0,
) -> np.ndarray:
    """Convert pressure to discrete interval levels.

    Args:
        pressure: Raw pressure array (any shape)
        max_value: Maximum tactile value (default: 3072.0)
        num_intervals: Discrete levels - 3/5/7 (default: 5)
        output_range: 'uint8' [0,255] or 'float' [0,1] (default: 'uint8')
        method: 'binning' 'linear' 'log' 'none'
        noise_threshold: Noise floor for linear/log methods (default: 500.0)

    Returns:
        Discretized pressure array (uint8 or float32)
    """
    frame = np.asarray(pressure, dtype=np.float32)
    frame = np.where(np.isfinite(frame), frame, 0.0)
    frame = np.clip(frame, 0.0, max_value)

    if method == "binning":
        if num_intervals > 1:
            bins = np.linspace(0.0, max_value, num_intervals + 1, dtype=np.float32)
            indices = np.digitize(frame, bins[1:-1], right=False)
            levels = np.linspace(0.0, 1.0, num_intervals, dtype=np.float32)
            frame = levels[indices]
        else:
            frame = frame / max_value

    elif method == "linear":
        vmin, vmax = noise_threshold, max_value
        clamped = np.clip(frame, vmin, vmax)
        norm = (clamped - vmin) / (vmax - vmin)

        if num_intervals > 1:
            levels = np.linspace(0.0, 1.0, num_intervals, dtype=np.float32)
            indices = np.round(norm * (num_intervals - 1)).astype(np.int32)
            frame = levels[np.clip(indices, 0, num_intervals - 1)]
        else:
            frame = norm

    elif method == "log":
        vmin, vmax = noise_threshold, max_value
        clamped = np.clip(frame, vmin, vmax)
        log_val = np.log10(clamped)
        norm = (log_val - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))

        if num_intervals > 1:
            levels = np.linspace(0.0, 1.0, num_intervals, dtype=np.float32)
            indices = np.round(norm * (num_intervals - 1)).astype(np.int32)
            frame = levels[np.clip(indices, 0, num_intervals - 1)]
        else:
            frame = norm

    else:
        # no discretization, use normalized pressure (0-1)
        frame = frame / max_value

    if output_range == "uint8":
        return (frame * 255.0).astype(np.uint8)
    elif output_range == "float":
        return frame.astype(np.float32)
    else:
        raise ValueError(f"Invalid output_range: {output_range}. Use 'uint8' or 'float'")


def load_hdf5(file_path):
    """Load data from an HDF5 file.
    
    Args:
        file_path (str): Path to the HDF5 file
        
    Returns:
        dict: Dictionary containing all datasets from the file
    """
    with h5py.File(file_path, 'r') as f:
        return {key: _load_hdf5_item(f[key]) for key in f.keys()}


def list_hdf5_keys(file_path, recursive=False, root=None):
    """List keys contained within an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file
        recursive (bool): If True, include nested keys in their full paths.
        root (str | None): Optional path within the file to start listing from.

    Returns:
        list[str]: Collection of keys (top-level by default, full paths if recursive)
    """
    def _collect_keys(group, prefix=""):
        keys = []
        for name, item in group.items():
            full_name = f"{prefix}/{name}" if prefix else name
            keys.append(full_name)
            if recursive and isinstance(item, h5py.Group):
                keys.extend(_collect_keys(item, full_name))
        return keys

    with h5py.File(file_path, 'r') as f:
        if not root:
            return _collect_keys(f)

        obj = f[root]
        if isinstance(obj, h5py.Dataset):
            return [root]

        keys = [root]
        keys.extend(_collect_keys(obj, prefix=root))
        return keys


def _require_data_group(h5file, file_path):
    """Retrieve the `data` group from an HDF5 file.
    
    Raises:
        KeyError: If 'data' group is not found
    """
    if "data" not in h5file:
        raise KeyError(f"No 'data' group found in '{file_path}'.")
    return h5file["data"]


def _numeric_suffix(value):
    match = re.search(r"(\d+)$", value or "")
    return int(match.group(1)) if match else None


def _resolve_demo_name(data_group, requested_demo):
    """
    Resolve a requested demo id to an actual name inside the `data` group.

    Supports exact matches (e.g., 'grocery_clip_000') and aliases that share the same
    numeric suffix (e.g., 'demo_00'). When suffix matching fails, falls back to ordering.
    """
    if requested_demo in data_group:
        return requested_demo

    target_index = _numeric_suffix(requested_demo)
    if target_index is None:
        return None

    names = sorted(data_group.keys())
    for name in names:
        if _numeric_suffix(name) == target_index:
            return name

    if 0 <= target_index < len(names):
        return names[target_index]

    return None


_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

_POSE_COLORS = {
    "left_hand_landmarks": (66, 165, 245),
    "right_hand_landmarks": (244, 81, 96),
}


def export_demo(file_path, demo_id, output_path, skip_keys=None):
    """Export an individual demo to a separate HDF5 file.

    Args:
        file_path (str): Path to the source HDF5 file.
        demo_id (str): Demo identifier under the `data` group.
        output_path (str): Destination path for the exported HDF5.
        skip_keys (Iterable[str] | None): Optional top-level datasets/groups to remove.
        
    Raises:
        KeyError: If demo is not found
    """
    skip = set(skip_keys or [])

    with h5py.File(file_path, 'r') as src:
        data_group = _require_data_group(src, file_path)
        resolved_demo = _resolve_demo_name(data_group, demo_id)
        if resolved_demo is None:
            available = ", ".join(list(data_group.keys())[:5])
            raise KeyError(f"Demo '{demo_id}' not found in '{file_path}'. Available demos include: {available}")
        
        demo_group = data_group[resolved_demo]

        with h5py.File(output_path, 'w') as dst:
            src.copy(demo_group, dst, name=demo_id)
            
            if skip:
                out_group = dst[demo_id]
                for key in skip:
                    if key in out_group:
                        del out_group[key]


def load_all_demos(file_path, demos=None, skip_keys=None):
    """Load one or more demos from the `data` group of an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        demos (Iterable[str] | None): Demo ids to load (e.g., {"demo_00"}).
            Loads all demos when omitted.
        skip_keys (Iterable[str] | None): Dataset/group names within each demo to skip.

    Returns:
        dict: Mapping of demo id to loaded content.
        
    Raises:
        KeyError: If requested demos are not found
    """
    skip = set(skip_keys or [])
    with h5py.File(file_path, 'r') as f:
        data_group = _require_data_group(f, file_path)
        
        if demos is None:
            selected = sorted(data_group.keys())
        else:
            selected = []
            missing = []
            for name in demos:
                resolved = _resolve_demo_name(data_group, name)
                if resolved is None:
                    missing.append(name)
                else:
                    selected.append(resolved)
            
            if missing:
                available = ", ".join(list(data_group.keys())[:5])
                raise KeyError(f"Demo ids not found: {', '.join(missing)}. Available demos include: {available}")

        loaded = {}
        for demo in selected:
            group = data_group[demo]
            loaded[demo] = {
                key: _load_hdf5_item(group[key])
                for key in group.keys()
                if key not in skip
            }
        return loaded


def _decode_attr(value):
    """Best-effort conversion of HDF5 attribute values to python strings."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_attr(value.item())
        return None
    return str(value)


def _get_demo_dataset(src, resolved_demo, dataset_name):
    """Get dataset from resolved demo path.
    
    Raises:
        KeyError: If dataset is not found
        TypeError: If path points to a group instead of dataset
    """
    demo_dataset_path = f"data/{resolved_demo}/{dataset_name}"
    if demo_dataset_path not in src:
        raise KeyError(f"Dataset '{dataset_name}' not found for demo '{resolved_demo}'.")
    
    dataset = src[demo_dataset_path]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"'{demo_dataset_path}' is not a dataset.")
    
    return dataset

def load_hdf5_dataset(file_path, dataset_name):
    """Load a specific dataset from an HDF5 file.
    
    Args:
        file_path (str): Path to the HDF5 file
        dataset_name (str): Name of the dataset to load
        
    Returns:
        numpy.ndarray: The requested dataset
        
    Raises:
        KeyError: If dataset is not found
        TypeError: If the path points to a group instead of dataset
    """
    with h5py.File(file_path, 'r') as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in '{file_path}'.")
        
        obj = f[dataset_name]
        if isinstance(obj, h5py.Group):
            raise TypeError(f"'{dataset_name}' is an HDF5 group. Provide the full path to a dataset.")
        
        return obj[()]


def _load_hdf5_item(obj):
    """
    Recursively load datasets or groups from an HDF5 object.
    """
    if isinstance(obj, h5py.Dataset):
        return obj[()]
    if isinstance(obj, h5py.Group):
        return {name: _load_hdf5_item(obj[name]) for name in obj.keys()}
    raise TypeError(f"Unknown HDF5 object type: {type(obj)}")


def export_rgb_frames(
    file_path,
    demo_id,
    output_dir,
    dataset_name="rgb_images_jpeg",
    target_size=(480, 480),
    channel_order=None,
):
    """Extract RGB frames stored as JPEG byte arrays for a demo and write them to disk.

    Args:
        file_path (str): Path to the source HDF5 file.
        demo_id (str): Demo identifier under the `data` group.
        output_dir (str): Directory where frames will be written.
        dataset_name (str): Dataset name containing the encoded RGB frames.
        target_size (tuple[int, int] | None): Resize frames to this resolution.
        channel_order (str | None): Override color channel order when saving (e.g., "bgr").
        
    Raises:
        KeyError: If demo or dataset is not found
    """
    with h5py.File(file_path, "r") as src:
        data_group = _require_data_group(src, file_path)
        resolved_demo = _resolve_demo_name(data_group, demo_id)
        if resolved_demo is None:
            available = ", ".join(list(data_group.keys())[:5])
            raise KeyError(f"Demo '{demo_id}' not found in '{file_path}'. Available demos include: {available}")

        dataset = _get_demo_dataset(src, resolved_demo, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        from PIL import Image, ImageEnhance

        attr_order = _decode_attr(dataset.attrs.get("channel_order"))
        attr_order = attr_order or _decode_attr(dataset.attrs.get("color_order"))
        active_order = (channel_order or attr_order or "rgb").lower()

        for idx, entry in enumerate(dataset):
            buffer = entry.tobytes() if isinstance(entry, np.ndarray) else bytes(entry)
            image = Image.open(io.BytesIO(buffer)).convert("RGB")

            if active_order == "bgr":
                # swap channels
                image = Image.fromarray(np.array(image)[:, :, ::-1], mode="RGB")

            if target_size:
                image = image.resize(target_size, Image.BILINEAR)

            # ---- adjust brightness & contrast here ----
            brightness_factor = 1.2  # >1 = brighter, <1 = darker
            contrast_factor   = 1.2  # >1 = more contrast, <1 = less

            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)

            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
            # ------------------------------------------

            frame_path = os.path.join(output_dir, f"{demo_id}_{idx:05d}.jpg")
            image.save(frame_path, format="JPEG", quality=95)


def export_tactile_heatmaps_raw(
    file_path,
    demo_id,
    output_dir,
    process_fn,
    dataset_names=("left_pressure", "right_pressure"),
    cmap="viridis",
    target_size=(480, 480),
    max_value=3072.0,
    method="none",
    noise_threshold=500.0,
    gaussian_sigma=0.0,
    temporal_alpha=0.2,
):
    """Convert tactile pressure samples using custom processing function (e.g., raw normalization).

    Args:
        file_path (str): Path to the source HDF5 file.
        demo_id (str): Demo identifier under the `data` group.
        output_dir (str): Directory where heatmaps will be written.
        process_fn: Function to process pressure data (e.g., process_pressure_discrete_intervals).
        dataset_names (Iterable[str]): Names of tactile datasets to export.
        cmap (str): Matplotlib colormap name (fallback to grayscale if unavailable).
        target_size (tuple[int, int] | None): Resize each image to this size.
        max_value (float): Maximum tactile value expected; values above are clipped.
        method (str): Processing method passed to process_fn.
        noise_threshold (float): Noise threshold passed to process_fn.
        gaussian_sigma (float): Spatial Gaussian smoothing sigma in pixels. <=0 disables.
        temporal_alpha (float): Temporal smoothing factor (0-1). <=0 disables.

    Raises:
        KeyError: If demo is not found
    """
    try:
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap)
    except ImportError:
        colormap = None

    max_value = float(max_value) if max_value is not None else 3072.0
    if max_value <= 0.0:
        max_value = 3072.0

    with h5py.File(file_path, "r") as src:
        data_group = _require_data_group(src, file_path)
        resolved_demo = _resolve_demo_name(data_group, demo_id)
        if resolved_demo is None:
            available = ", ".join(list(data_group.keys())[:5])
            raise KeyError(f"Demo '{demo_id}' not found in '{file_path}'. Available demos include: {available}")

        demo_prefix = f"data/{resolved_demo}"
        for name in dataset_names:
            dataset_path = f"{demo_prefix}/{name}"
            if dataset_path not in src:
                continue

            dset = src[dataset_path]
            if not isinstance(dset, h5py.Dataset):
                continue

            target_dir = os.path.join(output_dir, name)
            os.makedirs(target_dir, exist_ok=True)

            prev_frame = None
            for idx, sample in enumerate(dset):
                frame = _prepare_tactile_frame(sample, dset.attrs)

                # Apply spatial Gaussian smoothing before processing
                if gaussian_sigma and gaussian_sigma > 0.0:
                    frame = _apply_gaussian(frame, gaussian_sigma)

                # Apply temporal smoothing
                if (
                    prev_frame is not None
                    and temporal_alpha is not None
                    and 0.0 < temporal_alpha < 1.0
                ):
                    frame = temporal_alpha * frame + (1.0 - temporal_alpha) * prev_frame
                prev_frame = frame

                # Use custom processing function
                normalized = process_fn(
                    pressure=frame,
                    max_value=max_value,
                    num_intervals=1,  # No discretization for raw
                    output_range="float",
                    method=method,
                    noise_threshold=noise_threshold,
                )

                # Apply colormap
                if colormap is None:
                    pixels = (normalized * 255.0).astype(np.uint8)
                    mode = "L"
                else:
                    rgba = (colormap(normalized) * 255.0).astype(np.uint8)
                    pixels = rgba[..., :3]
                    mode = "RGB"

                from PIL import Image
                image = Image.fromarray(pixels, mode=mode)

                if target_size:
                    image = image.resize(target_size, _HEATMAP_RESAMPLE)
                filename = os.path.join(target_dir, f"{demo_id}_{idx:05d}.png")
                image.save(filename)


def export_tactile_heatmaps(
    file_path,
    demo_id,
    output_dir,
    dataset_names=("left_pressure", "right_pressure"),
    cmap="viridis",
    target_size=(480, 480),
    max_value=3072.0,
    num_intervals=5,
    method="binning",
    gaussian_sigma=0.0,
    temporal_alpha=0.2,
):
    """Convert tactile pressure samples into discrete-interval heatmap images (PNG).

    Args:
        file_path (str): Path to the source HDF5 file.
        demo_id (str): Demo identifier under the `data` group.
        output_dir (str): Directory where heatmaps will be written.
        dataset_names (Iterable[str]): Names of tactile datasets to export.
        cmap (str): Matplotlib colormap name (fallback to grayscale if unavailable).
        target_size (tuple[int, int] | None): Resize each image to this size.
        max_value (float): Maximum tactile value expected; values above are clipped.
        num_intervals (int): Number of discrete intervals to map across the range (3, 6, or 9).
        method (str): Processing method ('binning', 'linear', or 'log').
        gaussian_sigma (float): Spatial Gaussian smoothing sigma in pixels. <=0 disables.
        temporal_alpha (float): Temporal smoothing factor (0-1). <=0 disables.
        
    Raises:
        KeyError: If demo is not found
    """
    try:
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap)
    except ImportError:
        colormap = None

    max_value = float(max_value) if max_value is not None else 3072.0
    if max_value <= 0.0:
        max_value = 3072.0
    num_intervals = int(num_intervals) if num_intervals else 0
    if num_intervals < 2:
        num_intervals = 5

    with h5py.File(file_path, "r") as src:
        data_group = _require_data_group(src, file_path)
        resolved_demo = _resolve_demo_name(data_group, demo_id)
        if resolved_demo is None:
            available = ", ".join(list(data_group.keys())[:5])
            raise KeyError(f"Demo '{demo_id}' not found in '{file_path}'. Available demos include: {available}")

        demo_prefix = f"data/{resolved_demo}"
        for name in dataset_names:
            dataset_path = f"{demo_prefix}/{name}"
            if dataset_path not in src:
                continue

            dset = src[dataset_path]
            if not isinstance(dset, h5py.Dataset):
                continue

            target_dir = os.path.join(output_dir, name)
            os.makedirs(target_dir, exist_ok=True)

            prev_frame = None
            for idx, sample in enumerate(dset):
                frame = _prepare_tactile_frame(sample, dset.attrs)

                if gaussian_sigma and gaussian_sigma > 0.0:
                    frame = _apply_gaussian(frame, gaussian_sigma)

                if (
                    prev_frame is not None
                    and temporal_alpha is not None
                    and 0.0 < temporal_alpha < 1.0
                ):
                    frame = temporal_alpha * frame + (1.0 - temporal_alpha) * prev_frame
                prev_frame = frame

                image = _render_heatmap(
                    frame,
                    colormap,
                    max_value=max_value,
                    num_intervals=num_intervals,
                    method=method,
                )
                if target_size:
                    image = image.resize(target_size, _HEATMAP_RESAMPLE)
                filename = os.path.join(target_dir, f"{demo_id}_{idx:05d}.png")
                image.save(filename)

def export_tactile_voronoi(
    file_path,
    demo_id,
    output_dir,
    dataset_names=("left_pressure", "right_pressure"),
    cmap="viridis",                 # kept for API compatibility (unused)
    target_size=(480, 480),
    max_value=3072.0,
    num_intervals=5,                # kept for API compatibility (unused)
    method="binning",               # kept for API compatibility (unused)
    gaussian_sigma=0.2,             # extra smoothing at the very end (optional)
    temporal_alpha=0.2,             # optional temporal smoothing (per-hand)
):
    """
    Export tactile frames rendered like visualize_tactile_realtime:
      1) Reweight 16x16 taxels using precomputed kernels (voronoi labels)
      2) Fill voronoi polygons with value-to-color
      3) Apply masked Gaussian blending (no bleeding to background)
      4) (Optional) Temporal smoothing per stream

    Signature, directory structure, and file naming match export_tactile_heatmaps.
    """
    import json, cv2
    from xml.etree import ElementTree as ET

    # --- Locate mapping + SVG like the realtime script does ---
    def _find_first_existing(cands):
        for p in cands:
            if p and os.path.exists(p):
                return p
        return None

    mapping_json = _find_first_existing([
        "point_weight_mappings_large.json",
        # "point_weight_mappings_small.json",
        os.path.join(os.path.dirname(__file__), "data", "point_weight_mappings_large.json"),
        # os.path.join(os.path.dirname(__file__), "data", "point_weight_mappings_small.json"),
    ])
    svg_file = _find_first_existing([
        "voronoi_regions_large.svg",
        # "voronoi_regions_small.svg",
        os.path.join(os.path.dirname(__file__), "data", "voronoi_regions_large.svg"),
        # os.path.join(os.path.dirname(__file__), "data", "voronoi_regions_small.svg"),
    ])
    if mapping_json is None or svg_file is None:
        raise FileNotFoundError(
            "Could not find mapping JSON or Voronoi SVG. "
            "Expected one of {point_weight_mappings_[small|large].json} and {voronoi_regions_[small|large].svg}."
        )

    mapping = json.load(open(mapping_json, "r"))

    # --- Parse Voronoi polygons from SVG (id -> [(x,y), ...]) ---
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    voronoi_polygons = {
        poly.attrib['id']: [
            tuple(map(float, p.split(','))) for p in poly.attrib['points'].strip().split()
        ]
        for poly in root.findall('.//svg:polygon', ns)
    }

    # --- Geometry extents + shift to positive canvas with margin ---
    all_points = np.array([pt for pts in voronoi_polygons.values() for pt in pts], dtype=np.float32)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    overlay_width  = int(max_x - min_x) + 20
    overlay_height = int(max_y - min_y) + 20
    shift = np.array([10 - min_x, 10 - min_y], dtype=np.float32)

    polygon_pts_shifted = {
        label: np.array([(x + shift[0], y + shift[1]) for (x, y) in pts], dtype=np.int32)
        for label, pts in voronoi_polygons.items()
    }

    # --- Weight kernels (copied/adapted from realtime) ---
    def _precompute_weights(_mapping, is_left=True):
        W = {}
        for label, neighbors in _mapping.items():
            mat = np.zeros((16, 16), dtype=np.float32)
            total = 0.0
            for q in ['NE', 'NW', 'SW', 'SE']:
                src, dist = neighbors.get(q, ('N/A', 0.0))
                if src != 'N/A':
                    y, x = map(int, src.split('-'))
                    if not is_left:
                        # mirror for right as in realtime
                        y, x = 15 - x, 15 - y
                    w = 1.0 / dist if dist and dist > 1e-3 else 1e6
                    mat[y, x] += w
                    total += w
            if total > 0:
                mat /= total
            W[label] = mat
        return W

    left_weights  = _precompute_weights(mapping, is_left=True)
    right_weights = _precompute_weights(mapping, is_left=False)

    def _interpolate_fast(pressure16x16, precomputed):
        return {label: float(np.sum(W * pressure16x16)) for label, W in precomputed.items()}

    # --- Color mapping (HSV→BGR for OpenCV; clamp using max_value) ---
    def _hsv_to_bgr(h, s, v):
        h = h % 360.0
        c = v * s
        x = c * (1 - abs((h / 60.0) % 2 - 1))
        m = v - c
        if h < 60:   rp, gp, bp = c, x, 0
        elif h < 120: rp, gp, bp = x, c, 0
        elif h < 180: rp, gp, bp = 0, c, x
        elif h < 240: rp, gp, bp = 0, x, c
        elif h < 300: rp, gp, bp = x, 0, c
        else:         rp, gp, bp = c, 0, x
        return (int((bp + m) * 255), int((gp + m) * 255), int((rp + m) * 255))

    def _value_to_color(v, vmin=0.0, vmax=3072.0):
        clamped = float(np.clip(v, vmin, vmax))
        norm = (clamped - vmin) / max(vmax - vmin, 1e-6)
        hue = norm * 240.0   # 0=red -> 240=blue (same convention)
        return _hsv_to_bgr(hue, 1.0, 1.0)

    # --- Masked Gaussian blur (avoid bleeding outside polygons) ---
    def _compute_kernel_size(w, h):
        base = int(max(15, 0.03 * min(w, h)))
        return 3
        return base + 1 if base % 2 == 0 else base

    def _masked_gaussian_blur(img_bgr, mask_binary, ksize):
        img  = img_bgr.astype(np.float32) / 255.0
        mask = (mask_binary.astype(np.float32) / 255.0)
        weighted = cv2.GaussianBlur(img * mask[..., None], (ksize, ksize), 0)
        norm = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        norm = np.maximum(norm, 1e-6)[..., None]
        blended = weighted / norm
        mask3 = (mask > 0.5).astype(np.float32)[..., None]
        out = img * (1.0 - mask3) + blended * mask3
        return (out * 255.0).clip(0, 255).astype(np.uint8)

    # --- Single-hand overlay (returns RGB PIL image if available, else BGR numpy) ---
    def _render_hand_overlay(interp_dict, mirror_left, target_size_xy, vmax):
        canvas = np.ones((overlay_height, overlay_width, 3), dtype=np.uint8) * 255

        for label, pts in polygon_pts_shifted.items():
            color = _value_to_color(interp_dict.get(label, 0.0), vmin=0.0, vmax=vmax)
            cv2.fillPoly(canvas, [pts], color)
        mask = (canvas != 255).any(axis=2).astype(np.uint8) * 255
        ksize = _compute_kernel_size(overlay_width, overlay_height)
        canvas = _masked_gaussian_blur(canvas, mask, ksize)

        if mirror_left:
            canvas = cv2.flip(canvas, 1)  # horizontal flip

        # Resize to requested target_size and convert to PIL if available
        canvas = cv2.resize(canvas, (int(target_size_xy[0]), int(target_size_xy[1])), interpolation=cv2.INTER_CUBIC)

        if Image is not None:
            # Convert BGR→RGB for saving with PIL
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb, mode="RGB")
        return canvas  # BGR numpy array

    # --- Read HDF5, iterate frames per dataset like export_tactile_heatmaps ---
    vmax = float(max_value) if max_value and max_value > 0 else 3072.0
    width, height = target_size

    with h5py.File(file_path, "r") as src:
        data_group = _require_data_group(src, file_path)
        resolved = _resolve_demo_name(data_group, demo_id)
        if resolved is None:
            available = ", ".join(list(data_group.keys())[:5])
            raise KeyError(f"Demo '{demo_id}' not found in '{file_path}'. Available demos include: {available}")

        # Maintain a simple per-dataset temporal buffer
        prev_interps = {}

        for name in dataset_names:
            dpath = f"data/{resolved}/{name}"
            if dpath not in src:
                continue
            dset = src[dpath]
            if not isinstance(dset, h5py.Dataset):
                continue

            is_left = "left" in name.lower()
            W = left_weights if is_left else right_weights

            out_dir = os.path.join(output_dir, name)
            os.makedirs(out_dir, exist_ok=True)

            prev_map = None
            for idx, sample in enumerate(dset):
                # Prepare 16×16 pressure grid (reuses your helper)
                grid = _prepare_tactile_frame(sample, dset.attrs)
                # Ensure 16x16 float32
                grid = np.asarray(grid, dtype=np.float32)
                if grid.shape != (16, 16):
                    # Best-effort reshape if metadata mismatch
                    grid = cv2.resize(grid, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)

                # Interpolate onto voronoi labels
                interp = _interpolate_fast(grid, W)

                # Temporal smoothing on interpolated scalar field (per label)
                if prev_map is not None and temporal_alpha and 0.0 < temporal_alpha < 1.0:
                    for k in interp.keys():
                        p = prev_map.get(k, interp[k])
                        interp[k] = temporal_alpha * interp[k] + (1.0 - temporal_alpha) * p
                prev_map = interp
                
                # Render single-hand overlay in the realtime style
                img_obj = _render_hand_overlay(
                    interp_dict=interp,
                    mirror_left=is_left,                     # realtime flips left
                    target_size_xy=(width, height),
                    vmax=vmax,
                )

                # Save
                fname = os.path.join(out_dir, f"{demo_id}_{idx:05d}.png")
                if Image is not None and isinstance(img_obj, Image.Image):
                    img_obj.save(fname, format="PNG")
                else:
                    # Fallback if PIL missing: img_obj is BGR numpy
                    cv2.imwrite(fname, img_obj)

def export_tactile_layout(
    file_path,
    demo_id,
    output_dir,
    dataset_names=("left_pressure", "right_pressure"),
    cmap="viridis",                 # kept for API compatibility (unused)
    target_size=(480, 480),
    max_value=3072.0,
    num_intervals=5,                # kept for API compatibility (unused)
    method="binning",               # kept for API compatibility (unused)
    gaussian_sigma=0.0,             # not used here (kept for API compatibility)
    temporal_alpha=0.0,             # optional per-frame EMA on grid before drawing
):
    """
    Export frames by drawing each valid taxel at its 2D layout position
    (the 'layout map' renderer). Naming, dirs, and signature mirror
    export_tactile_heatmaps/export_tactile_voronoi.
    """
    import os, json, cv2, numpy as np

    # --- helpers (nested) -----------------------------------------------------
    def _find_first_existing(cands):
        for p in cands:
            if p and os.path.exists(p):
                return p
        return None

    def _load_layout_json():
        # expects {"positions": {"r-c": {"x":..,"y":.., "mano_faceid":[...]}, ...},
        #          "erasedNodes": ["r-c", ...]}
        layout_json = _find_first_existing([
            "handLayoutNewest.json",
            os.path.join(os.path.dirname(__file__), "data", "handLayoutNewest.json"),
            os.path.join(os.path.dirname(__file__), "handLayoutNewest.json"),
        ])
        if layout_json is None:
            raise FileNotFoundError(
                "Could not find handLayoutNewest.json (layout). "
                "Place it next to load_data.py or in ./data/."
            )
        with open(layout_json, "r") as f:
            d = json.load(f)
        layout = d["positions"]
        erased = set(d.get("erasedNodes", []))
        return layout, erased

    def _draw_layout_frame(pressure16, layout, erased_nodes, canvas_size, vmin, vmax):
        Ht, Wt = canvas_size[1], canvas_size[0]

        # normalize on global min/max
        norm = ((pressure16 - vmin) / max(vmax - vmin, 1e-6)).clip(0, 1)

        # collect valid xy and also compute a nice aspect-preserving scale
        valid = {
            nid: (float(pos["x"]), float(pos["y"]))
            for nid, pos in layout.items()
            if nid not in erased_nodes
        }
        xs = [x for (x, y) in valid.values()]
        ys = [y for (x, y) in valid.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 20.0
        Wsrc = (max_x - min_x) + 2 * pad
        Hsrc = (max_y - min_y) + 2 * pad

        # fit into target canvas
        scale = min(Wt / Wsrc, Ht / Hsrc)
        ox = (Wt - scale * Wsrc) * 0.5
        oy = (Ht - scale * Hsrc) * 0.5

        canvas = np.zeros((Ht, Wt, 3), dtype=np.uint8)
        for nid, (x, y) in valid.items():
            r, c = map(int, nid.split('-'))
            val = float(norm[r, c])
            # viridis via OpenCV
            color = cv2.applyColorMap(
                np.array([[int(val * 255)]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS
            )[0, 0]
            px = int(ox + scale * (x - min_x + pad))
            py = int(oy + scale * (y - min_y + pad))
            cv2.circle(canvas, (px, py), radius=max(3, int(3 * scale)), color=tuple(int(k) for k in color), thickness=-1)

        return canvas

    # --- pipeline -------------------------------------------------------------
    layout, erased_nodes = _load_layout_json()
    width, height = target_size

    import h5py
    with h5py.File(file_path, "r") as src:
        data_group = _require_data_group(src, file_path)
        resolved = _resolve_demo_name(data_group, demo_id)
        if resolved is None:
            raise KeyError(f"Demo '{demo_id}' not found in '{file_path}'.")

        for name in dataset_names:
            dpath = f"data/{resolved}/{name}"
            if dpath not in src or not isinstance(src[dpath], h5py.Dataset):
                continue
            dset = src[dpath]

            # Compute global vmin/vmax over valid nodes for consistent colors
            vals = []
            for sample in dset:
                grid = _prepare_tactile_frame(sample, dset.attrs)
                grid = np.asarray(grid, dtype=np.float32)
                if grid.shape != (16, 16):
                    grid = cv2.resize(grid, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
                vals.append(grid)
            stack = np.stack(vals, 0) if vals else np.zeros((0, 16, 16), np.float32)

            # mask only nodes present in layout & not erased
            valid_mask = np.zeros((16, 16), dtype=bool)
            for nid in layout.keys():
                if nid in erased_nodes:
                    continue
                r, c = map(int, nid.split('-'))
                valid_mask[r, c] = True

            if stack.size:
                vmin = float(stack[:, valid_mask].min())
                vmax = float(stack[:, valid_mask].max())
            else:
                vmin, vmax = 0.0, 1.0

            # optional temporal EMA
            prev = None

            out_dir = os.path.join(output_dir, name)
            os.makedirs(out_dir, exist_ok=True)

            for idx, grid in enumerate(vals):
                if temporal_alpha and prev is not None:
                    grid = temporal_alpha * grid + (1.0 - temporal_alpha) * prev
                prev = grid

                img = _draw_layout_frame(grid, layout, erased_nodes, (width, height), vmin, vmax)
                cv2.imwrite(os.path.join(out_dir, f"{demo_id}_{idx:05d}.png"), img)

    return True

def export_tactile_mano(
    file_path,
    demo_id,
    output_dir,
    dataset_names=("left_pressure", "right_pressure"),
    cmap="viridis",
    target_size=(480, 480),
    max_value=3072.0,
    num_intervals=5,
    method="binning",
    gaussian_sigma=0.0,
    temporal_alpha=0.4,
    index=[],                      # <<< NEW: a single index or an iterable of indices
):
    """
    Export frames by projecting taxels onto a MANO mesh and rendering with pyrender.
    Also export selected mesh frames (vertex/face/color) to OBJ when idx in `index`.
    """
    import os, json, cv2, numpy as np, h5py, trimesh
    from matplotlib import cm
    from pyrenderer import ManoRenderer

    # --- small helpers -------------------------------------------------------
    def _as_set(x):
        # accept int or iterable
        if isinstance(x, int):
            return {x}
        try:
            return set(x)
        except Exception:
            return {int(x)}

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
            edges.update({(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(c,a), max(c,a))})
        for i, j in edges:
            dij = np.linalg.norm(verts[i] - verts[j])
            nbrs[i].append(j); dists[i].append(dij)
            nbrs[j].append(i); dists[j].append(dij)
        return nbrs, dists

    def _gaussian_smooth_vertex_signal(vals, nbrs, dists, sigma=0.005, iters=2):
        if sigma <= 0 or iters <= 0: 
            return vals
        two_sig2 = 2.0 * (sigma * sigma)
        out = vals.astype(np.float32).copy()
        for _ in range(iters):
            new = out.copy()
            for i, (N, D) in enumerate(zip(nbrs, dists)):
                acc = out[i]; w_sum = 1.0
                for j, dij in zip(N, D):
                    w = np.exp(-(dij*dij)/two_sig2)
                    acc += w * out[j]; w_sum += w
                new[i] = acc / max(w_sum, 1e-8)
            out = new
        return out

    def _write_obj_with_vertex_colors(path, vertices, faces, vertex_colors_rgba):
        """Writes OBJ with per-vertex colors as `v x y z r g b` (0-255)."""
        vc = vertex_colors_rgba
        mask = np.all(vc == 0, axis=1)
        vc[mask] = [44/255, 44/255, 44/255, 255/255]
        # if vc.dtype != np.uint8:
        #     vc = np.clip(np.rint(vc*255.0), 0, 255).astype(np.uint8) if vc.max() <= 1.0 else vc.astype(np.uint8)
        vc = np.clip(vc*255.0, 0, 255)
        rgb = vc[:, :3]
        with open(path, "w") as f:
            f.write("# OBJ with per-vertex colors (r g b after xyz)\n")
            for (x, y, z), (r, g, b) in zip(vertices, rgb):
                f.write(f"v {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            for a, b, c in faces.astype(int):
                f.write(f"f {a+1} {b+1} {c+1}\n")

    # --- the per-frame renderer; returns both image and vertex colors --------
    def _render_pressure_mano(mano_vertices, mano_faces, renderer, pressure16, layout, erased_nodes, vmin, vmax, nbrs, dists):
        from collections import defaultdict
        # normalize pressure → [0,1]
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
            known_mask = np.zeros(n_verts, bool); known_mask[list(vert_to_vals.keys())] = True
            vert_max = float(vert_vals[known_mask].max())
            vert_vals[~known_mask] = vert_max

        # connectivity smoothing
        vert_vals = _gaussian_smooth_vertex_signal(vert_vals, nbrs, dists, sigma=0.005, iters=2)

        # invert + min-max normalize
        mn, mx = float(vert_vals.min()), float(vert_vals.max())
        if mx > mn:
            vert_vals = 1.0 - (vert_vals - mn) / (mx - mn)
        else:
            vert_vals[:] = 1.0
        # optional thresholding (example): low values to 0
        # vert_vals[vert_vals < 0.1] = 0.0

        # colormap_fn = lambda x: np.array(cm.jet(x))
        colormap_fn = lambda x: np.array(cm.gnuplot2(x))
        vertex_colors = colormap_fn(vert_vals)  # RGBA float in [0,1]
        img_rgb = renderer.render(vertex_colors=vertex_colors, colormap_fn=colormap_fn, smooth=True)
        return img_rgb[:, :, ::-1], vertex_colors

    # --- pipeline -------------------------------------------------------------
    selected = _as_set(index)
    layout, erased_nodes = _load_layout_json()
    width, height = target_size

    with h5py.File(file_path, "r") as src:
        data_group = _require_data_group(src, file_path)
        resolved = _resolve_demo_name(data_group, demo_id)
        if resolved is None:
            raise KeyError(f"Demo '{demo_id}' not found in '{file_path}'.")

        # MANO from OBJ
        obj_path = _find_first_existing([
            os.path.join("data", "mano_right_neutral_subdiv.obj"),
            os.path.join(os.path.dirname(__file__), "data", "mano_right_neutral_subdiv.obj"),
            os.path.join(os.path.dirname(__file__), "scratch", "mano_right_neutral_subdiv.obj"),
        ])
        if obj_path is None:
            raise FileNotFoundError("Missing mano_right_neutral_subdiv.obj")
        mesh = trimesh.load(obj_path, process=False)
        mano_vertices = np.asarray(mesh.vertices, dtype=np.float32)
        mano_faces    = np.asarray(mesh.faces, dtype=np.int32)

        # precompute connectivity once
        nbrs, dists = _build_vertex_graph(mano_vertices, mano_faces)

        renderer = ManoRenderer(image_size=(width, height),
                                mano_vertices=mano_vertices,
                                mano_faces=mano_faces)

        for name in dataset_names:
            dpath = f"data/{resolved}/{name}"
            if dpath not in src or not isinstance(src[dpath], h5py.Dataset):
                continue
            dset = src[dpath]

            # gather all frames to set global vmin/vmax on valid nodes
            vals = []
            for sample in dset:
                grid = _prepare_tactile_frame(sample, dset.attrs).astype(np.float32)
                if grid.shape != (16, 16):
                    grid = cv2.resize(grid, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
                vals.append(grid)
            stack = np.stack(vals, 0) if vals else np.zeros((0, 16, 16), np.float32)

            valid_mask = np.zeros((16, 16), dtype=bool)
            for nid in layout.keys():
                if nid in erased_nodes: continue
                r, c = map(int, nid.split('-'))
                valid_mask[r, c] = True

            if stack.size:
                vmin = float(stack[:, valid_mask].min())
                vmax = float(stack[:, valid_mask].max())
            else:
                vmin, vmax = 0.0, 1.0

            prev = None
            out_dir = os.path.join(output_dir, name)
            os.makedirs(out_dir, exist_ok=True)

            for idx, grid in enumerate(vals):
                if temporal_alpha and prev is not None:
                    grid = temporal_alpha * grid + (1.0 - temporal_alpha) * prev
                prev = grid

                img_bgr, vcolors = _render_pressure_mano(
                    mano_vertices, mano_faces, renderer, grid, layout, erased_nodes, vmin, vmax, nbrs, dists
                )
                # save frame image
                alpha = 1.2   # contrast factor (1.0 = no change, >1 increases contrast)
                beta  = 0.1 * 255  # brightness shift; 20% of 255 ≈ 51

                img_adj = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
                cv2.imwrite(os.path.join(out_dir, f"{demo_id}_{idx:05d}.png"), img_adj)

                # # export mesh snapshot with per-vertex colors (OBJ) if requested
                # if idx in selected:
                #     snap_dir = os.path.join(out_dir, "mesh_snapshots")
                #     os.makedirs(snap_dir, exist_ok=True)
                #     # obj_out = os.path.join(snap_dir, f"{demo_id}_{idx:05d}_mesh.obj")
                #     # _write_obj_with_vertex_colors(obj_out, mano_vertices, mano_faces, vcolors)

    return True


def export_pose_visualizations(
    file_path,
    demo_id,
    output_dir,
    dataset_names=("left_hand_landmarks", "right_hand_landmarks"),
    target_size=(480, 480),
    background_color=(255, 255, 255),
    point_radius=3,
    line_width=3,
    grid_lines=0,
):
    """
    Render pose landmark sequences to individual image directories per dataset.

    Args:
        file_path (str): Path to the source HDF5 file.
        demo_id (str): Demo identifier under the `data` group.
        output_dir (str): Root directory where pose visualizations will be written.
        dataset_names (Iterable[str]): Landmark datasets to visualise.
        target_size (tuple[int, int]): Output image resolution.
        background_color (tuple[int, int, int]): RGB background colour.
        point_radius (int): Radius (in pixels) for landmark markers.
        line_width (int): Line width (in pixels) for skeleton connections.
        grid_lines (int): Number of grid divisions for the overlay (both axes).

    Returns:
        dict[str, bool]: Mapping of dataset name to True when frames were exported.
    """
    width, height = target_size

    with h5py.File(file_path, "r") as src:
        data_group = _require_data_group(src, file_path)
        resolved_demo = _resolve_demo_name(data_group, demo_id)
        if resolved_demo is None:
            available = ", ".join(list(data_group.keys())[:5])
            raise KeyError(
                f"Demo '{demo_id}' not found in '{file_path}'. Available demos include: {available}"
            )

        sequences: dict[str, np.ndarray] = {}
        for name in dataset_names:
            dataset_path = f"data/{resolved_demo}/{name}"
            if dataset_path not in src:
                continue
            dset = src[dataset_path]
            if not isinstance(dset, h5py.Dataset):
                continue
            sequences[name] = dset[()].astype(np.float32, copy=False)

    if not sequences:
        return {}

    def _canonicalize_frame(points: np.ndarray, mirror: bool) -> np.ndarray | None:
        coords = np.asarray(points, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[0] < 3:
            return None

        if coords.shape[1] < 3:
            padded = np.zeros((coords.shape[0], 3), dtype=np.float32)
            padded[:, : coords.shape[1]] = coords
            coords = padded

        if not np.isfinite(coords).all():
            return None

        wrist = coords[0]
        index_mcp = coords[5] if coords.shape[0] > 5 else coords[0]
        pinky_mcp = coords[17] if coords.shape[0] > 17 else coords[-1]
        middle_mcp = coords[9] if coords.shape[0] > 9 else coords[-1]
        ring_mcp = coords[13] if coords.shape[0] > 13 else coords[-1]
        middle_tip = coords[12] if coords.shape[0] > 12 else coords[-1]

        x_axis = index_mcp - pinky_mcp
        if mirror:
            x_axis = -x_axis
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)

        tip_vectors = []
        for idx in (8, 12, 16, 20):
            if coords.shape[0] > idx:
                tip_vectors.append(coords[idx] - wrist)
        tip_mean = (
            np.mean(tip_vectors, axis=0)
            if tip_vectors
            else coords[-1] - wrist
        )

        y_dir = ((middle_mcp + ring_mcp) * 0.5) - wrist
        y_dir = y_dir - np.dot(y_dir, x_axis) * x_axis
        if np.linalg.norm(y_dir) < 1e-6:
            y_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            y_dir = y_dir / np.linalg.norm(y_dir)

        if np.linalg.norm(tip_mean) > 1e-6 and np.dot(tip_mean, y_dir) < 0.0:
            y_dir = -y_dir

        z_axis = np.cross(x_axis, y_dir)
        if np.linalg.norm(z_axis) < 1e-6:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            z_axis = z_axis / np.linalg.norm(z_axis)

        if np.dot(middle_tip - wrist, z_axis) < 0.0:
            z_axis = -z_axis
            y_dir = -y_dir

        rotation = np.stack([x_axis, y_dir, z_axis], axis=1)
        canonical = (coords - wrist) @ rotation
        return canonical

    canonical_sequences: dict[str, list[np.ndarray | None]] = {}
    canonical_xy_points: list[np.ndarray] = []

    for name, arr in sequences.items():
        mirror = "left" in name
        frames: list[np.ndarray | None] = []
        if arr.ndim != 3:
            canonical_sequences[name] = frames
            continue
        for frame in arr:
            canon = _canonicalize_frame(frame, mirror)
            frames.append(canon)
            if canon is not None:
                canonical_xy_points.append(canon[:, :2])
        canonical_sequences[name] = frames

    if not canonical_xy_points:
        return {}

    all_points = np.concatenate(canonical_xy_points, axis=0)
    min_xy = np.nanmin(all_points, axis=0)
    max_xy = np.nanmax(all_points, axis=0)

    if not np.isfinite(min_xy).all() or not np.isfinite(max_xy).all():
        return {}

    span = max(np.max(max_xy - min_xy), 1e-3)
    half_span = span * 0.65  # add ~30% margin around the extremal points
    center = (min_xy + max_xy) * 0.5
    bounds_min = center - half_span
    bounds_max = center + half_span

    denom = bounds_max - bounds_min
    denom[denom == 0.0] = 1e-3

    os.makedirs(output_dir, exist_ok=True)

    from PIL import Image, ImageDraw
    trail_length = 6
    trail_buffers = {
        name: deque(maxlen=trail_length) for name in sequences.keys()
    }

    results: dict[str, bool] = {}
    num_frames = max(arr.shape[0] for arr in sequences.values())

    for name, arr in sequences.items():
        target_dir = os.path.join(output_dir, name)
        os.makedirs(target_dir, exist_ok=True)

        label = "Left Hand" if "left" in name else "Right Hand"
        base_color = _POSE_COLORS.get(name, (200, 200, 200))
        trail = trail_buffers[name]
        canonical_frames = canonical_sequences.get(name, [])

        exported = False
        for frame_idx in range(num_frames):
            canvas = Image.new("RGBA", target_size, background_color + (255,))
            draw = ImageDraw.Draw(canvas)

            canonical = canonical_frames[frame_idx] if frame_idx < len(canonical_frames) else None
            pixel_points: list[tuple[float, float] | None] = []
            if canonical is not None:
                for point in canonical[:, :2]:
                    if not np.isfinite(point).all():
                        pixel_points.append(None)
                        continue
                    norm = (point - bounds_min) / denom
                    norm = np.clip(norm, 0.0, 1.0)
                    px = float(norm[0]) * (width - 1)
                    py = (1.0 - float(norm[1])) * (height - 1)
                    pixel_points.append((px, py))
            else:
                pixel_points = [None] * 21

            trail.append(
                [
                    None if p is None else (float(p[0]), float(p[1]))
                    for p in pixel_points
                ]
            )

            if len(trail) >= 2:
                overlay = Image.new("RGBA", target_size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                for depth in range(len(trail) - 1):
                    alpha = int(160 * (depth + 1) / len(trail)) + 40
                    segment_color = base_color + (max(0, min(255, alpha)),)
                    prev_points = trail[depth]
                    next_points = trail[depth + 1]
                    for p_prev, p_curr in zip(prev_points, next_points):
                        if p_prev is None or p_curr is None:
                            continue
                        overlay_draw.line(
                            (p_prev[0], p_prev[1], p_curr[0], p_curr[1]),
                            fill=segment_color,
                            width=max(1, line_width - 1),
                        )
                canvas = Image.alpha_composite(canvas, overlay)
                draw = ImageDraw.Draw(canvas)

            for start, end in _HAND_CONNECTIONS:
                if start >= len(pixel_points) or end >= len(pixel_points):
                    continue
                p0 = pixel_points[start]
                p1 = pixel_points[end]
                if p0 is None or p1 is None:
                    continue
                draw.line(
                    (p0[0], p0[1], p1[0], p1[1]),
                    fill=base_color,
                    width=line_width,
                )

            radius_outer = float(point_radius) * 1.6
            radius_inner = float(point_radius)
            outer_colour = tuple(max(0, int(c * 0.3)) for c in base_color)
            inner_colour = tuple(int(min(255, c + 30)) for c in base_color)

            for idx, p in enumerate(pixel_points):
                if p is None:
                    continue
                x, y = p
                ro = radius_outer * (1.6 if idx == 0 else 1.0)
                ri = radius_inner * (1.4 if idx == 0 else 1.0)
                draw.ellipse(
                    (x - ro, y - ro, x + ro, y + ro),
                    fill=outer_colour,
                )
                draw.ellipse(
                    (x - ri, y - ri, x + ri, y + ri),
                    fill=inner_colour,
                )

            frame_filename = f"{demo_id}_{frame_idx:05d}.png"
            frame_path = os.path.join(target_dir, frame_filename)
            canvas.convert("RGB").save(frame_path, format="PNG")
            exported = True

        results[name] = exported

    return results

def export_pose_mano(
    file_path,
    demo_id,
    output_dir,
    dataset_names=("left_hand_landmarks", "right_hand_landmarks"),
    target_size=(480, 480),
    background_color=(249, 235, 142),
    mano_model_root=None,
    mano_side="right",
    use_cuda=True,
    selected_frame_indices=None,
):
    import os, math, json
    import numpy as np, h5py, cv2, trimesh, pyrender
    # ---------------- helpers you already had (trimmed for space) ------------
    def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p
    def _pick_side(dataset_name, default="right"):
        return "right"  # your current hard-code
    # Patch numpy for chumpy compatibility (numpy >=1.24 removed these aliases)
    import numpy as _np
    for _attr in ("bool", "int", "float", "complex", "object", "unicode", "str"):
        if not hasattr(_np, _attr):
            setattr(_np, _attr, getattr(__builtins__, _attr, object))
    from easymocap.smplmodel.body_model import SMPLlayer
    import torch
    def _resolve_file(filename):
        _here = os.path.dirname(__file__)
        for candidate in [
            os.path.join("data", filename),
            os.path.join(_here, "data", filename),
            os.path.join(_here, "scratch", filename),
            os.path.join(_here, "..", "EasyMocap", "data", "smplx", filename),
        ]:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"Missing {filename}")

    def _load_mano_model(side, model_root, use_cuda=True, **kwargs):
        device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")
        lr = {'left': 'LEFT', 'right': 'RIGHT'}
        pkl_path = _resolve_file(f'MANO_{lr[side]}.pkl')
        reg_path = _resolve_file(f'J_regressor_mano_{lr[side]}.txt')
        body_model = SMPLlayer(pkl_path,
                               model_type='mano', gender='neutral',
                               device=device,
                               regressor_path=reg_path,
                               **kwargs).to(device)
        return body_model, device
    def _fit_sequence_to_mano(body_model, keypoints3d_seq):
        from easymocap.pipeline import smpl_from_keypoints3d
        from easymocap.dataset import CONFIG
        T = keypoints3d_seq.shape[0]
        kp = np.concatenate([keypoints3d_seq.astype(np.float32),
                             np.ones((T, keypoints3d_seq.shape[1], 1), np.float32)], axis=-1)
        class _Args: pass
        args = _Args(); args.robust3d=False; args.verbose=False; args.model='mano'
        w_pose = {'k3d':1e2,'k2d':1e-8,'reg_poses':1e-1,'smooth_body':1e2,'smooth_poses':1e2}
        return smpl_from_keypoints3d(body_model, kp, config=CONFIG['handr'],
                                     args=args,
                                     weight_shape={'s3d':1e6,'reg_shapes':1e1},
                                     weight_pose=w_pose)

    def nudge_thumb_lateral_3j(arr, k_tip=0.08, k_ip=0.05, k_mcp=0.03):
        """
        Lateral-only nudge for thumb MCP(2), IP(3), TIP(4) along palm axis
        (pinky MCP -> index MCP). Scales by palm width each frame.
        arr: (T,21,3)
        """
        import numpy as np
        T = arr.shape[0]
        for t in range(T):
            c = arr[t]

            wrist     = c[0]
            idx_mcp   = c[5]  if c.shape[0] > 5  else wrist
            pky_mcp   = c[17] if c.shape[0] > 17 else wrist
            mid_mcp   = c[9]  if c.shape[0] > 9  else wrist
            th_mcp    = c[2]  if c.shape[0] > 2  else wrist
            th_ip     = c[3]  if c.shape[0] > 3  else wrist
            th_tip    = c[4]  if c.shape[0] > 4  else wrist

            # Palm lateral axis: pinky -> index
            x_axis = idx_mcp - pky_mcp
            w = np.linalg.norm(x_axis)
            if w < 1e-8:
                x_axis = np.array([1.,0.,0.], np.float32); w = 1.0
            else:
                x_axis = x_axis / w

            # (not used for displacement; here just for clarity)
            fwd = (c[12] if c.shape[0] > 12 else mid_mcp) - wrist
            _ = np.cross(x_axis, fwd)  # palm normal (ignored)

            # Pure lateral offsets (scale by palm width)
            d_tip = x_axis * (k_tip * w)
            d_ip  = x_axis * (k_ip  * w)
            d_mcp = x_axis * (k_mcp * w)

            # Apply
            c[4] = th_tip + d_tip      # tip
            c[3] = th_ip  + d_ip       # under the tip (IP)
            c[2] = th_mcp + d_mcp      # second under (MCP)

            arr[t] = c
        return arr
    # ---------------- tiny helper for OBJ export ----------------
    def _export_obj(verts, faces, path):
        tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        tri.export(path)

    # -------------------------- one-time renderer ----------------------------
    class RenderOnce:
        def __init__(self, faces, image_size=(480,480), bg_rgb=(249, 235, 142)):
            W, H = image_size
            self.W, self.H = W, H
            self.scene = pyrender.Scene(bg_color=[*bg_rgb, 0], ambient_light=[0.25, 0.25, 0.25])
            # camera
            cam = pyrender.IntrinsicsCamera(fx=8000, fy=8000, cx=W/2, cy=H/2, zfar=1e12)
            T = np.eye(4); T[:3,3] = [0,0,2.0]
            self.scene.add(cam, pose=T)
            # a few directional lights
            def _dir(theta, phi, inten):
                T = np.eye(4); r=4.0
                th=np.radians(theta); ph=np.radians(phi)
                pos=np.array([r*np.sin(th)*np.cos(ph), r*np.sin(th)*np.sin(ph), r*np.cos(th)])
                z=-pos/np.linalg.norm(pos); x=np.array([-z[1],z[0],0.]); 
                if np.linalg.norm(x)==0: x=np.array([1.,0.,0.])
                x/=np.linalg.norm(x); y=np.cross(z,x); T[:3,:3]=np.stack([x,y,z],1); T[:3,3]=pos
                lightnode = pyrender.Node(light=pyrender.DirectionalLight(intensity=inten), matrix=T)
                self.scene.add_node(lightnode)
            _dir(40,   0, 3.0); _dir(65, 120, 2.0); _dir(70, -120, 2.0)
            # static faces, material (shiny), placeholder positions
            self.faces = faces.astype(np.int32)
            self.material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(153/255, 41/255, 234/255, 1.), metallicFactor=0.0, roughnessFactor=0.08,
                alphaMode='OPAQUE'
            )
            # rotation transforms you used
            self.Ry = trimesh.transformations.rotation_matrix(np.radians(90),  [0,1,0])
            self.Rx = trimesh.transformations.rotation_matrix(np.radians(-90), [1,0,0])

            # create an initial mesh node (will update positions each frame)
            dummy = np.zeros((self.faces.max()+1, 3), dtype=np.float32)
            tri = trimesh.Trimesh(vertices=dummy, faces=self.faces, process=False)
            tri.vertex_normals  # ensure normals array exists
            self.mesh = pyrender.Mesh.from_trimesh(tri, material=self.material, smooth=True)
            self.node = self.scene.add(self.mesh)

            self.renderer = pyrender.OffscreenRenderer(W, H)
            self.flags = pyrender.RenderFlags.RGBA
            if hasattr(pyrender.RenderFlags, "SHADOWS"):
                self.flags |= pyrender.RenderFlags.SHADOWS

        def render(self, verts):
            # apply your rotations to verts, no re-allocations
            v = np.asarray(verts, dtype=np.float32)
            v_h = np.concatenate([v, np.ones((len(v),1), np.float32)], axis=1)
            v_h = (self.Ry @ v_h.T).T
            v_h = (self.Rx @ v_h.T).T
            v = v_h[:, :3]

            # update positions in-place if available; otherwise replace mesh node
            prim = self.mesh.primitives[0]
            if hasattr(prim, "positions") and hasattr(prim, "needs_update"):
                prim.positions = v
                prim.needs_update = True
            else:
                self.scene.remove_node(self.node)
                tri = trimesh.Trimesh(vertices=v, faces=self.faces, process=False)
                tri.vertex_normals
                self.mesh = pyrender.Mesh.from_trimesh(tri, material=self.material, smooth=True)
                self.node = self.scene.add(self.mesh)

            rgba, _ = self.renderer.render(self.scene, flags=self.flags)
            return rgba[:, :, :3]

        def close(self):
            self.renderer.delete()

    # ----------------------------- data loading ------------------------------
    with h5py.File(file_path, "r") as src:
        data_group = _require_data_group(src, file_path)
        resolved_demo = _resolve_demo_name(data_group, demo_id)
        if resolved_demo is None:
            raise KeyError(f"Demo '{demo_id}' not found in '{file_path}'.")

        sequences = {}
        for name in dataset_names:
            dpath = f"data/{resolved_demo}/{name}"
            if dpath in src and isinstance(src[dpath], h5py.Dataset):
                arr = src[dpath][()].astype(np.float32, copy=False)
                if arr.ndim == 2 and arr.shape[0] == 21:
                    arr = arr[None, ...]
                arr[:, :, 0] = -arr[:, :, 0]
                nudge_thumb_lateral_3j(arr, k_tip=0.3, k_ip=0.2, k_mcp=0.1)
                sequences[name] = arr

    if not sequences:
        return {}

    results = {}
    W, H = target_size
    for name, seq in sequences.items():
        side = _pick_side(name)
        body_model, device = _load_mano_model(
            side=side, model_root=mano_model_root, use_cuda=use_cuda,
            num_pca_comps=6, use_pose_blending=True, use_shape_blending=True,
            use_pca=False, use_flat_mean=False
        )
        params = _fit_sequence_to_mano(body_model, seq)
        faces = body_model.faces

        # one-time renderer for the whole sequence
        rctx = RenderOnce(faces=faces, image_size=target_size, bg_rgb=background_color)
        out_dir = _ensure_dir(os.path.join(output_dir, name))
        obj_dir = _ensure_dir(os.path.join(out_dir, "meshes"))   # <---- NEW

        # normalize selection set
        sel = None
        if selected_frame_indices is not None:
            sel = set(int(i) for i in selected_frame_indices)

        T = seq.shape[0]
        def _select_nf(pd, nf):
            out={}
            for k,v in pd.items():
                if isinstance(v, np.ndarray) and v.shape[:1] == (T,):
                    v = v[nf]
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    v = v[None, ...]
                out[k]=v
            return out

        for nf in range(T):
            p = _select_nf(params, nf)
            if 'Th' in p: p['Th'] = np.zeros_like(p['Th'])
            if 'Rh' in p: p['Rh'] = np.zeros_like(p['Rh'])

            verts = body_model(return_verts=True, return_tensor=False, **p)[0]
            if verts.ndim == 3:
                verts = verts[0]

            # ---- OPTIONAL OBJ EXPORT FOR SELECTED FRAMES ----
            if sel is not None and nf in sel:
                obj_path = os.path.join(obj_dir, f"{demo_id}_{nf:05d}.obj")
                _export_obj(verts, faces, obj_path)

            # ---- render (reuses the single renderer/scene) ----
            img = rctx.render(verts)
            cv2.imwrite(os.path.join(out_dir, f"{demo_id}_{nf:05d}.png"), img[:, :, ::-1])

        rctx.close()
        results[name] = True

    return results

def _prepare_tactile_frame(sample, attrs):
    """
    Normalize tactile sample to a 2D array for visualization.
    """
    arr = np.asarray(sample)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 1:
        # Prefer explicit metadata if available.
        grid_shape = attrs.get("grid_shape") if attrs else None
        if grid_shape:
            rows, cols = map(int, grid_shape)
            return arr.reshape(rows, cols).astype(np.float32)

        length = arr.shape[0]
        root = int(math.sqrt(length))
        if root * root == length:
            return arr.reshape(root, root).astype(np.float32)

        # Last resort: treat as single-row heatmap.
        return arr.reshape(1, length).astype(np.float32)

    raise ValueError(f"Unsupported tactile sample shape: {arr.shape}")


def _apply_gaussian(frame, sigma):
    if sigma is None or sigma <= 0.0:
        return frame

    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(frame, sigma=sigma)
    except ImportError:
        # Fallback implementation without scipy
        radius = max(int(round(3 * sigma)), 1)
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()

        temp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=frame)
        smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=temp)
        return smoothed.astype(frame.dtype, copy=False)


def _render_heatmap(frame, colormap, max_value=3072.0, num_intervals=5, method="binning"):
    """
    Render a tactile frame to an image object (Pillow Image).

    Args:
        frame (np.ndarray): Raw tactile values shaped as a 2D grid.
        colormap: Optional Matplotlib colormap to apply; grayscale when None.
        max_value (float): Expected maximum tactile value for clipping.
        num_intervals (int): Number of discrete intervals mapped across the range (3, 6, or 9).
        method (str): Processing method ('binning', 'linear', or 'log').
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to export tactile heatmaps. Install with `pip install Pillow`."
        ) from exc
    frame = np.asarray(frame, dtype=np.float32)

    frame = np.where(np.isfinite(frame), frame, 0.0)

    max_value = float(max_value) if max_value is not None else 3072.0
    if max_value <= 0.0:
        max_value = 3072.0
    frame = np.clip(frame, 0.0, max_value)

    num_intervals = int(num_intervals) if num_intervals else 0

    # Apply processing method
    if method == "binning":
        # Original discrete binning method
        if num_intervals > 1:
            bins = np.linspace(0.0, max_value, num_intervals + 1, dtype=np.float32)
            indices = np.digitize(frame, bins[1:-1], right=False)
            levels = np.linspace(0.0, 1.0, num_intervals, dtype=np.float32)
            frame = levels[indices]
        else:
            frame = frame / max_value
            frame = np.clip(frame, 0.0, 1.0)
    
    elif method == "linear":
        # Linear normalization + quantization
        frame = frame / max_value
        frame = np.clip(frame, 0.0, 1.0)
        if num_intervals > 1:
            levels = np.linspace(0.0, 1.0, num_intervals, dtype=np.float32)
            indices = np.round(frame * (num_intervals - 1)).astype(np.int32)
            indices = np.clip(indices, 0, num_intervals - 1)
            frame = levels[indices]
    
    elif method == "log":
        # Logarithmic scale (sensor-aware)
        # Use logarithmic scale for thresholds to match sensor response curve
        # This gives more sensitivity when pressure first increases
        
        # Use reasonable vmin to avoid log(0) issues and match sensor characteristics
        vmin = 500.0  # Based on value_to_color function
        vmax = max_value
        
        # Clamp values to valid range
        clamped = np.clip(frame, vmin, vmax)
        
        # Apply logarithmic transformation
        log_val = np.log10(clamped)
        log_min = np.log10(vmin)
        log_max = np.log10(vmax)
        
        # Normalize to [0, 1]
        norm = (log_val - log_min) / (log_max - log_min)
        norm = np.clip(norm, 0.0, 1.0)
        
        # Discretize into intervals
        if num_intervals > 1:
            # Create discrete levels
            levels = np.linspace(0.0, 1.0, num_intervals, dtype=np.float32)
            # Map to discrete level indices
            indices = np.round(norm * (num_intervals - 1)).astype(np.int32)
            indices = np.clip(indices, 0, num_intervals - 1)
            frame = levels[indices]
        else:
            frame = norm


    else:
        raise ValueError(f"Unknown method: {method}. Use 'binning', 'linear', or 'log'")

    if isinstance(colormap, str):
        try:
            from matplotlib import cm
            cmap_callable = cm.get_cmap(colormap)
        except ImportError:
            cmap_callable = None
    else:
        cmap_callable = colormap

    if cmap_callable is None:
        pixels = (frame * 255.0).astype(np.uint8)
        mode = "L"
    else:
        rgba = (cmap_callable(frame) * 255.0).astype(np.uint8)
        pixels = rgba[..., :3]
        mode = "RGB"

    return Image.fromarray(pixels, mode=mode)


try:
    from PIL import Image as _ImageBase
    from PIL import ImageDraw
    Image = _ImageBase
    _HEATMAP_RESAMPLE = _ImageBase.NEAREST
except ImportError:
    Image = None
    _HEATMAP_RESAMPLE = None
    ImageDraw = None


def export_wrist_pose_visualizations(
    file_path,
    demo_id,
    output_dir,
    dataset_names=("left_wrist_pos", "right_wrist_pos"),
    target_size=(480, 480),
    background_color=(255, 255, 255),
    point_radius=8,
    trail_length=30,
    trail_alpha=0.3,
    use_3d=True,
):
    """
    Render wrist pose sequences to individual image directories per dataset.
    
    Args:
        file_path (str): Path to the source HDF5 file.
        demo_id (str): Demo identifier under the `data` group.
        output_dir (str): Root directory where wrist pose visualizations will be written.
        dataset_names (Iterable[str]): Wrist pose datasets to visualise.
        target_size (tuple[int, int]): Output image resolution.
        background_color (tuple[int, int, int]): RGB background colour.
        point_radius (int): Radius (in pixels) for wrist position markers.
        trail_length (int): Number of previous positions to show as trail.
        trail_alpha (float): Alpha value for trail visualization (0.0-1.0).
        use_3d (bool): Whether to render in 3D perspective view.
        
    Returns:
        dict[str, bool]: Mapping of dataset name to True when frames were exported.
    """
    if Image is None or ImageDraw is None:
        raise ImportError("PIL.Image and PIL.ImageDraw are required for wrist pose visualization")
    
    width, height = target_size
    
    def project_3d_to_2d(point_3d, camera_distance=1.5, rotation_angle=0.3):
        """3D to 2D projection with perspective, returning depth info."""
        x, y, z = point_3d
        
        # Apply rotation around Y axis for better 3D view
        cos_r, sin_r = math.cos(rotation_angle), math.sin(rotation_angle)
        x_rot = x * cos_r - z * sin_r
        z_rot = x * sin_r + z * cos_r
        
        # Simple perspective projection
        if z_rot + camera_distance > 0:
            scale = camera_distance / (z_rot + camera_distance)
            screen_x = x_rot * scale
            screen_y = y * scale
        else:
            # Handle points behind camera
            screen_x = x_rot * 5
            screen_y = y * 5
        
        return screen_x, screen_y, z_rot
    
    with h5py.File(file_path, "r") as src:
        data_group = _require_data_group(src, file_path)
        resolved_demo = _resolve_demo_name(data_group, demo_id)
        if resolved_demo is None:
            available = ", ".join(list(data_group.keys())[:5])
            raise KeyError(
                f"Demo '{demo_id}' not found in '{file_path}'. Available demos include: {available}"
            )
        
        sequences: dict[str, np.ndarray] = {}
        for name in dataset_names:
            dataset_path = f"data/{resolved_demo}/{name}"
            if dataset_path not in src:
                continue
            dset = src[dataset_path]
            if not isinstance(dset, h5py.Dataset):
                continue
            sequences[name] = dset[()].astype(np.float32, copy=False)
    
    if not sequences:
        return {}
    
    # Calculate bounds for normalization
    all_points = np.concatenate([arr for arr in sequences.values()])
    bounds_min = np.nanmin(all_points, axis=0)
    bounds_max = np.nanmax(all_points, axis=0)
    bounds_range = bounds_max - bounds_min
    denom = np.where(bounds_range > 0, bounds_range, 1.0)
    
    # Configuration
    wrist_colors = {
        "left_wrist_pos": (66, 165, 245),   # Blue
        "right_wrist_pos": (244, 81, 96),   # Red
    }
    
    trail_buffers = {name: deque(maxlen=trail_length) for name in sequences.keys()}
    results: dict[str, bool] = {}
    num_frames = max(arr.shape[0] for arr in sequences.values())
    
    # 3D visualization parameters
    center_3d = (bounds_min + bounds_max) / 2
    scale_3d = max(bounds_range) * 0.6
    
    for name, arr in sequences.items():
        target_dir = os.path.join(output_dir, name)
        os.makedirs(target_dir, exist_ok=True)
        
        base_color = wrist_colors.get(name, (200, 200, 200))
        trail = trail_buffers[name]
        
        exported = False
        for frame_idx in range(num_frames):
            canvas = Image.new("RGBA", target_size, background_color + (255,))
            draw = ImageDraw.Draw(canvas)
            
            # Skip drawing coordinate axes - just show wrist movement
            center_x, center_y = width // 2, height // 2
            
            # Get current wrist position
            if frame_idx < arr.shape[0]:
                wrist_pos = arr[frame_idx]
                if np.isfinite(wrist_pos).all():
                    if use_3d:
                        # 3D visualization with depth visualization
                        # Normalize to [-1, 1] range
                        normalized_pos = (wrist_pos - center_3d) / scale_3d
                        # Add dynamic rotation to better show Z changes
                        dynamic_rotation = 0.3 + (frame_idx * 0.02) % (2 * math.pi)
                        # Project to 2D with perspective
                        screen_x, screen_y, depth = project_3d_to_2d(normalized_pos, camera_distance=1.5, rotation_angle=dynamic_rotation)
                        # Convert to pixel coordinates
                        px = center_x + screen_x * (width * 0.25)
                        py = center_y - screen_y * (height * 0.25)  # Flip Y axis
                        current_pos = (px, py, depth)  # Include depth for visualization
                    else:
                        # 2D visualization (original)
                        norm = (wrist_pos - bounds_min) / denom
                        norm = np.clip(norm, 0.0, 1.0)
                        px = float(norm[0]) * (width - 1)
                        py = (1.0 - float(norm[1])) * (height - 1)
                        current_pos = (px, py)
                else:
                    current_pos = None
            else:
                current_pos = None
            
            # Add current position to trail
            trail.append(current_pos)
            
            # Draw trail with depth-based visualization
            if len(trail) >= 2:
                trail_points = [p for p in trail if p is not None]
                if len(trail_points) >= 2:
                    for i in range(len(trail_points) - 1):
                        alpha = int(trail_alpha * 255 * (i + 1) / len(trail_points))
                        
                        if use_3d and len(trail_points[i]) == 3 and len(trail_points[i + 1]) == 3:
                            # 3D trail with depth visualization
                            depth1, depth2 = trail_points[i][2], trail_points[i + 1][2]
                            avg_depth = (depth1 + depth2) / 2
                            
                            # Depth-based color and line width
                            depth_factor = max(0.3, min(1.0, (avg_depth + 1) / 2))
                            depth_color = tuple(int(c * depth_factor) for c in base_color)
                            trail_color = depth_color + (alpha,)
                            line_width = max(1, int(3 * depth_factor))
                        else:
                            # 2D trail
                            trail_color = base_color + (alpha,)
                            line_width = 3
                        
                        draw.line([(trail_points[i][0], trail_points[i][1]), 
                                 (trail_points[i + 1][0], trail_points[i + 1][1])], 
                                fill=trail_color, width=line_width)
            
            # Draw current position with depth visualization
            if current_pos is not None:
                if use_3d and len(current_pos) == 3:
                    # 3D position with depth visualization
                    px, py, depth = current_pos
                    
                    # Depth-based size and color
                    depth_factor = max(0.5, min(1.5, (depth + 1) / 2))
                    current_radius = int(point_radius * depth_factor)
                    
                    # Depth-based color (closer = brighter)
                    depth_color_factor = max(0.4, min(1.0, (depth + 1) / 2))
                    depth_color = tuple(int(c * depth_color_factor) for c in base_color)
                    
                    # Draw outer circle
                    outer_color = depth_color + (255,)
                    draw.ellipse((px - current_radius, py - current_radius, 
                                px + current_radius, py + current_radius), fill=outer_color)
                    
                    # Draw inner circle
                    inner_color = tuple(int(min(255, c + 80)) for c in depth_color) + (255,)
                    inner_radius = max(1, current_radius // 2)
                    draw.ellipse((px - inner_radius, py - inner_radius, 
                                px + inner_radius, py + inner_radius), fill=inner_color)
                else:
                    # 2D position
                    px, py = current_pos
                    outer_color = base_color + (255,)
                    draw.ellipse((px - point_radius, py - point_radius, 
                                px + point_radius, py + point_radius), fill=outer_color)
                    
                    inner_color = tuple(int(min(255, c + 50)) for c in base_color) + (255,)
                    inner_radius = max(1, point_radius // 2)
                    draw.ellipse((px - inner_radius, py - inner_radius, 
                                px + inner_radius, py + inner_radius), fill=inner_color)
            
            frame_filename = f"{demo_id}_{name}_{frame_idx:05d}.png"
            frame_path = os.path.join(target_dir, frame_filename)
            canvas.convert("RGB").save(frame_path, format="PNG")
            exported = True
        
        results[name] = exported
    
    return results




if __name__ == "__main__":
    # Example usage
    file_path = 'kit_data.hdf5'
    
    print("Top-level keys:", list_hdf5_keys(file_path))
    data_keys = list_hdf5_keys(file_path, recursive=True, root="data")
    print(f"Total entries under 'data': {len(data_keys)}")

    demos = load_all_demos(file_path, skip_keys={"rgb_images_jpeg"})
    print(f"Loaded {len(demos)} demos (skipping 'rgb_images_jpeg').")
    print("First demos:", list(demos.keys())[:10])

    # lets open the demo_00 and print its keys
    demo_00 = demos.get("demo_00", {})
    print("Keys in demo_00:", list(demo_00.keys()))
    # Keys in demo_00: ['camera_poses', 'left_hand_landmarks', 'left_palm_pos', 'left_pressure', 'left_wrist_pos', 'right_hand_landmarks', 'right_palm_pos', 'right_pressure', 'right_wrist_pos', 'timestamps']

    # Extract demo_00 RGB frames to disk for inspection
    rgb_output_dir = 'demo_00_rgb_frames'
    export_rgb_frames(file_path, 'demo_00', rgb_output_dir, channel_order="bgr")
    print(f"Wrote 480x480 RGB frames (channel order handled) for 'demo_00' to {rgb_output_dir}")

    # Export tactile heatmaps for demo_00
    tactile_output_dir = 'demo_00_tactile_heatmaps'
    # export_tactile_heatmaps(
    #     file_path,
    #     'demo_00',
    #     tactile_output_dir,
    #     max_value=3072.0,
    #     num_intervals=5,
    # )
    export_tactile_voronoi(
        file_path,
        'demo_00',
        tactile_output_dir,
        max_value=3072.0,
        num_intervals=5,
    )
    print(f"Wrote 480x480 tactile heatmaps for 'demo_00' to {tactile_output_dir}")
