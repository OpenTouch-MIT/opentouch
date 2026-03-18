#!/usr/bin/env python3
import os
import sys
import subprocess

import h5py

def main(h5_path):
    h5_path = os.path.join('..', 'auto_sample', h5_path)
    if not os.path.isfile(h5_path):
        print(f"File not found: {h5_path}")
        sys.exit(1)

    # e.g. "grocery_target_p1.hdf5" -> "grocery_target_p1"
    hdf5_name = os.path.splitext(os.path.basename(h5_path))[0]

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            print("No 'data' group found in the hdf5 file.")
            sys.exit(1)

        data_group = f["data"]
        demo_names = [k for k in data_group.keys() if k.startswith("demo_")]

    demo_names = sorted(demo_names)
    if not demo_names:
        print("No 'demo_xx' entries found under /data.")
        return

    print(f"Found {len(demo_names)} demos under /data.")
    # Take every 10th demo: 0, 10, 20, ...
    selected_demos = demo_names[0::10]

    print(f"Running on {len(selected_demos)} demos (every 10th):")
    for d in selected_demos:
        print("  ", d)

    for demo in selected_demos:
        cmd = ["bash", "scripts/extract_demo.sh", hdf5_name, demo]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/file.hdf5")
        sys.exit(1)
    main(sys.argv[1])
