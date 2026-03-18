#!/usr/bin/env python3
""" Utility helpers for concatenating multiple videos into a single layout using FFmpeg. """

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence


class VideoConcatError(RuntimeError):
    """Raised when concatenation fails."""


def check_executable(name: str) -> None:
    """Ensure an external command is available."""
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required executable '{name}' not found in PATH")


def probe_height(video_path: Path) -> int | None:
    """Return the height of the first video stream via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=height",
                "-of", "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise VideoConcatError(f"Failed to probe video metadata for '{video_path}': {exc.stderr.strip()}") from exc

    try:
        data = json.loads(result.stdout)
        if streams := data.get("streams"):
            if height := streams[0].get("height"):
                return int(height)
    except (ValueError, KeyError, TypeError, IndexError):
        pass
    return None


def concat_videos(
    *,
    videos: Sequence[Path] | None = None,
    left: Path | None = None,
    mid: Path | None = None,
    right: Path | None = None,
    output: Path,
    copy_audio: bool = False,
    crf: int | float = 23,
    preset: str = "medium",
    scale_height: int | None = None,
    layout: str = "horizontal",
) -> None:
    """Concatenate two or more videos into a single output using FFmpeg."""
    check_executable("ffmpeg")
    check_executable("ffprobe")

    if videos is None:
        if left is None or mid is None or right is None:
            raise ValueError("Either `videos` or `left`/`mid`/`right` must be provided")
        inputs: List[Path] = [Path(left), Path(mid), Path(right)]
    else:
        if len(videos) < 2:
            raise ValueError("At least two videos are required for concatenation")
        inputs = [Path(path) for path in videos]

    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")

    layout = layout.lower()
    if layout not in {"horizontal", "vertical"}:
        raise ValueError("layout must be 'horizontal' or 'vertical'")

    heights = [probe_height(path) for path in inputs]
    if scale_height is None and len({h for h in heights if h}) > 1:
        valid_heights = [h for h in heights if h]
        scale_height = min(valid_heights) if valid_heights else None

    filter_parts: List[str] = []
    mapped_labels: List[str] = []

    for idx, _ in enumerate(inputs):
        src_label = f"[{idx}:v]"
        dst_label = f"v{idx}"
        if scale_height:
            filter_parts.append(f"{src_label}scale=-2:{int(scale_height)},setsar=1[{dst_label}]")
        else:
            filter_parts.append(f"{src_label}setsar=1[{dst_label}]")
        mapped_labels.append(f"[{dst_label}]")

    stack_filter = "hstack" if layout == "horizontal" else "vstack"
    filter_parts.append(f"{''.join(mapped_labels)}{stack_filter}=inputs={len(inputs)}[vout]")
    filter_complex = ";".join(filter_parts)

    cmd: List[str] = ["ffmpeg"]
    for path in inputs:
        cmd.extend(["-i", str(path)])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-shortest"
    ])

    if copy_audio:
        audio_index = 1 if len(inputs) > 1 else 0
        cmd.extend(["-map", f"{audio_index}:a?", "-c:a", "aac", "-b:a", "192k"])

    cmd.extend(["-y", str(output)])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoConcatError(f"FFmpeg failed: {exc}") from exc


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate videos into a combined visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("videos", nargs="*", help="Input videos in order (default: left.mp4 mid.mp4 right.mp4)")
    parser.add_argument("-o", "--output", default="concatenated_output.mp4", help="Output video path")
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal", help="Stack direction")
    parser.add_argument("--scale-height", type=int, help="Force common output height before stacking")
    parser.add_argument("--crf", type=float, default=23.0, help="libx264 CRF value (lower = higher quality)")
    parser.add_argument("--preset", default="medium", help="libx264 preset (ultrafast..veryslow)")
    parser.add_argument("--copy-audio", action="store_true", help="Copy audio from the middle input if present")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.videos:
        if len(args.videos) < 2:
            print("Error: At least two videos are required")
            return 1
        videos = [Path(v) for v in args.videos]
    else:
        videos = [Path("left.mp4"), Path("mid.mp4"), Path("right.mp4")]

    try:
        concat_videos(
            videos=videos,
            output=Path(args.output),
            copy_audio=args.copy_audio,
            crf=args.crf,
            preset=args.preset,
            scale_height=args.scale_height,
            layout=args.layout,
        )
        print(f"✓ Video saved to: {args.output}")
        return 0
    except Exception as exc:
        print(f"✗ Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
