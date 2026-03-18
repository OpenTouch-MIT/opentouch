#!/usr/bin/env python3
"""Generate video from images using ffmpeg."""

import argparse
import subprocess
from pathlib import Path


def check_ffmpeg():
    """Check if ffmpeg is available.
    
    Returns:
        bool: True if ffmpeg is available, False otherwise
    """
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    
    

def generate_video_from_images(
    demo_id,
    image_dir,
    output_video,
    fps=30,
    max_frames=None,
    start_id=None,
    end_id=None,
):
    """Generate video from images using ffmpeg.
    
    Args:
        demo_id (str): Demo id used in filename pattern (e.g. demo_040_00012.png)
        image_dir (str): Directory containing images
        output_video (str): Output video file path
        fps (int): Frame rate for the video
        max_frames (int, optional): Maximum number of frames to include
        start_id (int, optional): First frame index (e.g. 40 for *_00040.png)
        end_id (int, optional): Last frame index (inclusive)
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        RuntimeError: If ffmpeg is not available or no images found
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg not found")

    image_path = Path(image_dir)
    image_files = sorted(list(image_path.glob('*.png')) + list(image_path.glob('*.jpg')))
    if not image_files:
        raise RuntimeError(f"No images found in {image_dir}")

    # Figure out existing frame indices from filenames: demo_XXX_00040.png -> 40
    def _idx(p: Path) -> int:
        return int(p.stem.split('_')[-1])

    indices = [ _idx(f) for f in image_files ]
    min_idx, max_idx = min(indices), max(indices)

    # Default range = whole sequence if not specified
    if start_id is None:
        start_id = min_idx
    else:
        start_id = max(start_id, min_idx)

    if end_id is None:
        end_id = max_idx
    else:
        end_id = min(end_id, max_idx)

    if start_id > end_id:
        raise RuntimeError(
            f"Invalid start/end range after clamping: start_id={start_id}, end_id={end_id}, "
            f"available [{min_idx}, {max_idx}]"
        )
    # Number of frames to feed ffmpeg
    total_range = end_id - start_id + 1
    frames_to_write = total_range
    if max_frames is not None:
        frames_to_write = min(frames_to_write, max_frames)

    ext = image_files[0].suffix  # assume all same extension

    # Use start_number so ffmpeg starts from the correct file
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-start_number', str(start_id),
        '-i', str(image_path / f'{demo_id}_%05d{ext}'),
        '-frames:v', str(frames_to_write),
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video,
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed: {exc}") from exc


def main():
    """Main entry point for video generation."""
    parser = argparse.ArgumentParser(description="Generate video from images")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("--output", "-o", default="output_video.mp4")
    parser.add_argument("--fps", "-f", type=int, default=30)
    parser.add_argument("--max-frames", "-m", type=int, default=None)
    args = parser.parse_args()
    
    try:
        generate_video_from_images(args.image_dir, args.output, fps=args.fps, max_frames=args.max_frames)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())
