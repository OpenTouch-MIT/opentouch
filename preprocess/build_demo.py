#!/usr/bin/env python3
"""
Pipeline to export frames and create combined videos from HDF5 data.

This script extracts tactile, RGB, and pose data from HDF5 files and generates
synchronized video visualizations.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from concat_videos import concat_videos
from generate_video import generate_video_from_images
from load_data import (
    export_pose_mano,
    export_rgb_frames,
    export_tactile_mano,
    export_wrist_pose_visualizations,
)


@dataclass
class PipelineConfig:
    """Configuration for the demo pipeline."""

    # Input/Output
    hdf5path: Path
    demoid: str
    outputdir: Path

    # Video settings
    fps: int = 30
    framesize: tuple[int, int] = (1280, 960)
    maxframes: Optional[int] = None
    startid: Optional[int] = None
    endid: Optional[int] = None

    # Processing settings
    channelorder: Optional[str] = "bgr"
    scaleheight: Optional[int] = None

    # Tactile settings
    tactilemax: Optional[float] = None
    tactileintervals: Optional[int] = None
    tactilemethod: str = "binning"

    # Hand selection
    hand: str = "right"
    showwrist: bool = False
    wrist3d: bool = False

    @property
    def useleft(self) -> bool:
        return self.hand in ("left", "both")

    @property
    def useright(self) -> bool:
        return self.hand in ("right", "both")


class DemoPipeline:
    """Pipeline for processing demo data from HDF5 to videos."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.framedirs: dict[str, Path] = {}
        self.videofiles: dict[str, Path] = {}

    def run(self) -> Path:
        """Execute the full pipeline.

        Returns:
            Path to the combined output video
        """
        self.setup()
        self.export()
        self.generate()
        return self.combine()

    def setup(self) -> None:
        """Create output directory structure."""
        self.config.outputdir.mkdir(parents=True, exist_ok=True)

        # Define frame directories
        self.framedirs = {
            "rgb": self.config.outputdir / "rgb",
            "tactile": self.config.outputdir / "tactile",
            "pose": self.config.outputdir / "pose",
            "wrist": self.config.outputdir / "wrist",
        }

    def export(self) -> None:
        """Export all frame types from HDF5."""
        print("Exporting frames...")
        cfg = self.config

        # RGB frames
        export_rgb_frames(
            file_path=str(cfg.hdf5path),
            demo_id=cfg.demoid,
            output_dir=str(self.framedirs["rgb"]),
            target_size=cfg.framesize,
            channel_order=cfg.channelorder,
        )

        # Tactile data
        export_tactile_mano(
            file_path=str(cfg.hdf5path),
            demo_id=cfg.demoid,
            output_dir=str(self.framedirs["tactile"]),
            target_size=cfg.framesize,
            max_value=cfg.tactilemax,
            num_intervals=cfg.tactileintervals,
            method=cfg.tactilemethod,
        )

        # Hand pose
        self.posedata = export_pose_mano(
            file_path=str(cfg.hdf5path),
            demo_id=cfg.demoid,
            output_dir=str(self.framedirs["pose"]),
            target_size=cfg.framesize,
        )

        # Wrist pose (optional)
        self.wristdata = {}
        if cfg.showwrist:
            self.wristdata = export_wrist_pose_visualizations(
                file_path=str(cfg.hdf5path),
                demo_id=cfg.demoid,
                output_dir=str(self.framedirs["wrist"]),
                target_size=cfg.framesize,
                use_3d=cfg.wrist3d,
            )

    def generate(self) -> None:
        """Generate individual videos from frame directories."""
        print("\nGenerating videos...")
        print(f"Hand selection: {self.config.hand}")

        cfg = self.config
        base = cfg.outputdir

        # RGB video (always generated)
        self.make_video("mid", self.framedirs["rgb"], base / "mid.mp4")

        # Left hand videos
        if cfg.useleft:
            self.make_hand("left")

        # Right hand videos
        if cfg.useright:
            self.make_hand("right")

    def make_hand(self, hand: str) -> None:
        """Generate videos for a specific hand."""
        base = self.config.outputdir
        tactiledir = self.framedirs["tactile"] / f"{hand}_pressure"
        posedir = self.framedirs["pose"] / f"{hand}_hand_landmarks"
        wristdir = self.framedirs["wrist"] / f"{hand}_wrist_pos"

        # Tactile video
        self.make_video(f"{hand}_touch", tactiledir, base / f"{hand}_touch.mp4")

        # Pose video (if available)
        if self.posedata.get(f"{hand}_hand_landmarks"):
            self.make_video(f"{hand}_pose", posedir, base / f"{hand}_pose.mp4")

        # Wrist video (if available)
        if self.wristdata.get(f"{hand}_wrist_pos"):
            self.make_video(f"{hand}_wrist", wristdir, base / f"{hand}_wrist.mp4")

    def make_video(self, name: str, framedir: Path, output: Path) -> None:
        """Create a single video from frames."""
        if not framedir.exists():
            raise FileNotFoundError(f"Frame directory missing: {framedir}")

        if not any(framedir.glob("*.png")) and not any(framedir.glob("*.jpg")):
            raise RuntimeError(f"No image frames found in {framedir}")

        success = generate_video_from_images(
            demo_id=self.config.demoid,
            image_dir=str(framedir),
            output_video=str(output),
            fps=self.config.fps,
            max_frames=self.config.maxframes,
            start_id=self.config.startid,
            end_id=self.config.endid,
        )

        if not success:
            raise RuntimeError(f"Failed to create video: {name}")

        self.videofiles[name] = output

    def combine(self) -> Path:
        """Combine all videos into final output."""
        print("\nCombining videos...")

        cfg = self.config
        base = cfg.outputdir
        ordered = []

        # Left column
        if cfg.useleft:
            leftvideos = self.build_column("left")
            if len(leftvideos) > 1:
                leftcol = base / "left_column.mp4"
                concat_videos(
                    videos=leftvideos,
                    output=leftcol,
                    layout="vertical",
                    copy_audio=False,
                    crf=18,
                    preset="medium",
                )
                ordered.append(leftcol)
            else:
                ordered.extend(leftvideos)

        # Middle (RGB)
        ordered.append(self.videofiles["mid"])

        # Right column
        if cfg.useright:
            rightvideos = self.build_column("right")
            if len(rightvideos) > 1:
                rightcol = base / "right_column.mp4"
                concat_videos(
                    videos=rightvideos,
                    output=rightcol,
                    layout="vertical",
                    copy_audio=False,
                    crf=18,
                    preset="medium",
                )
                ordered.append(rightcol)
            else:
                ordered.extend(rightvideos)

        # Final horizontal concatenation
        output = base / "combined.mp4"
        height = self.calc_height()

        concat_videos(
            videos=ordered,
            output=output,
            layout="horizontal",
            copy_audio=False,
            crf=18,
            preset="medium",
            scale_height=height,
        )

        print(f"\nCombined video saved: {output}")
        return output

    def build_column(self, hand: str) -> list[Path]:
        """Build list of videos for a hand column (top to bottom)."""
        videos = []

        # Order: wrist -> pose -> tactile (left)
        #        tactile -> pose -> wrist (right)
        if hand == "left":
            if f"{hand}_wrist" in self.videofiles:
                videos.append(self.videofiles[f"{hand}_wrist"])
            if f"{hand}_pose" in self.videofiles:
                videos.append(self.videofiles[f"{hand}_pose"])
            videos.append(self.videofiles[f"{hand}_touch"])
        else:  # right
            videos.append(self.videofiles[f"{hand}_touch"])
            if f"{hand}_pose" in self.videofiles:
                videos.append(self.videofiles[f"{hand}_pose"])
            if f"{hand}_wrist" in self.videofiles:
                videos.append(self.videofiles[f"{hand}_wrist"])

        return videos

    def calc_height(self) -> Optional[int]:
        """Calculate appropriate scale height for final video."""
        if self.config.scaleheight:
            return self.config.scaleheight

        # Check if any vertical stacking occurred
        hasstack = (
            len(self.build_column("left")) > 1 if self.config.useleft else False
        ) or (
            len(self.build_column("right")) > 1 if self.config.useright else False
        )

        if hasstack:
            return self.config.framesize[1] * 2

        return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export frames and build combined videos from HDF5 demo data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=Path("kit_data.hdf5"),
        help="Input HDF5 file path",
    )
    parser.add_argument(
        "--demo-id",
        default="demo_00",
        help="Demo ID within the 'data' group",
    )

    # Video settings
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--max-frames", type=int, help="Limit frames per video")
    parser.add_argument("--start-id", type=int, help="First frame index")
    parser.add_argument("--end-id", type=int, help="Last frame index")
    parser.add_argument("--scale-height", type=int, help="Resize height for output")

    # Processing settings
    parser.add_argument(
        "--channel-order",
        choices=["rgb", "bgr"],
        default="bgr",
        help="RGB channel order override",
    )
    parser.add_argument(
        "--tactile-max",
        type=float,
        help="Max tactile value for normalization",
    )
    parser.add_argument(
        "--tactile-intervals",
        type=int,
        help="Number of tactile heatmap intervals",
    )
    parser.add_argument(
        "--tactile-method",
        choices=["binning", "linear", "log"],
        default="binning",
        help="Tactile processing method",
    )

    # Hand selection
    parser.add_argument(
        "--hand",
        choices=["left", "right", "both"],
        default="right",
        help="Which hand(s) to process",
    )
    parser.add_argument(
        "--show-wrist-pose",
        action="store_true",
        help="Include wrist pose visualization",
    )
    parser.add_argument(
        "--wrist-pose-3d",
        action="store_true",
        help="Use 3D wrist pose (requires --show-wrist-pose)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Build configuration
    config = PipelineConfig(
        hdf5path=args.hdf5,
        demoid=args.demo_id,
        outputdir=Path.cwd().parent / "data" / args.hdf5.stem / args.demo_id,
        fps=args.fps,
        maxframes=args.max_frames,
        startid=args.start_id,
        endid=args.end_id,
        channelorder=args.channel_order,
        scaleheight=args.scale_height,
        tactilemax=args.tactile_max,
        tactileintervals=args.tactile_intervals,
        tactilemethod=args.tactile_method,
        hand=args.hand,
        showwrist=args.show_wrist_pose,
        wrist3d=args.wrist_pose_3d,
    )

    try:
        pipeline = DemoPipeline(config)
        pipeline.run()
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
