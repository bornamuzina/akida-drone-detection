"""
Convert an RGB video to an event stream (.dat), writing EVERY event.

Prophesee's viz_video_to_event_simulator.py is a *display* tool: it waits
until 50,000 events have accumulated, draws them, and keeps only that many.
Its -o flag saves whatever the display happened to show, so sparse
conversions lose most of the clip.

This script drives the same EventSimulator but writes events straight to
the DAT file as they are produced, with no windowing. Slicing is the
reader's job.

Usage:
    python convert_video_to_events.py INPUT.mp4 -o OUTPUT.dat [--Cp 0.2 ...]
"""

import argparse
import sys

import numpy as np

from metavision_core_ml.video_to_event.simulator import EventSimulator
from metavision_core_ml.data.video_stream import TimedVideoStream
from metavision_core.event_io import DatWriter


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a video to events, keeping every event."
    )

    p.add_argument("path", help="input video")
    p.add_argument("-o", "--output", required=True, help="output .dat file")

    p.add_argument("--height_width", nargs=2, type=int, default=None,
                   help="rescale input to this height and width")
    p.add_argument("-fps", "--override_fps", type=float, default=0,
                   help="override the video frame rate")
    p.add_argument("--quiet", action="store_true",
                   help="suppress progress output")

    sim = p.add_argument_group("Simulator parameters")
    sim.add_argument("--Cp", type=float, default=0.15,
                     help="positive contrast threshold")
    sim.add_argument("--Cn", type=float, default=0.10,
                     help="negative contrast threshold")
    sim.add_argument("--refractory_period", type=float, default=1,
                     help="dead time per pixel after an event, in us")
    sim.add_argument("--sigma_threshold", type=float, default=0.001,
                     help="threshold spread across pixels")
    sim.add_argument("--cutoff_hz", type=float, default=0,
                     help="photodiode latency simulation")
    sim.add_argument("--leak_rate_hz", type=float, default=0,
                     help="reference value leakage")
    sim.add_argument("--shot_noise_rate_hz", type=float, default=10,
                     help="shot noise frequency")

    return p.parse_args()


def main():
    args = parse_args()

    # ---- input ---------------------------------------------------
    if args.height_width is None:
        height, width = -1, -1
    else:
        height, width = args.height_width

    stream = TimedVideoStream(args.path, height, width,
                              override_fps=args.override_fps)

    if args.height_width is None:
        height, width = stream.get_size()

    if not args.quiet:
        print(f"Input      : {args.path}")
        print(f"Resolution : {width} x {height}")

    # ---- simulator -----------------------------------------------
    simulator = EventSimulator(
        height, width,
        args.Cp, args.Cn, args.refractory_period,
        sigma_threshold=args.sigma_threshold,
        cutoff_hz=args.cutoff_hz,
        leak_rate_hz=args.leak_rate_hz,
        shot_noise_rate_hz=args.shot_noise_rate_hz,
    )

    if not args.quiet:
        print(f"Cp/Cn      : {args.Cp} / {args.Cn}")
        print(f"Shot noise : {args.shot_noise_rate_hz} Hz")
        print(f"Refractory : {args.refractory_period} us")
        print()

    writer = DatWriter(args.output, height=height, width=width)

    # ---- convert -------------------------------------------------
    #
    # The key difference from the sample: no threshold, no buffer.
    # Every frame's events are drained and written immediately.
    #
    total_written = 0
    frames = 0
    last_ts = 0

    for img, ts in stream:
        simulator.image_callback(img, ts)

        events = simulator.get_events()
        simulator.flush_events()

        if len(events):
            writer.write(events)
            total_written += len(events)
            last_ts = max(last_ts, int(events["t"][-1]))

        frames += 1

        if not args.quiet and frames % 30 == 0:
            print(f"\r  frame {frames:5d} | events written {total_written:9d}",
                  end="")
            sys.stdout.flush()

    # Drain anything the simulator still holds after the last frame.
    events = simulator.get_events()
    simulator.flush_events()

    if len(events):
        writer.write(events)
        total_written += len(events)
        last_ts = max(last_ts, int(events["t"][-1]))

    writer.close()

    # ---- report --------------------------------------------------
    if not args.quiet:
        print()
        print()
        print(f"Frames     : {frames}")
        print(f"Events     : {total_written}")

        if total_written:
            print(f"Last event : {last_ts / 1000:.1f} ms")
            print(f"Rate       : {total_written / max(last_ts / 1e6, 1e-9):.0f} ev/s")
        else:
            print("WARNING: no events were produced. Check the contrast")
            print("thresholds -- they may be too high for this footage.")

        print(f"Output     : {args.output}")


if __name__ == "__main__":
    main()