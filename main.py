#!/usr/bin/env python3
import argparse
import subprocess
import signal
import sys
import os
import re
import time
from datetime import datetime
from pathlib import Path


def run_mode(script_path: str, output_path: str, interval: float):
    script_path = os.path.abspath(script_path)
    output_path = os.path.abspath(output_path)

    if not os.path.isfile(script_path):
        print(f"Error: script not found: {script_path}")
        sys.exit(1)

    with open(output_path, "w") as f:
        f.write(f"# GPU Monitor Log - Started {datetime.now().isoformat()}\n")
        f.write(f"# Script: {script_path}\n")
        f.write(f"# Interval: {interval}s\n\n")

    print(f"[gpu_monitor] Logging to : {output_path}")
    print(f"[gpu_monitor] Running    : {script_path}")
    print(f"[gpu_monitor] Interval   : {interval}s")
    print()

    sampler_script = f"""
import subprocess, time, sys, os, signal

output = "{output_path}"
interval = {interval}
stop = False

def handle(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGTERM, handle)
signal.signal(signal.SIGINT, handle)

while not stop:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10
        )
        with open(output, "a") as f:
            f.write(f"@@SAMPLE_START {{{{time.time()}}}}\\n")
            f.write(result.stdout)
            f.write(f"@@SAMPLE_END\\n\\n")
    except Exception as e:
        with open(output, "a") as f:
            f.write(f"@@ERROR {{{{e}}}}\\n")
    time.sleep(interval)
"""

    proc_sampler = subprocess.Popen(
        [sys.executable, "-c", sampler_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        proc_script = subprocess.Popen(
            ["bash", script_path],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        proc_script.wait()
        exit_code = proc_script.returncode
        print(f"\n[gpu_monitor] Script finished with exit code {exit_code}")
    except KeyboardInterrupt:
        print("\n[gpu_monitor] Interrupted by user")
        proc_script.terminate()
        proc_script.wait()
        exit_code = 1
    finally:
        proc_sampler.terminate()
        proc_sampler.wait()
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            with open(output_path, "a") as f:
                f.write(f"@@SAMPLE_START {time.time()}\n")
                f.write(result.stdout)
                f.write(f"@@SAMPLE_END\n\n")
        except Exception:
            pass

    print(f"[gpu_monitor] Log saved to {output_path}")
    return exit_code


def parse_log(log_path: str) -> list[dict]:
    with open(log_path, "r") as f:
        content = f.read()

    samples = []
    pattern = r"@@SAMPLE_START\s+([\d.]+)\n(.*?)@@SAMPLE_END"
    matches = re.findall(pattern, content, re.DOTALL)

    for timestamp_str, smi_output in matches:
        timestamp = float(timestamp_str)
        sample = parse_nvidia_smi(smi_output, timestamp)
        if sample:
            samples.extend(sample)

    return samples


def parse_nvidia_smi(output: str, timestamp: float) -> list[dict]:
    gpu_entries = []

    lines = output.strip().split("\n")
    i = 0
    while i < len(lines) - 1:
        line = lines[i]
        gpu_id_match = re.match(r"\|\s+(\d+)\s+(.+?)\s+(On|Off)\s*\|", line)
        if gpu_id_match:
            gpu_id = int(gpu_id_match.group(1))
            gpu_name = gpu_id_match.group(2).strip()
            gpu_name = re.sub(r"\s+(On|Off)$", "", gpu_name).strip()
            gpu_name = re.sub(r"\.\.\.\s*$", "", gpu_name).strip()

            if i + 1 < len(lines):
                stats_line = lines[i + 1]
                entry = {
                    "timestamp": timestamp,
                    "gpu_id": gpu_id,
                    "gpu_name": gpu_name,
                }

                temp_match = re.search(r"(\d+)C", stats_line)
                if temp_match:
                    entry["temperature_c"] = int(temp_match.group(1))

                power_match = re.search(r"(\d+)W\s*/\s*(\d+)W", stats_line)
                if power_match:
                    entry["power_w"] = int(power_match.group(1))
                    entry["power_cap_w"] = int(power_match.group(2))

                mem_match = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", stats_line)
                if mem_match:
                    entry["memory_used_mib"] = int(mem_match.group(1))
                    entry["memory_total_mib"] = int(mem_match.group(2))

                fan_match = re.search(r"\|\s*(\d+)%", stats_line)
                if fan_match:
                    entry["fan_pct"] = int(fan_match.group(1))

                sections = stats_line.split("|")
                if len(sections) >= 4:
                    util_section = sections[3]
                    util_match = re.search(r"(\d+)%", util_section)
                    if util_match:
                        entry["gpu_util_pct"] = int(util_match.group(1))

                gpu_entries.append(entry)
        i += 1

    return gpu_entries


def analyze_mode(log_path: str, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if not os.path.isfile(log_path):
        print(f"Error: log file not found: {log_path}")
        sys.exit(1)

    samples = parse_log(log_path)
    if not samples:
        print("Error: no valid GPU samples found in log file.")
        sys.exit(1)

    gpus: dict[int, dict] = {}
    for s in samples:
        gid = s["gpu_id"]
        if gid not in gpus:
            gpus[gid] = {
                "name": s.get("gpu_name", f"GPU {gid}"),
                "timestamps": [],
                "gpu_util": [],
                "memory_used": [],
                "memory_total": [],
                "temperature": [],
                "power": [],
                "power_cap": [],
            }
        g = gpus[gid]
        g["timestamps"].append(s["timestamp"])
        g["gpu_util"].append(s.get("gpu_util_pct", 0))
        g["memory_used"].append(s.get("memory_used_mib", 0))
        g["memory_total"].append(s.get("memory_total_mib", 0))
        g["temperature"].append(s.get("temperature_c", 0))
        g["power"].append(s.get("power_w", 0))
        g["power_cap"].append(s.get("power_cap_w", 0))

    num_gpus = len(gpus)

    global_t0 = min(g["timestamps"][0] for g in gpus.values())

    fig, axes = plt.subplots(4, 1, figsize=(14, 3.5 * 4), sharex=True)
    fig.suptitle("GPU Usage Over Time", fontsize=16, fontweight="bold", y=0.98)

    colors = plt.cm.tab10.colors

    for gid, g in sorted(gpus.items()):
        t = [(ts - global_t0) for ts in g["timestamps"]]
        color = colors[gid % len(colors)]
        label = f"GPU {gid}: {g['name']}"

        axes[0].plot(t, g["gpu_util"], label=label, color=color, linewidth=1.2)
        axes[0].fill_between(t, g["gpu_util"], alpha=0.15, color=color)

        mem_total = g["memory_total"][-1] if g["memory_total"] else 1
        mem_pct = [u / mem_total * 100 for u in g["memory_used"]]
        axes[1].plot(t, g["memory_used"], label=label, color=color, linewidth=1.2)
        axes[1].fill_between(t, g["memory_used"], alpha=0.15, color=color)

        axes[2].plot(t, g["temperature"], label=label, color=color, linewidth=1.2)

        axes[3].plot(t, g["power"], label=label, color=color, linewidth=1.2)
        axes[3].fill_between(t, g["power"], alpha=0.15, color=color)
        if g["power_cap"] and g["power_cap"][-1] > 0:
            axes[3].axhline(
                y=g["power_cap"][-1], color=color, linestyle="--",
                alpha=0.5, linewidth=0.8
            )

    axes[0].set_ylabel("GPU Util (%)")
    axes[0].set_ylim(-2, 105)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Memory (MiB)")
    for gid, g in sorted(gpus.items()):
        if g["memory_total"] and g["memory_total"][-1] > 0:
            axes[1].axhline(
                y=g["memory_total"][-1], color=colors[gid % len(colors)],
                linestyle="--", alpha=0.4, linewidth=0.8
            )
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Temp (°C)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("Power (W)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylim(bottom=0)
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[gpu_monitor] Plot saved to {output_path}")

    print("\n── Summary ─────────────────────────────────────")
    for gid, g in sorted(gpus.items()):
        duration = g["timestamps"][-1] - g["timestamps"][0] if len(g["timestamps"]) > 1 else 0
        print(f"  GPU {gid}: {g['name']}")
        print(f"    Samples   : {len(g['timestamps'])}  ({duration:.1f}s)")
        print(f"    GPU Util  : avg={sum(g['gpu_util'])/len(g['gpu_util']):.1f}%  "
              f"max={max(g['gpu_util'])}%")
        print(f"    Memory    : avg={sum(g['memory_used'])/len(g['memory_used']):.0f} MiB  "
              f"max={max(g['memory_used'])} MiB  "
              f"/ {g['memory_total'][-1]} MiB")
        print(f"    Temp      : avg={sum(g['temperature'])/len(g['temperature']):.1f}°C  "
              f"max={max(g['temperature'])}°C")
        print(f"    Power     : avg={sum(g['power'])/len(g['power']):.1f}W  "
              f"max={max(g['power'])}W  "
              f"/ {g['power_cap'][-1]}W cap")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="GPU Usage Visualizer — monitor and analyze nvidia-smi output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run train.sh                        # Monitor GPU while running train.sh
  %(prog)s run train.sh -o my_log.txt -i 0.5   # Custom output file and 0.5s interval
  %(prog)s analyze gpu_log.txt                  # Generate plot from log
  %(prog)s analyze gpu_log.txt -o report.png    # Custom output image
        """,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run_parser = subparsers.add_parser("run", help="Run a script while monitoring GPU")
    run_parser.add_argument("script", help="Path to the shell script to run")
    run_parser.add_argument(
        "-o", "--output", default="gpu_log.txt",
        help="Output log file (default: gpu_log.txt)",
    )
    run_parser.add_argument(
        "-i", "--interval", type=float, default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )

    analyze_parser = subparsers.add_parser("analyze", help="Analyze and visualize a GPU log")
    analyze_parser.add_argument("log", help="Path to the GPU log file")
    analyze_parser.add_argument(
        "-o", "--output", default="gpu_usage.png",
        help="Output plot image (default: gpu_usage.png)",
    )

    args = parser.parse_args()

    if args.mode == "run":
        exit_code = run_mode(args.script, args.output, args.interval)
        sys.exit(exit_code)
    elif args.mode == "analyze":
        analyze_mode(args.log, args.output)


if __name__ == "__main__":
    main()
