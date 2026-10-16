#!/usr/bin/env python3
"""Benchmark sweep for the Blackhole AXPY stencil.

Runs the already built `axpy_stencil` executable over a grid of problem sizes, iteration counts and
core layouts, parses its timing output, stores every run in a CSV and draws the plots. Every run of the
program times three versions of the same sweeps (the problem and the CPU versions are the ones of
optimized_blackhole, so results compare directly with bench_jacobi.py):
  - golden:          the CPU reference, single-threaded, fp32 math rounded to bf16 like the device
  - golden_opt:      the optimised CPU baseline, plain fp32 on 32 threads (sweeps only)
  - Tenstorrent:     per sweep, the device run, the copy of the result back to the host, the host
                     building the shifted UP/LEFT/RIGHT/DOWN copies and writing them to the device

The default sweep is the one of the EuroSys'26 report "Stencil Computations on Tenstorrent Wormhole"
(Sec. 5.1): N x N grids with N in {1024, 2048, 4096, 8192, 16384, 30720}, 100 / 500 / 1000 iterations.
By default every size runs on the full Blackhole grid (11 x 10); the program uses fewer cores when the
grid has fewer 32 x 32 tiles than that.

Examples:
  ./bench_axpy.py                                    # full report sweep, 3 repetitions each
  ./bench_axpy.py --sizes 1024 2048 --iters 100      # a quick subset
  ./bench_axpy.py --cores 1x1 2x2 4x4 8x8 max        # also produce a core-scaling plot
  ./bench_axpy.py --resume results/20261015_1530     # finish an interrupted sweep
  ./bench_axpy.py --plot-only results/20261015_1530
"""

import argparse
import csv
import datetime
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Report sweep (Sec. 5.1 and run.sh). Table 2 also uses 128 x 128.
REPORT_SIZES = [1024, 2048, 4096, 8192, 16384, 30720]
REPORT_ITERS = [100, 500, 1000]
DEFAULT_CORES = ["max"]
MAX_GRID = "11x10"  # Blackhole compute grid (the program checks it against the device's)

TILE = 32  # sizes are multiples of 32, so N x N is a whole number of 32 x 32 tiles

CSV_FIELDS = [
    "size", "its", "cx", "cy", "cores", "rep", "status", "returncode",
    "total_s", "xfer_on_s", "exec_s", "xfer_off_s", "host_s", "exec_gpts", "total_gpts",
    "golden_s", "golden_opt_s", "cpu_threads", "log",
]
TIME_FIELDS = ["total_s", "xfer_on_s", "exec_s", "xfer_off_s", "host_s", "exec_gpts", "total_gpts", "golden_s", "golden_opt_s"]
# Computed per run from the fields above, then summarised the same way (see derived_metrics).
DERIVED_FIELDS = ["golden_gpts", "golden_opt_gpts", "speedup_exec_vs_golden", "speedup_exec_vs_opt",
                  "speedup_total_vs_opt"]

# Lines printed by tt_main.cpp.
RE_TIMES = re.compile(
    r"Run completed, total time ([\d.eE+-]+) sec\. ([\d.eE+-]+) sec transfer on, "
    r"([\d.eE+-]+) sec execution, ([\d.eE+-]+) sec transfer off, ([\d.eE+-]+) sec host")
RE_CORES_USED = re.compile(r"Using (\d+) of \d+ x \d+ cores")
RE_EXEC_GPTS = re.compile(r"Execution GPts: ([\d.eE+-]+)")
RE_TOTAL_GPTS = re.compile(r"Total GPts: ([\d.eE+-]+)")
RE_GOLDEN = re.compile(r"Calculated reference version over \d+ iterations in ([\d.eE+-]+) secs")
RE_GOLDEN_OPT = re.compile(
    r"Calculated optimized fp32 version over \d+ iterations on (\d+) threads in ([\d.eE+-]+) secs")
RE_VERIFY = re.compile(r"^(Match|Error): ", re.MULTILINE)


def parse_cores(text):
    if text == "max":
        return text
    m = re.fullmatch(r"(\d+)[xX](\d+)", text)
    if not m:
        raise argparse.ArgumentTypeError(f"core layout must look like 8x8 or be 'max', got {text!r}")
    return int(m.group(1)), int(m.group(2))


def resolve_cores(layout, size, max_grid):
    """The CX x CY cores `layout` uses for an N = `size` grid: 'max' is the whole `max_grid` (the
    program itself leaves cores idle when there are fewer tiles than cores)."""
    return tuple(max_grid) if layout == "max" else layout


def layout_name(layout, max_grid):
    return f"max (up to {max_grid[0]}x{max_grid[1]})" if layout == "max" else f"{layout[0]}x{layout[1]}"


def layout_tag(layout):
    return "max" if layout == "max" else f"{layout[0]}x{layout[1]}"


def parse_output(out):
    """Return the timing fields found in one run's stdout (missing ones are absent)."""
    rec = {}
    m = RE_TIMES.search(out)
    if m:
        rec["total_s"], rec["xfer_on_s"], rec["exec_s"], rec["xfer_off_s"], rec["host_s"] = map(float, m.groups())
    m = RE_CORES_USED.search(out)
    if m:
        rec["cores"] = int(m.group(1))
    m = RE_EXEC_GPTS.search(out)
    if m:
        rec["exec_gpts"] = float(m.group(1))
    m = RE_TOTAL_GPTS.search(out)
    if m:
        rec["total_gpts"] = float(m.group(1))
    m = RE_GOLDEN.search(out)
    if m:
        rec["golden_s"] = float(m.group(1))
    m = RE_GOLDEN_OPT.search(out)
    if m:
        rec["cpu_threads"] = int(m.group(1))
        rec["golden_opt_s"] = float(m.group(2))
    m = RE_VERIFY.search(out)
    if m:
        rec["verify"] = m.group(1)
    return rec


def run_one(exe, size, its, cx, cy, rep, timeout, log_dir):
    cmd = [str(exe), str(its), str(size), str(size), str(cx), str(cy)]
    log = log_dir / f"axpy_{size}_{its}_{cx}x{cy}_r{rep}.log"
    env = dict(os.environ)
    env.setdefault("TT_METAL_DEVICE_PROFILER", "0")

    # "cores" is replaced by the number the program reports using, when it gets that far.
    rec = {"size": size, "its": its, "cx": cx, "cy": cy, "cores": cx * cy, "rep": rep, "log": log.name}
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              timeout=timeout, env=env)
        out, rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout) or ""
        rc = None
    log.write_text(f"$ {' '.join(cmd)}\n{out}")

    parsed = parse_output(out)
    rec.update({k: v for k, v in parsed.items() if k in CSV_FIELDS})
    rec["returncode"] = "" if rc is None else rc
    if rc is None:
        rec["status"] = "timeout"
    elif "exec_s" not in parsed:
        rec["status"] = "crash"
    elif parsed.get("verify") == "Error":
        rec["status"] = "mismatch"
    elif rc != 0:
        rec["status"] = "fail"
    else:
        rec["status"] = "ok"
    return rec


def load_csv(path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ("size", "its", "cx", "cy", "cores", "rep"):
                r[k] = int(r[k])
            if r.get("golden_s") is None:  # older CSVs called it baseline_s
                r["golden_s"] = r.get("baseline_s")
            for k in TIME_FIELDS:
                r[k] = float(r[k]) if r.get(k) not in (None, "") else None
            r["cpu_threads"] = int(r["cpu_threads"]) if r.get("cpu_threads") not in (None, "") else None
            rows.append(r)
    return rows


def run_key(r):
    return r["size"], r["its"], r["cx"], r["cy"], r["rep"]


def latest_rows(rows):
    """Keep only the last row of every run, so a retried run replaces its failed attempt."""
    return list({run_key(r): r for r in rows}.values())


def sweep(args, out_dir):
    exe = Path(args.exe).resolve()
    if not exe.is_file() or not os.access(exe, os.X_OK):
        sys.exit(f"executable not found: {exe} (build it first, or pass --exe)")
    if "TT_METAL_RUNTIME_ROOT" not in os.environ:
        print("warning: TT_METAL_RUNTIME_ROOT is not set; axpy_stencil may not find the tt-metal runtime",
              file=sys.stderr)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "runs.csv"
    # Failed runs are retried; their new row supersedes the old one (see latest_rows).
    done = {run_key(r) for r in load_csv(csv_path) if r["status"] == "ok"}

    plan = []
    for layout in args.cores:
        for size in args.sizes:
            cx, cy = resolve_cores(layout, size, args.max_grid)
            if size < TILE or size % TILE:
                print(f"skip {size}x{size}: size must be a multiple of 32", file=sys.stderr)
                continue
            if cx > args.max_grid[0] or cy > args.max_grid[1]:
                print(f"skip {size}x{size} on {cx}x{cy}: the compute grid is "
                      f"{args.max_grid[0]}x{args.max_grid[1]}", file=sys.stderr)
                continue
            for its in args.iters:
                for rep in range(args.reps):
                    key = (size, its, cx, cy, rep)
                    if key not in done and key not in plan:  # 'max' can coincide with an explicit layout
                        plan.append(key)

    new_file = not csv_path.exists()
    if not new_file:
        with csv_path.open() as f:
            header = next(csv.reader(f), [])
        if header != CSV_FIELDS:  # written by an older version of this script: rewrite with the new columns
            old_rows = load_csv(csv_path)
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(old_rows)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for i, (size, its, cx, cy, rep) in enumerate(plan, 1):
            print(f"[{i}/{len(plan)}] N={size} its={its} cores={cx}x{cy} rep={rep} ... ",
                  end="", flush=True)
            rec = run_one(exe, size, its, cx, cy, rep, args.timeout, log_dir)
            writer.writerow(rec)
            f.flush()
            if rec["status"] == "ok":
                cpu = "".join(f" {label} {rec[k]:.4f}s" for k, label in
                              (("golden_s", "golden"), ("golden_opt_s", "golden_opt")) if k in rec)
                print(f"exec {rec['exec_s']:.4f}s total {rec['total_s']:.4f}s "
                      f"({rec['exec_gpts']:.3f} GPts/s){cpu}")
            else:
                print(f"{rec['status'].upper()} (see logs/{rec['log']})")
    return csv_path


def derived_metrics(r):
    """CPU throughputs and device speedups of one run (None where a time is missing)."""
    points = r["size"] * r["size"] * r["its"]

    def ratio(a, b):
        return a / b if a is not None and b else None
    return {
        "golden_gpts": ratio(points / 2**30, r["golden_s"]),
        "golden_opt_gpts": ratio(points / 2**30, r["golden_opt_s"]),
        "speedup_exec_vs_golden": ratio(r["golden_s"], r["exec_s"]),
        "speedup_exec_vs_opt": ratio(r["golden_opt_s"], r["exec_s"]),
        "speedup_total_vs_opt": ratio(r["golden_opt_s"], r["total_s"]),
    }


PLOTTABLE_STATUSES = ("ok", "mismatch")  # mismatch runs still have valid timings


def summarise(rows):
    """Median and min/max over repetitions of each (size, its, cx, cy), plottable runs only."""
    groups = {}
    for r in rows:
        if r["status"] in PLOTTABLE_STATUSES:
            groups.setdefault((r["size"], r["its"], r["cx"], r["cy"]), []).append(r)
    summary = []
    for (size, its, cx, cy), rs in sorted(groups.items()):
        s = {"size": size, "its": its, "cx": cx, "cy": cy, "cores": rs[-1]["cores"], "n": len(rs),
             "cpu_threads": rs[-1]["cpu_threads"]}
        rs = [{**r, **derived_metrics(r)} for r in rs]
        for k in TIME_FIELDS + DERIVED_FIELDS:
            vals = [r[k] for r in rs if r[k] is not None]  # CPU times are missing in older CSVs
            s[k] = statistics.median(vals) if vals else None
            s[k + "_min"] = min(vals) if vals else None
            s[k + "_max"] = max(vals) if vals else None
        summary.append(s)
    return summary


def write_summary(summary, path):
    if not summary:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


# ---------------------------------------------------------------- plots

# Categorical slots 1-4 of the reference palette (validated, colorblind-safe; slots 3-4 are under 3:1
# contrast on white, so every line also gets a direct label and its own marker).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    })
    return plt


def size_label(n):
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def series_colors(values):
    """Fixed color per series value (follows the entity, never cycled past the palette)."""
    return {v: SERIES[i] if i < len(SERIES) else INK_2 for i, v in enumerate(sorted(values))}


def direct_labels(ax, ends, min_gap_px=12):
    """Label each line at its last point, pushed apart vertically so close lines don't overprint.
    `ends` is a list of (x, y, text). Call once the axis scales and limits are final."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    placed = sorted(((to_px((x, y))[1], x, y, text) for x, y, text in ends))
    targets = []
    for py, *_ in placed:
        targets.append(max(py, targets[-1] + min_gap_px) if targets else py)
    to_points = 72.0 / fig.dpi
    for (py, x, y, text), ty in zip(placed, targets):
        ax.annotate(text, (x, y), xytext=(6, (ty - py) * to_points), textcoords="offset points",
                    va="center", color=INK_2, fontsize=8, annotation_clip=False)


def errbars(rows, key):
    mid = [r[key] for r in rows]
    return [[m - r[key + "_min"] for m, r in zip(mid, rows)],
            [r[key + "_max"] - m for m, r in zip(mid, rows)]]


def layout_rows(summary, layout, max_grid):
    """Summary rows run with `layout`, i.e. on the layout it resolves to for each size."""
    return [s for s in summary if (s["cx"], s["cy"]) == resolve_cores(layout, s["size"], max_grid)]


def plot_vs_size(plt, rows, name, key, ylabel, title, path, logy=True, ref=None):
    rows = [s for s in rows if s[key] is not None]
    if not rows:
        return
    its_values = sorted({s["its"] for s in rows})
    colors = series_colors(its_values)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ends = []
    for its in its_values:
        rs = sorted((s for s in rows if s["its"] == its), key=lambda s: s["size"])
        x = [s["size"] for s in rs]
        y = [s[key] for s in rs]
        ax.errorbar(x, y, yerr=errbars(rs, key), color=colors[its], marker="o", markersize=6,
                    capsize=3, label=f"{its} iterations")
        ends.append((x[-1], y[-1], f"{its} it."))
    if ref is not None:
        ax.axhline(ref, color=INK_2, linewidth=1, linestyle="--")
    sizes = sorted({s["size"] for s in rows})
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([size_label(n) for n in sizes])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Grid size N (N x N points)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - {name} cores", loc="left")
    ax.legend(loc="best")
    direct_labels(ax, ends)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def plot_breakdown_share(plt, rows, name, path):
    """Host-to-device memcpy / execution / device-to-host memcpy / host shuffle as a share of total time
    (like Fig. 6 of the report)."""
    rows = sorted(rows, key=lambda s: (s["its"], s["size"]))
    if not rows:
        return
    parts = [("xfer_on_s", "Host-to-device memcpy"), ("exec_s", "Execution"), ("xfer_off_s", "Device-to-host memcpy"),
             ("host_s", "Host shuffle")]
    labels = [f"{s['its']}\n{size_label(s['size'])}" for s in rows]
    fig, ax = plt.subplots(figsize=(max(6.4, 0.5 * len(rows) + 2), 4.2))
    bottom = [0.0] * len(rows)
    for (key, label), color in zip(parts, SERIES):
        vals = [100 * (s[key] or 0) / s["total_s"] for s in rows]
        ax.bar(range(len(rows)), vals, 0.7, bottom=bottom, color=color, label=label,
               edgecolor="white", linewidth=1)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Iterations / grid size N")
    ax.set_ylabel("Share of total time (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Time share by phase - {name} cores", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def plot_cpu_vs_device(plt, rows, name, its, path):
    """Throughput of the device (execution and end to end) and of both CPU versions, one iteration count."""
    rows = sorted((s for s in rows if s["its"] == its), key=lambda s: s["size"])
    if not rows:
        return
    threads = rows[-1]["cpu_threads"]
    series = [("exec_gpts", "Tenstorrent, execution"), ("total_gpts", "Tenstorrent, end to end"),
              ("golden_opt_gpts", f"CPU fp32, {threads} threads"), ("golden_gpts", "CPU reference, 1 thread")]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ends = []
    for (key, label), color, marker in zip(series, SERIES, MARKERS):
        rs = [s for s in rows if s[key] is not None]
        if not rs:
            continue
        x = [s["size"] for s in rs]
        y = [s[key] for s in rs]
        ax.errorbar(x, y, yerr=errbars(rs, key), color=color, marker=marker, markersize=6, capsize=3,
                    label=label)
        ends.append((x[-1], y[-1], label))
    sizes = [s["size"] for s in rows]
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([size_label(n) for n in sizes])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Grid size N (N x N points)")
    ax.set_ylabel("Throughput (GPts/s, log)")
    ax.set_title(f"Tenstorrent vs CPU - {its} iterations, {name} cores", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    direct_labels(ax, ends)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def plot_core_scaling(plt, summary, its, path):
    """Execution throughput against the number of cores, one line per grid size."""
    rows = [s for s in summary if s["its"] == its]
    if len({s["cores"] for s in rows}) < 2:
        return
    sizes = sorted({s["size"] for s in rows})
    # One line per size; with more than three sizes, fall back to a sequential ramp of one hue.
    import matplotlib.colors as mcolors
    ramp = mcolors.LinearSegmentedColormap.from_list("seq", ["#a9c9f0", "#0d3a75"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, size in enumerate(sizes):
        rs = sorted((s for s in rows if s["size"] == size), key=lambda s: s["cores"])
        color = ramp(i / max(1, len(sizes) - 1))
        x = [s["cores"] for s in rs]
        y = [s["exec_gpts"] for s in rs]
        ax.errorbar(x, y, yerr=errbars(rs, "exec_gpts"), color=color, marker="o", markersize=6,
                    capsize=3, label=f"N = {size_label(size)}")
    cores = sorted({s["cores"] for s in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(cores)
    ax.set_xticklabels([str(c) for c in cores])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Tensix cores")
    ax.set_ylabel("Execution throughput (GPts/s)")
    ax.set_title(f"Core scaling - {its} iterations", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def make_plots(csv_path, layouts, max_grid):
    out_dir = csv_path.parent
    rows = latest_rows(load_csv(csv_path))
    bad = [r for r in rows if r["status"] not in PLOTTABLE_STATUSES]
    if bad:
        print(f"{len(bad)} of {len(rows)} runs did not succeed and are left out of the plots:")
        for r in bad:
            print(f"  N={r['size']} its={r['its']} cores={r['cx']}x{r['cy']} rep={r['rep']}: "
                  f"{r['status']} ({r['log']})")
    summary = summarise(rows)
    if not summary:
        print("no successful runs to plot")
        return
    write_summary(summary, out_dir / "summary.csv")
    print(f"wrote {out_dir / 'summary.csv'}")

    plt = setup_matplotlib()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for layout in layouts:
        rows, name, tag = layout_rows(summary, layout, max_grid), layout_name(layout, max_grid), layout_tag(layout)
        plot_vs_size(plt, rows, name, "exec_s", "Execution time (s, log)",
                     "Kernel execution time", plot_dir / f"exec_time_{tag}.png")
        plot_vs_size(plt, rows, name, "total_s", "Total time (s, log)",
                     "Total time (memcpys + host + execution)", plot_dir / f"total_time_{tag}.png")
        plot_vs_size(plt, rows, name, "exec_gpts", "Throughput (GPts/s)",
                     "Execution throughput", plot_dir / f"exec_gpts_{tag}.png", logy=False)
        plot_vs_size(plt, rows, name, "total_gpts", "Throughput (GPts/s)",
                     "End-to-end throughput", plot_dir / f"total_gpts_{tag}.png", logy=False)
        plot_breakdown_share(plt, rows, name, plot_dir / f"breakdown_share_{tag}.png")
        plot_vs_size(plt, rows, name, "speedup_exec_vs_opt", "Speedup (x, log)",
                     "Execution vs 32-thread fp32 CPU", plot_dir / f"speedup_exec_vs_opt_{tag}.png", ref=1)
        plot_vs_size(plt, rows, name, "speedup_total_vs_opt", "Speedup (x, log)",
                     "End to end vs 32-thread fp32 CPU", plot_dir / f"speedup_total_vs_opt_{tag}.png", ref=1)
        plot_vs_size(plt, rows, name, "speedup_exec_vs_golden", "Speedup (x, log)",
                     "Execution vs CPU reference", plot_dir / f"speedup_exec_vs_golden_{tag}.png", ref=1)
        for its in sorted({s["its"] for s in rows}):
            plot_cpu_vs_device(plt, rows, name, its, plot_dir / f"cpu_vs_tt_{tag}_{its}.png")
    for its in sorted({s["its"] for s in summary}):
        plot_core_scaling(plt, summary, its, plot_dir / f"core_scaling_{its}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exe", default=str(HERE / "axpy_stencil"), help="axpy_stencil executable (default: %(default)s)")
    p.add_argument("--sizes", type=int, nargs="+", default=REPORT_SIZES, help="grid sizes N (N x N)")
    p.add_argument("--iters", type=int, nargs="+", default=REPORT_ITERS, help="iteration counts")
    p.add_argument("--cores", type=parse_cores, nargs="+", default=[parse_cores(c) for c in DEFAULT_CORES],
                   help="core layouts CXxCY, or 'max' for the whole compute grid "
                        "(default: max); with --plot-only, the layouts to plot")
    p.add_argument("--max-grid", type=parse_cores, default=parse_cores(MAX_GRID),
                   help="compute grid of the device, the upper bound of 'max' (default: 11x10)")
    p.add_argument("--reps", type=int, default=3, help="repetitions per configuration (default: 3)")
    # One run also times both CPU versions: the reference alone takes over an hour at 30720 x 30720, 1000 its.
    p.add_argument("--timeout", type=float, default=4 * 3600, help="seconds per run (default: %(default)s)")
    p.add_argument("--out", type=Path, help="output directory (default: results/<timestamp>)")
    p.add_argument("--resume", type=Path, metavar="DIR", help="continue a sweep in DIR, skipping finished runs")
    p.add_argument("--plot-only", type=Path, metavar="DIR", help="only redraw the plots from DIR/runs.csv")
    args = p.parse_args()

    if args.plot_only:
        make_plots(args.plot_only / "runs.csv", args.cores, args.max_grid)
        return
    out_dir = args.resume or args.out or HERE / "results" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"results in {out_dir}")
    csv_path = sweep(args, out_dir)
    make_plots(csv_path, args.cores, args.max_grid)


if __name__ == "__main__":
    main()
