import argparse
import math
import os
from dataclasses import dataclass
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.use14corefonts": True,
        "ps.useafm": True,
        "axes.labelsize": 30,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
    }
)


@dataclass
class Episode:
    source: str
    trial_name: str
    obs: np.ndarray
    acts: np.ndarray
    terminals: np.ndarray
    fixing_start: int = 0
    fixing_finish: int = 0
    fixing_success: bool = False


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x

    def _movmean_1d(arr: np.ndarray) -> np.ndarray:
        # MATLAB-like centered moving mean with truncated edge windows (no zero padding).
        n = arr.shape[0]
        left = (window - 1) // 2
        right = window // 2
        csum = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            s = max(0, i - left)
            e = min(n, i + right + 1)
            out[i] = (csum[e] - csum[s]) / float(e - s)
        return out

    if x.ndim == 1:
        return _movmean_1d(x.astype(np.float64, copy=False))
    if x.ndim == 2:
        out = np.empty_like(x, dtype=np.float64)
        for c in range(x.shape[1]):
            out[:, c] = _movmean_1d(x[:, c].astype(np.float64, copy=False))
        return out
    raise ValueError(f"Unsupported array shape for moving average: {x.shape}")


def list_subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def load_fixing_episodes(fixing_dir: str, env_step: int) -> List[Episode]:
    _ = env_step  # kept for parity with MATLAB signature
    fixing_trials = list_subdirs(fixing_dir)
    episodes: List[Episode] = []

    for trial_name in fixing_trials:
        trial_path = os.path.join(fixing_dir, trial_name)
        mios_trials = list_subdirs(trial_path)
        if not mios_trials:
            print(f"[WARN] no mios subfolder under fixing trial: {trial_path}")
            continue

        mios_traj = mios_trials[0]
        mios_path = os.path.join(trial_path, mios_traj)
        try:
            obs, acts, fixing_start, fixing_finish, fixing_success = load_fixing_mios_episode_py(
                trial_path, mios_traj
            )
        except Exception as exc:
            print(f"[WARN] failed loading fixing trial {trial_name}/{mios_traj}: {exc}")
            continue

        terminals = np.zeros(obs.shape[0], dtype=bool)
        if fixing_finish >= 0 and fixing_finish < terminals.shape[0]:
            terminals[fixing_finish:] = True
        else:
            terminals[-1] = True

        if obs is None or acts is None or terminals is None:
            continue

        episodes.append(
            Episode(
                source="fixing",
                trial_name=trial_name,
                obs=obs,
                acts=acts,
                terminals=terminals,
                fixing_start=fixing_start,
                fixing_finish=fixing_finish,
                fixing_success=bool(fixing_success),
            )
        )
    return episodes


def _is_numeric_line(line: str) -> bool:
    parts = line.strip().split()
    if not parts:
        return False
    try:
        for p in parts:
            float(p)
        return True
    except ValueError:
        return False


def load_fixing_mios_episode_py(base_dir: str, traj_id: str):
    traj_path = os.path.join(base_dir, str(traj_id))
    ext_proj_path = os.path.join(traj_path, "Telepresence_Clip_TF_F_Ext_with_projection.txt")
    sensor_path = os.path.join(traj_path, "Telepresence_F_ext_sensor.txt")
    ff_path = os.path.join(traj_path, "Telepresence_Clip_cmd_F_ff.txt")

    with open(ext_proj_path, "r", encoding="utf-8") as f:
        ext_force_lines = f.readlines()

    stamp_map = {}
    for i, line in enumerate(ext_force_lines, start=1):
        if not _is_numeric_line(line):
            key = line.strip()
            stamp_map.setdefault(key, []).append(i)

    if "sensed" not in stamp_map:
        raise ValueError(f"'sensed' stamp not found in {ext_proj_path}")

    fixing_initial = stamp_map["sensed"][0]
    fixing_start = stamp_map["sensed"][0]

    if "finished" in stamp_map:
        fixing_finish_line = stamp_map["finished"][0]
        fixing_end = fixing_finish_line + 1000
        fixing_success = True
    elif "ff_finished" in stamp_map:
        fixing_finish_line = stamp_map["ff_finished"][0]
        fixing_end = fixing_finish_line
        fixing_success = False
    else:
        fixing_finish_line = None
        fixing_end = stamp_map.get("ended", [len(ext_force_lines)])[0]
        fixing_success = False

    stamp_count = len(stamp_map) - 1
    sensor_force = np.loadtxt(sensor_path, ndmin=2)
    start_idx = max(0, fixing_initial - 1)
    end_idx = max(start_idx + 1, fixing_end - stamp_count)
    end_idx = min(end_idx, sensor_force.shape[0])
    sensor_fix_force = sensor_force[start_idx:end_idx, :]
    sensor_force_filt = moving_average(sensor_fix_force, 150)

    feedforward_force = np.loadtxt(ff_path, ndmin=2)
    obs = np.column_stack([sensor_force_filt[:, 0], sensor_force_filt[:, 1]])
    acts = np.column_stack([feedforward_force[:, 0], feedforward_force[:, 1]])

    fixing_start_idx = fixing_start - fixing_initial - 1
    if fixing_finish_line is None:
        fixing_finish_idx = obs.shape[0] - 1
    else:
        fixing_finish_idx = fixing_finish_line - fixing_initial - 2
    fixing_start_idx = max(0, min(fixing_start_idx, obs.shape[0] - 1))
    fixing_finish_idx = max(0, min(fixing_finish_idx, obs.shape[0] - 1))

    print(f"Loaded fixing mios episode {traj_id} with {obs.shape[0]} steps.")
    return obs, acts, fixing_start_idx, fixing_finish_idx, fixing_success


def load_transport_episodes(
    transport_dir: str, obs_dim: int, act_dim: int, smoothing_window: int
) -> List[Episode]:
    transport_trials = list_subdirs(transport_dir)
    episodes: List[Episode] = []

    for trial_name in transport_trials:
        trial_dir = os.path.join(transport_dir, trial_name)
        force_file = os.path.join(trial_dir, "K_F_ext_K_s.txt")
        duration_file = os.path.join(trial_dir, "transport_stage.txt")

        if not os.path.isfile(force_file):
            print(f"[WARN] missing force file: {force_file}")
            continue

        raw_force = np.loadtxt(force_file, ndmin=2)
        if raw_force.shape[1] < 2:
            print(f"[WARN] force file has <2 columns: {force_file}")
            continue
        full_tension = raw_force[:, 1]

        if os.path.isfile(duration_file):
            raw_duration = np.loadtxt(duration_file, ndmin=2)
            if raw_duration.shape[1] >= 2:
                positive_idxs = np.flatnonzero(raw_duration[:, 1] > 0)
                start_idx = int(positive_idxs[0]) if positive_idxs.size else 0
            else:
                start_idx = 0
            tension = full_tension[start_idx:]
        else:
            tension = full_tension

        tension_filt = moving_average(tension, smoothing_window)
        n = tension_filt.shape[0]
        if n == 0:
            continue

        obs = np.zeros((n, obs_dim), dtype=np.float64)
        acts = np.zeros((n, act_dim), dtype=np.float64)
        terminals = np.zeros((n,), dtype=bool)
        obs[:, 0] = tension_filt
        terminals[-1] = True

        episodes.append(
            Episode(
                source="transport_tension",
                trial_name=trial_name,
                obs=obs,
                acts=acts,
                terminals=terminals,
            )
        )
    return episodes


def build_interleaved_plot_list(
    transport_eps: List[Episode], fixing_eps: List[Episode], n_pairs: int
) -> List[Episode]:
    max_pairs = min(len(transport_eps), len(fixing_eps))
    if max_pairs == 0:
        return []
    if n_pairs <= 0:
        use_pairs = max_pairs
    else:
        use_pairs = min(max_pairs, n_pairs)

    plot_list: List[Episode] = []
    for i in range(use_pairs):
        plot_list.append(transport_eps[i])
        plot_list.append(fixing_eps[i])
    return plot_list


def compute_segment_layout(
    plot_list: List[Episode],
    px_per_ms: float,
    topup_length: int,
    add_length: int,
    fixing_tail_ms: int,
) -> dict:
    finish_ms = []
    end_plot_ms = []
    x_end_ms = []
    width_px = []

    dt_ms_transport = 1.0
    dt_ms_fixing = 1.0

    for ep in plot_list:
        n = ep.obs.shape[0]
        if ep.source == "fixing":
            dt = dt_ms_fixing
            term_idx = ep.fixing_finish if ep.fixing_finish >= 0 else n - 1
            term_idx = min(term_idx, n - 1)
            range_end_idx = min(term_idx + fixing_tail_ms, n - 1)
            finish_val = term_idx * dt
            end_val = range_end_idx * dt
        else:
            dt = dt_ms_transport
            term_idxs = np.flatnonzero(ep.terminals)
            term_idx = int(term_idxs[-1]) if term_idxs.size else n - 1
            finish_val = term_idx * dt
            end_val = finish_val

        seg_x_end = math.ceil(end_val / topup_length) * topup_length + add_length
        seg_x_end = max(seg_x_end, 200)
        seg_w = max(1, int(round(seg_x_end * px_per_ms)))

        finish_ms.append(float(finish_val))
        end_plot_ms.append(float(end_val))
        x_end_ms.append(float(seg_x_end))
        width_px.append(int(seg_w))

    seg_x0 = []
    x_cursor = 0
    for w in width_px:
        seg_x0.append(x_cursor)
        x_cursor += w

    return {
        "finish_ms": np.asarray(finish_ms),
        "end_plot_ms": np.asarray(end_plot_ms),
        "x_end_ms": np.asarray(x_end_ms),
        "width_px": np.asarray(width_px),
        "seg_x0": np.asarray(seg_x0),
        "total_content_w": int(x_cursor),
    }


def draw_episode_segment(
    ax: plt.Axes,
    ep: Episode,
    finish_this: float,
    end_plot_this: float,
    x_end_this: float,
    ylim: tuple,
    linewidth: float,
    topup_length: int,
    show_ylabel: bool,
    show_xlabel: bool,
):
    col_stretch = (167 / 255.0, 139 / 255.0, 250 / 255.0) # "#ee7c8b"  # (93 / 255.0, 184 / 255.0, 71 / 255.0)
    col_push =  (238 / 255.0, 124 / 255.0, 139 / 255.0)   #"#9c6ade"  # (238 / 255.0, 124 / 255.0, 139 / 255.0)
    alpha_ext = 0.6

    ax.set_axisbelow(True)
    ax.grid(True, zorder=0)

    if ep.source == "fixing":
        n = ep.obs.shape[0]
        term_idx = min(ep.fixing_finish, n - 1)
        start_idx = min(max(ep.fixing_start, 0), term_idx)
        range_end_idx = min(int(end_plot_this), n - 1)
        ff_end_idx = min(term_idx, ep.acts.shape[0] - 1)
        ff_start_idx = min(start_idx, ff_end_idx) if ff_end_idx >= 0 else 0

        time_full = np.arange(0, range_end_idx + 1, dtype=np.float64)
        time_ff = np.arange(ff_start_idx, ff_end_idx + 1, dtype=np.float64)

        if time_ff.size > 0 and ep.acts.shape[0] > 0:
            ax.plot(
                time_ff,
                ep.acts[ff_start_idx : ff_end_idx + 1, 0],
                "--",
                color=col_stretch,
                linewidth=linewidth,
                zorder=2,
            )
            ax.plot(
                time_ff,
                ep.acts[ff_start_idx : ff_end_idx + 1, 1],
                "--",
                color=col_push,
                linewidth=linewidth,
                zorder=2,
            )

        ax.plot(
            time_full,
            ep.obs[: range_end_idx + 1, 0],
            "-",
            color=(*col_stretch, alpha_ext),
            linewidth=linewidth,
            zorder=2,
        )
        obs_push_idx = 1 if ep.obs.shape[1] > 1 else 0
        ax.plot(
            time_full,
            ep.obs[: range_end_idx + 1, obs_push_idx],
            "-",
            color=(*col_push, alpha_ext),
            linewidth=linewidth,
            zorder=2,
        )

        if ep.fixing_success:
            ax.axvline(finish_this, linestyle="-.", color="black", linewidth=linewidth, zorder=4)
    else:
        n = ep.obs.shape[0]
        term_idxs = np.flatnonzero(ep.terminals)
        term_idx = int(term_idxs[-1]) if term_idxs.size else n - 1
        t = np.arange(0, term_idx + 1, dtype=np.float64)
        ax.plot(
            t,
            ep.obs[: term_idx + 1, 0],
            "-",
            color=(*col_stretch, alpha_ext),
            linewidth=linewidth,
            zorder=2,
        )

    ax.set_xlim(0, x_end_this)
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(0, 31, 10))

    if x_end_this > end_plot_this:
        grid_lw = 1.0
        grid_lines = ax.yaxis.get_gridlines()
        if grid_lines:
            grid_lw = grid_lines[0].get_linewidth()
        # Light-gray mask region, then draw darker horizontal lines aligned with major Y ticks.
        ax.axvspan(end_plot_this, x_end_this, facecolor=(0.88, 0.88, 0.88), alpha=1.0, zorder=5)
        for y in ax.get_yticks():
            if ylim[0] <= y <= ylim[1]:
                ax.plot(
                    [end_plot_this, x_end_this],
                    [y, y],
                    color=(0.60, 0.60, 0.60),
                    linewidth=grid_lw,
                    zorder=6,
                )

    x_tick_max = math.floor(end_plot_this / topup_length) * topup_length
    if x_tick_max >= 0:
        ax.set_xticks(np.arange(0, x_tick_max + 1, topup_length))

    if show_ylabel:
        ax.set_ylabel("Force (N)")
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    if show_xlabel:
        ax.set_xlabel("Time (ms)")
    else:
        ax.set_xlabel("")


def export_pages(
    plot_list: List[Episode],
    layout: dict,
    out_stem: str,
    page_width_px: int,
    fig_height_px: int,
    dpi_png: int,
    topup_length: int,
    ylim: tuple,
    linewidth: float,
    pad_l: int,
    pad_r: int,
    pad_b: int,
    pad_t: int,
):
    finish_ms = layout["finish_ms"]
    end_plot_ms = layout["end_plot_ms"]
    x_end_ms = layout["x_end_ms"]
    width_px = layout["width_px"]
    seg_x0 = layout["seg_x0"]
    total_content_w = layout["total_content_w"]

    page_start = 0
    page_idx = 0
    while page_start < total_content_w:
        page_idx += 1
        page_end = min(page_start + page_width_px, total_content_w)

        in_page = (seg_x0 + width_px > page_start) & (seg_x0 < page_end)
        if not np.any(in_page):
            page_start = page_end
            continue

        left_most = int(np.min(seg_x0[in_page]))
        right_most = int(np.max(seg_x0[in_page] + width_px[in_page]))
        page_content_w = right_most - left_most
        page_fig_w = pad_l + page_content_w + pad_r
        page_fig_h = fig_height_px

        fig = plt.figure(figsize=(page_fig_w / 100.0, page_fig_h / 100.0), dpi=100)
        ax_h = fig_height_px - pad_b - pad_t
        first_visible = True

        for i, ep in enumerate(plot_list):
            if not in_page[i]:
                continue

            seg_start = int(seg_x0[i])
            seg_end = int(seg_x0[i] + width_px[i])
            ov_start = max(seg_start, page_start)
            ov_end = min(seg_end, page_end)
            ov_w = ov_end - ov_start
            if ov_w <= 0:
                continue

            x_px = pad_l + (ov_start - left_most)
            y_px = pad_b
            rect = [
                x_px / page_fig_w,
                y_px / page_fig_h,
                ov_w / page_fig_w,
                ax_h / page_fig_h,
            ]
            ax = fig.add_axes(rect)

            draw_episode_segment(
                ax=ax,
                ep=ep,
                finish_this=float(finish_ms[i]),
                end_plot_this=float(end_plot_ms[i]),
                x_end_this=float(x_end_ms[i]),
                ylim=ylim,
                linewidth=linewidth,
                topup_length=topup_length,
                show_ylabel=first_visible,
                show_xlabel=(i == len(plot_list) - 1 and ov_end == seg_end),
            )
            first_visible = False

        out_pdf = f"{out_stem}_page_{page_idx:02d}.pdf"
        out_png = f"{out_stem}_page_{page_idx:02d}.png"
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_png, dpi=dpi_png, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        print(
            f"Exported page {page_idx}: X=[{page_start}, {page_end}], segments={int(np.count_nonzero(in_page))}"
        )
        page_start = page_end


def export_legend_only(out_stem: str, linewidth: float, dpi_png: int):
    col_stretch = (93 / 255.0, 184 / 255.0, 71 / 255.0)
    col_push = (238 / 255.0, 124 / 255.0, 139 / 255.0)
    alpha_ext = 0.6

    fig = plt.figure(figsize=(18, 1.5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    labels = ["FF Stretch", "FF Push", "Ext Stretch", "Ext Push", "Finish"]
    styles = [
        {"color": col_stretch, "linestyle": "--", "alpha": 1.0},
        {"color": col_push, "linestyle": "--", "alpha": 1.0},
        {"color": col_stretch, "linestyle": "-", "alpha": alpha_ext},
        {"color": col_push, "linestyle": "-", "alpha": alpha_ext},
        {"color": (0.0, 0.0, 0.0), "linestyle": "-.", "alpha": 1.0},
    ]

    y_center = 0.5
    start_x = 0.03
    spacing = 0.205
    icon_len = 0.085
    icon_text_gap = 0.014
    legend_linewidth = max(2.0, (linewidth - 1.5) * 1.2)

    for i, (label, style) in enumerate(zip(labels, styles)):
        xc = start_x + i * spacing
        x1 = xc - icon_len / 2.0
        x2 = xc + icon_len / 2.0
        ax.plot(
            [x1, x2],
            [y_center, y_center],
            linestyle=style["linestyle"],
            linewidth=legend_linewidth,
            color=(*style["color"], style["alpha"]),
            solid_capstyle="butt",
        )
        ax.text(
            x2 + icon_text_gap,
            y_center,
            label,
            va="center",
            ha="left",
            fontsize=24,
        )

    out_pdf = f"{out_stem}_legend.pdf"
    out_png = f"{out_stem}_legend.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=dpi_png, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot interleaved transport/fixing force workflow.")
    parser.add_argument("--whole-run-root", default="/home/tp2/Documents/kejia/whole_run")
    parser.add_argument("--plan-scene", default="target_shape_plane5_10")
    parser.add_argument("--cable-name", default="B_L_L")
    parser.add_argument("--env-step", type=int, default=5)
    parser.add_argument("--n-pairs", type=int, default=3, help="<=0 means use all available pairs")
    parser.add_argument("--smoothing-window", type=int, default=150)
    parser.add_argument("--topup-length", type=int, default=1000)
    parser.add_argument("--add-length", type=int, default=1000)
    parser.add_argument("--fixing-tail-ms", type=int, default=1000)
    parser.add_argument("--px-per-ms", type=float, default=0.1)
    parser.add_argument("--page-width-px", type=int, default=12000)
    parser.add_argument("--fig-height-px", type=int, default=280)
    parser.add_argument("--dpi-png", type=int, default=300)
    args = parser.parse_args()

    base_dir = os.path.join(args.whole_run_root, args.plan_scene, args.cable_name)
    fixing_dir = os.path.join(base_dir, "fixing")
    transport_dir = os.path.join(base_dir, "transport")

    if not os.path.isdir(fixing_dir):
        raise FileNotFoundError(f"missing fixing dir: {fixing_dir}")
    if not os.path.isdir(transport_dir):
        raise FileNotFoundError(f"missing transport dir: {transport_dir}")

    print(f"Loading FIXING data from {fixing_dir}")
    fixing_eps = load_fixing_episodes(fixing_dir, env_step=args.env_step)
    if fixing_eps:
        obs_dim = fixing_eps[0].obs.shape[1]
        act_dim = fixing_eps[0].acts.shape[1]
    else:
        obs_dim, act_dim = 4, 2
    print(f"Loaded {len(fixing_eps)} fixing episodes")

    print(f"Loading TRANSPORT data from {transport_dir}")
    transport_eps = load_transport_episodes(
        transport_dir, obs_dim=obs_dim, act_dim=act_dim, smoothing_window=args.smoothing_window
    )
    print(f"Loaded {len(transport_eps)} transport episodes")

    plot_list = build_interleaved_plot_list(transport_eps, fixing_eps, n_pairs=args.n_pairs)
    if not plot_list:
        raise RuntimeError("No interleaved episodes available. Check fixing/transport trial folders.")

    layout = compute_segment_layout(
        plot_list,
        px_per_ms=args.px_per_ms,
        topup_length=args.topup_length,
        add_length=args.add_length,
        fixing_tail_ms=args.fixing_tail_ms,
    )

    out_stem = os.path.join(transport_dir, f"whole_process_{args.plan_scene}_{args.cable_name}")
    export_pages(
        plot_list=plot_list,
        layout=layout,
        out_stem=out_stem,
        page_width_px=args.page_width_px,
        fig_height_px=args.fig_height_px,
        dpi_png=args.dpi_png,
        topup_length=args.topup_length,
        ylim=(-5, 35),
        linewidth=5.0,
        pad_l=100,
        pad_r=0,
        pad_b=95,
        pad_t=20,
    )
    export_legend_only(out_stem=out_stem, linewidth=7.0, dpi_png=args.dpi_png)

    print("done")


if __name__ == "__main__":
    main()
