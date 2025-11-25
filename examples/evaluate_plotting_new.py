from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math

import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

mpl.rcParams['pdf.fonttype'] = 42  
mpl.rcParams['ps.fonttype'] = 42

FIG_TITLE_FONTSIZE = 28
SUBFIG_TITLE_FONTSIZE = 26
LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 24
TICK_FONTSIZE = 26
LINEWIDTH = 2


# ================================
#  SHARED AGGREGATION + STATS
# ================================

def _aggregate_data(random_returns, trained_returns, properties):
    """
    Aggregate raw returns for given properties into per-(cable, clip) structures.

    Returns:
        agg[(cable, clip)] = {
            "random": [sum_per_trial...],
            "trained": [...],
            "parts_random": {prop: [per-trial values...]},
            "parts_trained": {prop: [...]},
        }
    """
    properties = tuple(properties)

    agg = defaultdict(lambda: {
        "random": [],
        "trained": [],
        "parts_random": {p: [] for p in properties},
        "parts_trained": {p: [] for p in properties},
    })

    def add_policy(source, prefix):
        parts_key = f"parts_{prefix}"
        for group_key, data in source.items():
            # Derive cable_id and clip_id from group_key as you did
            cable_id = group_key[:7]
            clip_id  = group_key[8:13]
            key = (cable_id, clip_id)

            # Collect property lists
            prop_lists = []
            for p in properties:
                vals = [float(v) for v in data.get(p, [])]
                agg[key][parts_key][p].extend(vals)
                prop_lists.append(vals)

            # Skip if no data at all
            non_empty = [lst for lst in prop_lists if len(lst) > 0]
            if not non_empty:
                continue

            n = min(len(lst) for lst in non_empty)
            if n == 0:
                continue

            # Per-trial sum over properties
            totals = []
            for i in range(n):
                s = 0.0
                for lst in prop_lists:
                    if i < len(lst):
                        s += lst[i]
                totals.append(s)

            agg[key][prefix].extend(totals)

    add_policy(random_returns, "random")
    add_policy(trained_returns, "trained")

    return agg


def _compute_stats(agg, properties):
    """
    Convert agg into cable_to_clips structure:

        cable_to_clips[cable_id][clip_id] = {
            "mean_random", "std_random",
            "mean_trained", "std_trained",
            "mean_parts_random": np.array([...]),  # one per property
            "mean_parts_trained": np.array([...]),
        }
    """
    properties = tuple(properties)
    cable_to_clips = defaultdict(dict)

    for (cable, clip), d in agg.items():
        r = np.array(d["random"], dtype=float)
        t = np.array(d["trained"], dtype=float)

        if len(r) > 0:
            mean_r = float(np.mean(r))
            std_r  = float(np.std(r, ddof=0))
        else:
            mean_r = np.nan
            std_r  = np.nan

        if len(t) > 0:
            mean_t = float(np.mean(t))
            std_t  = float(np.std(t, ddof=0))
        else:
            mean_t = np.nan
            std_t  = np.nan

        mean_parts_r = []
        mean_parts_t = []
        for p in properties:
            vals_r = d["parts_random"][p]
            vals_t = d["parts_trained"][p]
            mean_parts_r.append(np.mean(vals_r) if len(vals_r) > 0 else np.nan)
            mean_parts_t.append(np.mean(vals_t) if len(vals_t) > 0 else np.nan)

        cable_to_clips[cable][clip] = {
            "mean_random":        mean_r,
            "std_random":         std_r,
            "mean_trained":       mean_t,
            "std_trained":        std_t,
            "mean_parts_random":  np.array(mean_parts_r, dtype=float),
            "mean_parts_trained": np.array(mean_parts_t, dtype=float),
        }

    return cable_to_clips

import matplotlib.colors as mcolors

def _with_alpha(color, alpha=0.4):
    """
    Return the input color but with a different transparency (alpha).
    Accepts any Matplotlib color spec: name, hex, RGB tuple, etc.
    """
    rgba = mcolors.to_rgba(color)
    return (rgba[0], rgba[1], rgba[2], alpha)



# ================================
#  MAIN CONFIGURABLE PLOTTING FUNC
# ================================

def plot_avg_returns_per_cable(
    random_returns,
    trained_returns,
    labels=("Random", "Trained"),
    properties=("total_energy", "total_effort"),
    style="sum",   # "sum" or "split"
    title="Average returns per clip for each cable",
    save_path="avg_returns_per_cable.png",
):
    """
    Unified plotting function.

    - Aggregation & statistics are shared.
    - Two different plot_row implementations:
        style="sum"   → grouped bars (bar height = sum(properties))
        style="split" → stacked bars showing each property

    Args
    ----
    random_returns, trained_returns : dict
        Your episode results, keyed by group_key (cable+clip), each containing
        lists for the properties, e.g. "total_energy", "total_effort", etc.
    properties : tuple of str
        Which keys to sum/stack per trial.
    style : {"sum", "split"}
        "sum"   = single solid bar per policy (total of properties)
        "split" = stacked bar per policy, one layer per property.
    """

    properties = tuple(properties)

    # ---------- 1) Shared aggregation + stats ----------
    agg = _aggregate_data(random_returns, trained_returns, properties)
    cable_to_clips = _compute_stats(agg, properties)

    if not cable_to_clips:
        print("No valid data found.")
        return

    # ---------- 2) Layout: ONE ROW ----------
    cable_ids = sorted(cable_to_clips.keys())
    clips_per_cable = {cid: len(cable_to_clips[cid]) for cid in cable_ids}
    total_slots = sum(clips_per_cable.values())

    fig = plt.figure(figsize=(0.9 * total_slots + 3, 3.5))
    gs = fig.add_gridspec(1, total_slots, width_ratios=[1] * total_slots)

    bar_width = 0.35
    all_y_vals = []   # for global y-axis
    axes = []

    # ------------------------------------------------
    #  plot_row for style="sum"  (simple grouped bars)
    # ------------------------------------------------
    def plot_row_bar_sum(cables_row, row_index=0):
        nonlocal all_y_vals
        col_start = 0
        row_axes = []

        for cable in cables_row:
            n_clips = clips_per_cable[cable]
            if n_clips == 0:
                continue

            clip_ids = sorted(cable_to_clips[cable].keys())
            x = np.arange(n_clips)

            ax = fig.add_subplot(gs[row_index, col_start:col_start + n_clips])
            row_axes.append(ax)

            means_r = np.array(
                [cable_to_clips[cable][cid]["mean_random"] for cid in clip_ids],
                dtype=float,
            )
            std_r = np.array(
                [cable_to_clips[cable][cid]["std_random"] for cid in clip_ids],
                dtype=float,
            )
            means_t = np.array(
                [cable_to_clips[cable][cid]["mean_trained"] for cid in clip_ids],
                dtype=float,
            )
            std_t = np.array(
                [cable_to_clips[cable][cid]["std_trained"] for cid in clip_ids],
                dtype=float,
            )

            # collect for global y-limits
            for m, s in zip(means_r, std_r):
                if not np.isnan(m):
                    all_y_vals.append(m)
                    if not np.isnan(s):
                        all_y_vals.append(m + s)
                        all_y_vals.append(m - s)
            for m, s in zip(means_t, std_t):
                if not np.isnan(m):
                    all_y_vals.append(m)
                    if not np.isnan(s):
                        all_y_vals.append(m + s)
                        all_y_vals.append(m - s)

            # grouped bars
            ax.bar(
                x - bar_width / 2,
                means_r,
                width=bar_width,
                yerr=std_r,
                capsize=3,
                label=labels[0],
            )
            ax.bar(
                x + bar_width / 2,
                means_t,
                width=bar_width,
                yerr=std_t,
                capsize=3,
                label=labels[1],
            )

            if row_index == 0:
                ax.set_title(cable, fontsize=SUBFIG_TITLE_FONTSIZE, y=1.05)
            ax.set_xticks(x)
            ax.set_xticklabels(clip_ids)

            col_start += n_clips

        return row_axes

    # ------------------------------------------------
    #  plot_row for style="split"  (stacked bars)
    # ------------------------------------------------
    def plot_row_bar_split(cables_row, row_index=0):
        nonlocal all_y_vals
        col_start = 0
        row_axes = []

        def pretty(name: str) -> str:
            return name.replace("_", " ").capitalize()

        for cable in cables_row:
            n_clips = clips_per_cable[cable]
            if n_clips == 0:
                continue

            clip_ids = sorted(cable_to_clips[cable].keys())
            x = np.arange(n_clips)

            ax = fig.add_subplot(gs[row_index, col_start:col_start + n_clips])
            row_axes.append(ax)

            stats_for_clips = [cable_to_clips[cable][cid] for cid in clip_ids]

            mean_r  = np.array([s["mean_random"]  for s in stats_for_clips], dtype=float)
            std_r   = np.array([s["std_random"]   for s in stats_for_clips], dtype=float)
            mean_t  = np.array([s["mean_trained"] for s in stats_for_clips], dtype=float)
            std_t   = np.array([s["std_trained"]  for s in stats_for_clips], dtype=float)

            parts_r = np.vstack([s["mean_parts_random"]  for s in stats_for_clips]).T
            parts_t = np.vstack([s["mean_parts_trained"] for s in stats_for_clips]).T

            # collect for global y-limits (use total mean ± std)
            for m, s in zip(mean_r, std_r):
                if not np.isnan(m):
                    all_y_vals.append(m)
                    if not np.isnan(s):
                        all_y_vals.append(m + s)
                        all_y_vals.append(m - s)
            for m, s in zip(mean_t, std_t):
                if not np.isnan(m):
                    all_y_vals.append(m)
                    if not np.isnan(s):
                        all_y_vals.append(m + s)
                        all_y_vals.append(m - s)

            # stacked bars
            bottoms_r = np.zeros_like(x, dtype=float)
            bottoms_t = np.zeros_like(x, dtype=float)

            for i, prop in enumerate(properties):
                label_r = f"{labels[0]} – {pretty(prop)}"
                label_t = f"{labels[1]} – {pretty(prop)}"

                ax.bar(
                    x - bar_width / 2,
                    parts_r[i],
                    width=bar_width,
                    bottom=bottoms_r,
                    label=label_r,
                )
                ax.bar(
                    x + bar_width / 2,
                    parts_t[i],
                    width=bar_width,
                    bottom=bottoms_t,
                    label=label_t,
                )

                bottoms_r += parts_r[i]
                bottoms_t += parts_t[i]

            # total error bars on top of stacks
            ax.errorbar(
                x - bar_width / 2,
                mean_r,
                yerr=std_r,
                fmt="none",
                capsize=3,
                ecolor="black",
            )
            ax.errorbar(
                x + bar_width / 2,
                mean_t,
                yerr=std_t,
                fmt="none",
                capsize=3,
                ecolor="black",
            )

            if row_index == 0:
                ax.set_title(cable, fontsize=SUBFIG_TITLE_FONTSIZE)
            ax.set_xticks(x)
            ax.set_xticklabels(clip_ids, fontsize=TICK_FONTSIZE)

            col_start += n_clips

        return row_axes

    # ---------- 3) Choose plot_row based on style ----------
    if style == "sum":
        axes = plot_row_bar_sum(cable_ids, 0)
    elif style == "split":
        axes = plot_row_bar_split(cable_ids, 0)
    else:
        raise ValueError(f"Unknown style '{style}', expected 'sum' or 'split'")

    # ---------- 4) Shared y-axis handling ----------
    if all_y_vals and axes:
        ymax = max(all_y_vals)
        if ymax <= 0:
            ymax = 1.0
        ymax *= 1.05

        for ax in axes:
            ax.set_ylim(0, ymax)

        fig.canvas.draw()
        ref_ticks = axes[0].get_yticks()
        ref_ticklabels = [t.get_text() for t in axes[0].get_yticklabels()]

        for i, ax in enumerate(axes):
            ax.set_yticks(ref_ticks)
            if i == 0:
                ax.set_yticklabels(ref_ticklabels)
            else:
                ax.set_yticklabels([])

    fig.supylabel("Average score")
    fig.suptitle(title)

    # ---------- 5) Legend ----------
    handles = []
    label_list = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        label_list.extend(l)

    # deduplicate labels
    uniq = {}
    for h, l in zip(handles, label_list):
        if l not in uniq:
            uniq[l] = h

    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        top=0.85,
        bottom=0.12,
        wspace=0.4,
    )

    plt.savefig(save_path)
    plt.show()


def plot_avg_returns_per_cable_4rows(
    row_configs,
    labels=("Random", "Trained"),
    title="Average returns per clip for each cable",
    save_path="avg_returns_per_cable_4rows.png",
):
    """
    Plot up to 4 rows of cable/clip bar charts on one figure.

    Each row has its own:
      - random_returns   : dict
      - trained_returns  : dict
      - properties       : tuple of keys to use (summed/stacked)
      - style            : "sum" or "split"

    All rows:
      - share the SAME cable/clip layout on x (union of all rows' data)
      - only the BOTTOM row shows x tick labels
      - each row has its OWN y scale and y ticks (left axis only)

    row_configs is a list of dicts, e.g.:

      row_configs = [
          {
            "random_returns": random_returns_1,
            "trained_returns": trained_returns_1,
            "properties": ("total_energy", "total_effort"),
            "style": "sum",
          },
          {
            "random_returns": random_returns_2,
            "trained_returns": trained_returns_2,
            "properties": ("total_energy",),
            "style": "sum",
          },
          {
            "random_returns": random_returns_3,
            "trained_returns": trained_returns_3,
            "properties": ("stretch_energy", "push_energy"),
            "style": "split",
          },
          {
            "random_returns": random_returns_4,
            "trained_returns": trained_returns_4,
            "properties": ("total_effort",),
            "style": "split",
          },
      ]
    """

    n_rows = min(4, len(row_configs))
    if n_rows == 0:
        print("No row_configs provided.")
        return

    # ---------- 1) Build shared geometry (cables + clips) across ALL rows ----------
    geom = defaultdict(set)

    def add_geom_from_returns(ret_dict):
        for group_key in ret_dict.keys():
            cable_id = group_key[:7]
            clip_id  = group_key[8:13]
            geom[cable_id].add(clip_id)

    for cfg in row_configs[:n_rows]:
        add_geom_from_returns(cfg["random_returns"])
        add_geom_from_returns(cfg["trained_returns"])

    if not geom:
        print("No valid cable/clip combinations found across all rows.")
        return

    cable_ids = sorted(geom.keys())
    clip_ids_by_cable = {cid: sorted(list(geom[cid])) for cid in cable_ids}
    clips_per_cable = {cid: len(clip_ids_by_cable[cid]) for cid in cable_ids}
    total_slots = sum(clips_per_cable.values())

    # ---------- 2) Figure & GridSpec ----------
    fig_height_per_row = 3.0
    fig = plt.figure(figsize=(0.9 * total_slots + 4, fig_height_per_row * n_rows))
    gs = fig.add_gridspec(n_rows, total_slots, width_ratios=[1] * total_slots)

    bar_width = 0.35
    axes_by_row = []
    yvals_by_row = []

    # ---------- 3) row-specific plot_row helpers ----------

    def plot_row_bar_sum(row_stats, row_index):
        """Grouped bars: bar height = sum(properties)."""
        row_axes = []
        row_yvals = []
        col_start = 0

        for cable in cable_ids:
            n_clips = clips_per_cable[cable]
            if n_clips == 0:
                continue

            clip_ids = clip_ids_by_cable[cable]
            x = np.arange(n_clips)

            ax = fig.add_subplot(gs[row_index, col_start:col_start + n_clips])
            row_axes.append(ax)

            means_r = []
            std_r   = []
            means_t = []
            std_t   = []

            for clip in clip_ids:
                stats = row_stats.get(cable, {}).get(clip, None)
                if stats is None:
                    means_r.append(np.nan)
                    std_r.append(np.nan)
                    means_t.append(np.nan)
                    std_t.append(np.nan)
                else:
                    means_r.append(stats["mean_random"])
                    std_r.append(stats["std_random"])
                    means_t.append(stats["mean_trained"])
                    std_t.append(stats["std_trained"])

            means_r = np.array(means_r, dtype=float)
            std_r   = np.array(std_r, dtype=float)
            means_t = np.array(means_t, dtype=float)
            std_t   = np.array(std_t, dtype=float)

            # collect row y values (mean ± std)
            for m, s in zip(means_r, std_r):
                if not np.isnan(m):
                    row_yvals.append(m)
                    if not np.isnan(s):
                        row_yvals.append(m + s)
                        row_yvals.append(m - s)
            for m, s in zip(means_t, std_t):
                if not np.isnan(m):
                    row_yvals.append(m)
                    if not np.isnan(s):
                        row_yvals.append(m + s)
                        row_yvals.append(m - s)

            # grouped bars
            ax.bar(
                x - bar_width / 2,
                means_r,
                width=bar_width,
                yerr=std_r,
                capsize=3,
            )
            ax.bar(
                x + bar_width / 2,
                means_t,
                width=bar_width,
                yerr=std_t,
                capsize=3,
            )

            if row_index == 0:
                ax.set_title(cable, fontsize=SUBFIG_TITLE_FONTSIZE, y=1.05)
            ax.set_xticks(x)
            show_clip_ids = [clip_ids[i][2:] for i in range(n_clips)]
            ax.set_xticklabels(show_clip_ids, fontsize=TICK_FONTSIZE)

            col_start += n_clips

        return row_axes, row_yvals

    def plot_row_bar_split(row_stats, row_index, properties, color_random="#1f77b4", color_trained="#ff7f0e"):
        """Stacked bars: each property as a layer, total shown via error bars.
        Upper / lower parts are distinguished by hatch patterns.
        """
        row_axes = []
        row_yvals = []
        col_start = 0
        properties = tuple(properties)
        n_props = len(properties)

        # hatch patterns for each stacked layer
        # extend this list if you ever have >5 properties
        hatch_patterns = ["///", "xxx", "...", "\\\\"]
        hatch_patterns = hatch_patterns[:n_props]

        for cable in cable_ids:
            n_clips = clips_per_cable[cable]
            if n_clips == 0:
                continue

            clip_ids = clip_ids_by_cable[cable]
            x = np.arange(n_clips)

            ax = fig.add_subplot(gs[row_index, col_start:col_start + n_clips])
            row_axes.append(ax)

            means_r = []
            std_r   = []
            means_t = []
            std_t   = []
            parts_r_list = []
            parts_t_list = []

            for clip in clip_ids:
                stats = row_stats.get(cable, {}).get(clip, None)
                if stats is None:
                    means_r.append(np.nan)
                    std_r.append(np.nan)
                    means_t.append(np.nan)
                    std_t.append(np.nan)
                    parts_r_list.append(np.full(n_props, np.nan))
                    parts_t_list.append(np.full(n_props, np.nan))
                else:
                    means_r.append(stats["mean_random"])
                    std_r.append(stats["std_random"])
                    means_t.append(stats["mean_trained"])
                    std_t.append(stats["std_trained"])
                    parts_r_list.append(stats["mean_parts_random"])
                    parts_t_list.append(stats["mean_parts_trained"])

            means_r = np.array(means_r, dtype=float)
            std_r   = np.array(std_r, dtype=float)
            means_t = np.array(means_t, dtype=float)
            std_t   = np.array(std_t, dtype=float)
            parts_r = np.vstack(parts_r_list).T   # shape: (n_props, n_clips)
            parts_t = np.vstack(parts_t_list).T

            # collect row y values (total mean ± std)
            for m, s in zip(means_r, std_r):
                if not np.isnan(m):
                    row_yvals.append(m)
                    if not np.isnan(s):
                        row_yvals.append(m + s)
                        row_yvals.append(m - s)
            for m, s in zip(means_t, std_t):
                if not np.isnan(m):
                    row_yvals.append(m)
                    if not np.isnan(s):
                        row_yvals.append(m + s)
                        row_yvals.append(m - s)

            bottoms_r = np.zeros_like(x, dtype=float)
            bottoms_t = np.zeros_like(x, dtype=float)

            for i in range(n_props):
                hatch = hatch_patterns[i]

                ax.bar(
                    x - bar_width / 2,
                    parts_r[i],
                    width=bar_width,
                    bottom=bottoms_r,
                    color=_with_alpha(color_random, alpha=0.4),
                    hatch=hatch,
                    edgecolor=color_random,
                    linewidth=2,
                )
                ax.bar(
                    x + bar_width / 2,
                    parts_t[i],
                    width=bar_width,
                    bottom=bottoms_t,
                    color=_with_alpha(color_trained, alpha=0.4),
                    hatch=hatch,
                    edgecolor=color_trained,
                    linewidth=2,
                )

                bottoms_r += parts_r[i]
                bottoms_t += parts_t[i]

            # total error bars
            ax.errorbar(
                x - bar_width / 2,
                means_r,
                yerr=std_r,
                fmt="none",
                capsize=3,
                ecolor="black",
            )
            ax.errorbar(
                x + bar_width / 2,
                means_t,
                yerr=std_t,
                fmt="none",
                capsize=3,
                ecolor="black",
            )

            if row_index == 0:
                ax.set_title(cable, fontsize=SUBFIG_TITLE_FONTSIZE, y=1.05)
            ax.set_xticks(x)
            show_clip_ids = [clip_ids[i][2:] for i in range(n_clips)]
            ax.set_xticklabels(show_clip_ids, fontsize=TICK_FONTSIZE)

            col_start += n_clips

        return row_axes, row_yvals
    
    def plot_row_violin(row_stats, row_index, 
                        color_random="#1f77b4", 
                        color_trained="#ff7f0e"):
        """
        Row of violin plots, with custom colors.
        `color_random` and `color_trained` specify the violin fill colors.
        """
        row_axes = []
        row_yvals = []
        col_start = 0

        offset = bar_width * 0.4
        violin_width = bar_width * 0.7

        for cable in cable_ids:
            n_clips = clips_per_cable[cable]
            if n_clips == 0:
                continue

            clip_ids = clip_ids_by_cable[cable]
            x = np.arange(n_clips)

            ax = fig.add_subplot(gs[row_index, col_start:col_start+n_clips])
            row_axes.append(ax)

            random_datasets = []
            random_positions = []
            trained_datasets = []
            trained_positions = []

            # Collect values
            for xi, clip in enumerate(clip_ids):
                stats = row_stats.get(cable, {}).get(clip)
                if not stats:
                    continue

                rv = np.asarray(stats.get("random_values", []), dtype=float)
                tv = np.asarray(stats.get("trained_values", []), dtype=float)

                if rv.size > 0:
                    random_datasets.append(rv)
                    random_positions.append(xi - offset)
                    row_yvals.extend(rv.tolist())

                if tv.size > 0:
                    trained_datasets.append(tv)
                    trained_positions.append(xi + offset)
                    row_yvals.extend(tv.tolist())

            # Draw and color violins
            if random_datasets:
                parts_r = ax.violinplot(
                    random_datasets,
                    positions=random_positions,
                    widths=violin_width,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for pc in parts_r['bodies']:
                    pc.set_facecolor(color_random)
                    # pc.set_edgecolor("black")
                    pc.set_alpha(0.4)

            if trained_datasets:
                parts_t = ax.violinplot(
                    trained_datasets,
                    positions=trained_positions,
                    widths=violin_width,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for pc in parts_t['bodies']:
                    pc.set_facecolor(color_trained)
                    # pc.set_edgecolor("black")
                    pc.set_alpha(0.4)

            # Add thick mean lines manually
            for vals, xpos in zip(random_datasets, random_positions):
                ax.hlines(np.mean(vals), xpos - violin_width*0.45,
                        xpos + violin_width*0.45,
                        color=color_random, linewidth=2.5, zorder=5)

            for vals, xpos in zip(trained_datasets, trained_positions):
                ax.hlines(np.mean(vals), xpos - violin_width*0.45,
                        xpos + violin_width*0.45,
                        color=color_trained, linewidth=2.5, zorder=5)

            if row_index == 0:
                ax.set_title(cable, fontsize=SUBFIG_TITLE_FONTSIZE, y=1.05)
            ax.set_xticks(x)
            show_clip_ids = [clip_ids[i][2:] for i in range(n_clips)]
            ax.set_xticklabels(show_clip_ids, fontsize=TICK_FONTSIZE)

            col_start += n_clips

        return row_axes, row_yvals


    # ---------- 4) Build each row ----------
    for row_index in range(n_rows):
        cfg = row_configs[row_index]
        row_random  = cfg["random_returns"]
        row_trained = cfg["trained_returns"]
        props       = tuple(cfg["properties"])
        style       = cfg.get("style", "sum")
        y_label     = cfg.get("ylabel", f"Row {row_index + 1} Average Score")

        agg_row   = _aggregate_data(row_random, row_trained, props)
        stats_row = _compute_stats(agg_row, props)

        row_raw_values = defaultdict(dict)
        for (cable, clip), d in agg_row.items():
            r_vals = d["random"]
            t_vals = d["trained"]

            # store only full distributions
            row_raw_values[cable][clip] = {
                "random_values":  r_vals,
                "trained_values": t_vals,
            }

        if style == "sum":
            row_axes, row_yvals = plot_row_bar_sum(stats_row, row_index)
        elif style == "split":
            row_axes, row_yvals = plot_row_bar_split(stats_row, row_index, props)
        elif style == "violin":
            row_axes, row_yvals = plot_row_violin(row_raw_values, row_index, color_random="#1f77b4", color_trained="#ff7f0e")
        else:
            raise ValueError(f"Unknown style '{style}', expected 'sum' or 'split'.")

        axes_by_row.append(row_axes)
        yvals_by_row.append(row_yvals)

    # ---------- 5) Per-row y-axis, shared bottom x labels ----------
    fig.canvas.draw()

    for row_index, (row_axes, row_yvals) in enumerate(zip(axes_by_row, yvals_by_row)):
        if not row_axes:
            continue

        # Per-row y limits
        if row_yvals:
            ymax = max(row_yvals)
            if ymax <= 0:
                ymax = 1.0
            ymax *= 1.05
            for ax in row_axes:
                ax.set_ylim(0, ymax)

        # Per-row y ticks (left axis only)
        ref_ax = row_axes[0]
        ref_ticks = ref_ax.get_yticks()
        ref_ticklabels = [t.get_text() for t in ref_ax.get_yticklabels()]

        for j, ax in enumerate(row_axes):
            ax.set_yticks(ref_ticks)
            if j == 0:
                ax.set_yticklabels(ref_ticklabels, fontsize=TICK_FONTSIZE)
            else:
                ax.set_yticklabels([])

        ref_ax.set_ylabel(cfg.get("ylabel", row_configs[row_index].get("y_label", "Average score")), fontsize=SUBFIG_TITLE_FONTSIZE)
        # move y-label further left
        ref_ax.yaxis.set_label_coords(-0.4, 0.5)

        # X tick labels only on bottom row
        if row_index < n_rows - 1:
            for ax in row_axes:
                ax.set_xticklabels([])

    fig.suptitle(title, fontsize=FIG_TITLE_FONTSIZE)
    # fig.supylabel("Average score")

    # -------------------------------------------------------
    # ONE ROW LEGEND AT THE BOTTOM
    # -------------------------------------------------------
    import matplotlib.patches as mpatches

    # Policy patches (colors only)
    patch_random  = mpatches.Patch(
        facecolor=_with_alpha("#1f77b4", 0.4),
        edgecolor="#1f77b4",
        label="Random"
    )
    patch_trained = mpatches.Patch(
        facecolor=_with_alpha("#ff7f0e", 0.4),
        edgecolor="#ff7f0e",
        label="Trained"
    )

    # Component patches (hatch only, neutral color)
    patch_ciu = mpatches.Patch(
        facecolor="white",
        edgecolor="black",
        hatch="///",
        label=r"$c_{i,u}$"
    )
    patch_cit = mpatches.Patch(
        facecolor="white",
        edgecolor="black",
        hatch="xxx",
        label=r"$c_{i,t}$"
    )

    legend_handles = [patch_random, patch_trained, patch_ciu, patch_cit]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,                  # one row
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=LEGEND_FONTSIZE,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.08,
        hspace=0.2,
        wspace=0.4,
    )

    plt.savefig(save_path)
    plt.show()

