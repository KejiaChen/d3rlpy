from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math


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
    def plot_row_sum(cables_row, row_index=0):
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

            ax.set_title(cable)
            ax.set_xticks(x)
            ax.set_xticklabels(clip_ids)

            col_start += n_clips

        return row_axes

    # ------------------------------------------------
    #  plot_row for style="split"  (stacked bars)
    # ------------------------------------------------
    def plot_row_split(cables_row, row_index=0):
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

            ax.set_title(cable)
            ax.set_xticks(x)
            ax.set_xticklabels(clip_ids)

            col_start += n_clips

        return row_axes

    # ---------- 3) Choose plot_row based on style ----------
    if style == "sum":
        axes = plot_row_sum(cable_ids, 0)
    elif style == "split":
        axes = plot_row_split(cable_ids, 0)
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
