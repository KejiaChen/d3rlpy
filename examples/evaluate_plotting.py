import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

FIG_TITLE_FONTSIZE = 24
SUBFIG_TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 20
TICK_FONTSIZE = 20
LINEWIDTH = 2


def plot_avg_returns_per_cable_bar(
    random_returns,
    trained_returns,
    labels=("Random", "Trained"),
    properties=("total_energy", "total_effort"),
    title="Average returns per clip for each cable",
    save_path="avg_returns_per_cable.png"
):
    """
    One 2D bar subplot per cable_id (average over trials per clip_id).

    - X axis: clip_id
    - Bars: random vs trained
    - Each bar value = average over trials of the SUM of `properties`
        e.g. total_energy + total_effort, or whatever you pass in.
    - Error bars: std of that sum across trials
    - Exactly 2 rows of subplots.
    - Subplot width is proportional to number of clips.
    - Bars have the same visual width across ALL subplots
      (achieved by a single 2xN GridSpec of uniform 'clip slots').

    Assumes each entry in random_returns / trained_returns looks like:
        {
            "cable_id": ...,
            "clip_id": ...,
            "<prop1>": [ ... ],   # per trial
            "<prop2>": [ ... ],
            ...
        }
    where <propX> are the names in `properties`.
    """

    properties = tuple(properties)  # just to be safe

    # 1) Aggregate raw data: (cable_id, clip_id) -> list of summed values per trial
    agg = defaultdict(lambda: {
        "random_total": [],
        "trained_total": [],
    })

    def add_policy_data(source_dict, policy_prefix):
        for group_key, data in source_dict.items():
            # cable_id = data.get("cable_id")
            # clip_id  = data.get("clip_id")
            cable_id = group_key[:7]
            clip_id  = group_key[8:13]
            if cable_id is None or clip_id is None:
                continue

            # collect lists for all requested properties
            prop_lists = []
            for prop in properties:
                vals = data.get(prop, [])
                vals = [float(v) for v in vals]
                prop_lists.append(vals)

            # drop if no property has data
            non_empty = [lst for lst in prop_lists if len(lst) > 0]
            if not non_empty:
                continue

            # ensure same number of trials by truncating to min length
            n_trials = min(len(lst) for lst in non_empty)
            if n_trials == 0:
                continue

            # compute per-trial sum over properties
            summed = []
            for i in range(n_trials):
                s = 0.0
                for lst in prop_lists:
                    if i < len(lst):
                        s += lst[i]
                summed.append(s)

            key = (cable_id, clip_id)
            agg[key][f"{policy_prefix}_total"].extend(summed)

    # Random + trained
    add_policy_data(random_returns,  "random")
    add_policy_data(trained_returns, "trained")

    # 2) Per-cable structure: mean/std of the summed properties
    # cable_id -> clip_id -> dict(...)
    cable_to_clips = defaultdict(dict)
    for (cable_id, clip_id), d in agg.items():
        # --- random ---
        r_tot = d["random_total"]
        if r_tot:
            mean_r = float(np.mean(r_tot))
            std_r  = float(np.std(r_tot, ddof=0))
        else:
            mean_r, std_r = np.nan, np.nan

        # --- trained ---
        t_tot = d["trained_total"]
        if t_tot:
            mean_t = float(np.mean(t_tot))
            std_t  = float(np.std(t_tot, ddof=0))
        else:
            mean_t, std_t = np.nan, np.nan

        cable_to_clips[cable_id][clip_id] = {
            "mean_random":  mean_r,
            "std_random":   std_r,
            "mean_trained": mean_t,
            "std_trained":  std_t,
        }

    if not cable_to_clips:
        print("No (cable_id, clip_id) data found.")
        return

    # 3) Layout: 2 rows, width proportional to #clips
    cable_ids = sorted(cable_to_clips.keys())
    n_cables = len(cable_ids)

    split_idx = math.ceil(n_cables / 2)
    row0_cables = cable_ids[:split_idx]
    row1_cables = cable_ids[split_idx:]

    clips_per_cable = {cid: len(cable_to_clips[cid]) for cid in cable_ids}

    gap_slots = 0

    def row_slots(cables_row):
        if not cables_row:
            return 1
        return sum(clips_per_cable[cid] for cid in cables_row)

    total_clips_row0 = row_slots(row0_cables)
    total_clips_row1 = row_slots(row1_cables)
    total_slots = max(total_clips_row0, total_clips_row1)

    # Wider figure: more horizontal room overall
    fig = plt.figure(figsize=(0.9 * total_slots + 7, 9))
    gs = fig.add_gridspec(2, total_slots, width_ratios=[1] * total_slots)

    bar_width = 0.35
    axes_row0 = []
    axes_row1 = []

    # We'll collect all values (means ± std) to compute a global y-range
    all_y_vals = []

    def plot_row(cable_ids_row, row_index):
        """Place each cable in this row, spanning slots equal to its #clips, with gaps."""
        nonlocal all_y_vals
        row_axes = []
        col_start = 0

        for _, cable_id in enumerate(cable_ids_row):
            n_clips = clips_per_cable[cable_id]
            if n_clips == 0:
                continue

            col_end = col_start + n_clips
            ax = fig.add_subplot(gs[row_index, col_start:col_end])
            row_axes.append(ax)

            clip_dict = cable_to_clips[cable_id]
            clip_ids = sorted(clip_dict.keys())
            x = np.arange(n_clips)

            means_random = []
            std_random = []
            means_trained = []
            std_trained = []

            for clip_id in clip_ids:
                stats = clip_dict[clip_id]
                means_random.append(stats["mean_random"])
                std_random.append(stats["std_random"])
                means_trained.append(stats["mean_trained"])
                std_trained.append(stats["std_trained"])

            means_random = np.array(means_random, dtype=float)
            std_random = np.array(std_random, dtype=float)
            means_trained = np.array(means_trained, dtype=float)
            std_trained = np.array(std_trained, dtype=float)

            # collect values for global y-limits (mean ± std)
            for m, s in zip(means_random, std_random):
                if not np.isnan(m):
                    all_y_vals.append(m)
                    if not np.isnan(s):
                        all_y_vals.append(m + s)
                        all_y_vals.append(m - s)
            for m, s in zip(means_trained, std_trained):
                if not np.isnan(m):
                    all_y_vals.append(m)
                    if not np.isnan(s):
                        all_y_vals.append(m + s)
                        all_y_vals.append(m - s)

            # Bars + error bars (sum of properties)
            ax.bar(
                x - bar_width / 2,
                means_random,
                width=bar_width,
                yerr=std_random,
                capsize=3,
                label=labels[0],
            )
            ax.bar(
                x + bar_width / 2,
                means_trained,
                width=bar_width,
                yerr=std_trained,
                capsize=3,
                label=labels[1],
            )

            ax.set_title(f"{cable_id}", fontsize=SUBFIG_TITLE_FONTSIZE)
            ax.set_xticks(x)
            ax.set_xticklabels(clip_ids, rotation=0, ha="center", fontsize=TICK_FONTSIZE)
            ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

            # move start to end + gap_slots
            col_start = col_end + gap_slots

        return row_axes

    axes_row0 = plot_row(row0_cables, 0)
    axes_row1 = plot_row(row1_cables, 1)
    axes = axes_row0 + axes_row1

    # ------------- GLOBAL Y AXIS FOR ALL SUBPLOTS --------------------
    if all_y_vals and axes:
        global_max = max(all_y_vals)
        if global_max <= 0:
            global_max = 1.0

        # Global y limits: always start at zero for bar charts
        ylim = (0, global_max * 1.05)

        # Apply the same limits to all axes
        for ax in axes:
            ax.set_ylim(ylim)

        # Force draw so ticks are computed
        fig.canvas.draw()

        # Take ticks from the first axis as global ticks
        ref_ax = axes[0]
        ref_ticks = ref_ax.get_yticks()
        ref_ticklabels = [lab.get_text() for lab in ref_ax.get_yticklabels()]

        # Apply same ticks everywhere
        for ax in axes:
            ax.set_yticks(ref_ticks)
            ax.set_yticklabels(ref_ticklabels, fontsize=TICK_FONTSIZE)

        # Only leftmost subplot in each row shows labels
        if len(axes_row0) > 1:
            for ax in axes_row0[1:]:
                ax.set_yticklabels([])

        if len(axes_row1) > 1:
            for ax in axes_row1[1:]:
                ax.set_yticklabels([])

    fig.supylabel("Average score", fontsize=FIG_TITLE_FONTSIZE)
    fig.suptitle(title, y=0.95, fontsize=FIG_TITLE_FONTSIZE)

    # Global legend with more right margin
    fig.legend(
        labels,
        loc="center right",
        bbox_to_anchor=(0.95, 0.8),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )

    # Layout: keep margins for ylabel and legend, and add vertical spacing
    fig.subplots_adjust(
        left=0.06,   # room for y-label
        right=0.95,  # space on the right for legend
        top=0.85,
        bottom=0.08,
        hspace=0.3,
        wspace=0.5,
    )

    plt.savefig(save_path)
    plt.show()

def plot_avg_returns_per_cable_bar_split(
    random_returns,
    trained_returns,
    labels=("Random", "Trained"),
    properties=("total_energy", "total_effort"),
    title="Average returns per clip for each cable",
    save_path="avg_returns_per_cable.png"
):
    """
    One 2D bar subplot per cable_id (average over trials per clip_id).

    - X axis: clip_id
    - Bars: random vs trained
    - Each bar is a stack of the given `properties`, e.g. ("total_energy", "total_effort")
    - Error bars: std of the TOTAL (sum over `properties`) across trials
    - Exactly 2 rows of subplots.
    - Subplot width is proportional to number of clips.
    - Bars have the same visual width across ALL subplots
      (achieved by a single 2xN GridSpec of uniform 'clip slots').

    Assumes each entry in random_returns / trained_returns looks like:
        {
            "cable_id": ...,
            "clip_id": ...,
            "<prop1>": [ ... ],   # per trial
            "<prop2>": [ ... ],
            ...
        }
    where <propX> are the names in `properties`.
    """

    properties = tuple(properties)  # ensure it's indexable

    def make_empty():
        d = {}
        for policy in ("random", "trained"):
            for p in properties:
                d[f"{policy}_{p}"] = []
        return d

    # 1) Aggregate raw data: (cable_id, clip_id) -> lists for each policy and property
    agg = defaultdict(make_empty)

    def add_policy_data(source_dict, policy_prefix):
        for _, data in source_dict.items():
            cable_id = data.get("cable_id")
            clip_id  = data.get("clip_id")
            if cable_id is None or clip_id is None:
                continue

            key = (cable_id, clip_id)
            for prop in properties:
                vals = data.get(prop, [])
                agg[key][f"{policy_prefix}_{prop}"].extend([float(v) for v in vals])

    # Random + trained
    add_policy_data(random_returns,  "random")
    add_policy_data(trained_returns, "trained")

    # 2) Per-cable structure: totals + parts
    # cable_id -> clip_id -> dict(...)
    cable_to_clips = defaultdict(dict)

    for (cable_id, clip_id), d in agg.items():
        # --- helper for one policy ---
        def compute_policy_stats(prefix):
            # list of arrays, one per property
            arrays = [np.array(d[f"{prefix}_{prop}"], dtype=float)
                      for prop in properties]

            # keep only non-empty arrays to determine common length
            non_empty = [a for a in arrays if len(a) > 0]
            if not non_empty:
                return np.nan, np.nan, np.full(len(properties), np.nan, dtype=float)

            # use the minimum length across properties (truncate to align)
            n = min(len(a) for a in non_empty)
            arrays = [a[:n] for a in arrays]

            # shape = (n_properties, n_trials)
            mat = np.vstack(arrays)

            # per-trial total = sum over properties
            totals = mat.sum(axis=0)

            mean_total = float(np.mean(totals))
            std_total  = float(np.std(totals, ddof=0))
            means_parts = mat.mean(axis=1)  # one mean per property

            return mean_total, std_total, means_parts

        mean_r, std_r, mean_r_parts = compute_policy_stats("random")
        mean_t, std_t, mean_t_parts = compute_policy_stats("trained")

        cable_to_clips[cable_id][clip_id] = {
            "mean_random":        mean_r,
            "std_random":         std_r,
            "mean_trained":       mean_t,
            "std_trained":        std_t,
            "mean_random_parts":  mean_r_parts,   # len = len(properties)
            "mean_trained_parts": mean_t_parts,
        }

    if not cable_to_clips:
        print("No (cable_id, clip_id) data found.")
        return

    # 3) Layout: 2 rows, width proportional to #clips
    cable_ids = sorted(cable_to_clips.keys())
    n_cables = len(cable_ids)

    split_idx = math.ceil(n_cables / 2)
    row0_cables = cable_ids[:split_idx]
    row1_cables = cable_ids[split_idx:]

    clips_per_cable = {cid: len(cable_to_clips[cid]) for cid in cable_ids}

    gap_slots = 0

    def row_slots(cables_row):
        if not cables_row:
            return 1
        return sum(clips_per_cable[cid] for cid in cables_row)

    total_clips_row0 = row_slots(row0_cables)
    total_clips_row1 = row_slots(row1_cables)
    total_slots = max(total_clips_row0, total_clips_row1)

    fig = plt.figure(figsize=(0.9 * total_slots + 7, 9))
    gs = fig.add_gridspec(2, total_slots, width_ratios=[1] * total_slots)

    bar_width = 0.35
    axes_row0 = []
    axes_row1 = []

    all_y_vals = []

    def prettify(name: str) -> str:
        return name.replace("_", " ").capitalize()

    def plot_row(cable_ids_row, row_index):
        nonlocal all_y_vals
        row_axes = []
        col_start = 0

        for _, cable_id in enumerate(cable_ids_row):
            n_clips = clips_per_cable[cable_id]
            if n_clips == 0:
                continue

            col_end = col_start + n_clips
            ax = fig.add_subplot(gs[row_index, col_start:col_end])
            row_axes.append(ax)

            clip_dict = cable_to_clips[cable_id]
            clip_ids = sorted(clip_dict.keys())
            x = np.arange(n_clips)

            means_random = []
            std_random = []
            means_trained = []
            std_trained = []

            for clip_id in clip_ids:
                stats = clip_dict[clip_id]
                means_random.append(stats["mean_random"])
                std_random.append(stats["std_random"])
                means_trained.append(stats["mean_trained"])
                std_trained.append(stats["std_trained"])

            means_random = np.array(means_random, dtype=float)
            std_random = np.array(std_random, dtype=float)
            means_trained = np.array(means_trained, dtype=float)
            std_trained = np.array(std_trained, dtype=float)

            # global y-limits based on mean ± std
            for m, s in zip(means_random, std_random):
                if not np.isnan(m):
                    all_y_vals.append(m)
                    if not np.isnan(s):
                        all_y_vals.append(m + s)
                        all_y_vals.append(m - s)
            for m, s in zip(means_trained, std_trained):
                if not np.isnan(m):
                    all_y_vals.append(m)
                    if not np.isnan(s):
                        all_y_vals.append(m + s)
                        all_y_vals.append(m - s)

            # --- STACKED BARS FOR N PROPERTIES ---

            x_random  = x - bar_width / 2
            x_trained = x + bar_width / 2

            # shape: (n_properties, n_clips)
            means_random_parts  = np.vstack(
                [clip_dict[cid]["mean_random_parts"] for cid in clip_ids]
            ).T
            means_trained_parts = np.vstack(
                [clip_dict[cid]["mean_trained_parts"] for cid in clip_ids]
            ).T

            part_labels = [prettify(p) for p in properties]
            n_parts = means_random_parts.shape[0]

            bottom_r = np.zeros_like(x, dtype=float)
            bottom_t = np.zeros_like(x, dtype=float)

            for i in range(n_parts):
                ax.bar(
                    x_random,
                    means_random_parts[i],
                    width=bar_width,
                    bottom=bottom_r,
                    label=f"{labels[0]} – {part_labels[i]}",
                )
                ax.bar(
                    x_trained,
                    means_trained_parts[i],
                    width=bar_width,
                    bottom=bottom_t,
                    label=f"{labels[1]} – {part_labels[i]}",
                )

                bottom_r += means_random_parts[i]
                bottom_t += means_trained_parts[i]

            # --- TOTAL ERROR BARS ON TOP OF STACKS ---
            ax.errorbar(
                x_random,
                means_random,
                yerr=std_random,
                fmt="none",
                capsize=3,
                elinewidth=2.5,
                ecolor="black",
            )
            ax.errorbar(
                x_trained,
                means_trained,
                yerr=std_trained,
                fmt="none",
                capsize=3,
                elinewidth=2.5,
                ecolor="black",
            )

            ax.set_title(f"{cable_id}", fontsize=SUBFIG_TITLE_FONTSIZE)
            ax.set_xticks(x)
            ax.set_xticklabels(clip_ids, rotation=0, ha="center", fontsize=TICK_FONTSIZE)
            ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

            col_start = col_end + gap_slots

        return row_axes

    axes_row0 = plot_row(row0_cables, 0)
    axes_row1 = plot_row(row1_cables, 1)
    axes = axes_row0 + axes_row1

    # ------------- GLOBAL Y AXIS FOR ALL SUBPLOTS --------------------
    if all_y_vals and axes:
        global_max = max(all_y_vals)
        if global_max <= 0:
            global_max = 1.0

        ylim = (0, global_max * 1.05)

        for ax in axes:
            ax.set_ylim(ylim)

        fig.canvas.draw()

        ref_ax = axes[0]
        ref_ticks = ref_ax.get_yticks()
        ref_ticklabels = [lab.get_text() for lab in ref_ax.get_yticklabels()]

        for ax in axes:
            ax.set_yticks(ref_ticks)
            ax.set_yticklabels(ref_ticklabels, fontsize=TICK_FONTSIZE)

        if len(axes_row0) > 1:
            for ax in axes_row0[1:]:
                ax.set_yticklabels([])

        if len(axes_row1) > 1:
            for ax in axes_row1[1:]:
                ax.set_yticklabels([])

    fig.supylabel("Average score", fontsize=FIG_TITLE_FONTSIZE)
    fig.suptitle(title, y=0.95, fontsize=FIG_TITLE_FONTSIZE)

    # -------- GLOBAL LEGEND (using actual bar handles) --------
    handles = []
    labels_all = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels_all.extend(l)

    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels_all):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)

    fig.legend(
        unique_handles,
        unique_labels,
        loc="center right",
        bbox_to_anchor=(0.95, 0.8),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.95,
        top=0.85,
        bottom=0.08,
        hspace=0.3,
        wspace=0.5,
    )

    plt.savefig(save_path)
    plt.show()