#!/usr/bin/env python
# coding: utf-8

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import scipy.stats

# Plotting defaults
font_manager.fontManager.ttflist.extend(
    font_manager.createFontList(
        font_manager.findSystemFonts(fontpaths="/users/amtseng/modules/fonts")
    )
)

plot_params = {
    "figure.titlesize": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.family": "Roboto",
    "font.weight": "bold"
}
plt.rcParams.update(plot_params)


def import_metrics_json(models_path, run_num):
    """
    Looks in {models_path}/{run_num}/metrics.json and returns the contents as a
    Python dictionary. Returns None if the path does not exist, or the JSON is
    not well-formed.
    """
    path = os.path.join(models_path, str(run_num), "metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("Metrics JSON at %s is not well-formed" % path)

def get_best_metric_at_best_epoch(models_path, metric_name, reduce_func, compare_func, max_epoch=None):
    """
    Given the path to a set of runs, determines the run with the best metric value,
    for the given `metric_name`. For each run, the function `reduce_func` must take
    the array of all values for that metric and return a (scalar) value FOR EACH
    SUBARRAY/VALUE in the value array to use for comparison. The best metric value
    is determined by `metric_compare_func`, which must take in two arguments, and
    return True if the _first_ one is better. If `max_epoch` is provided, will only
    report everything up to this epoch (1-indexed).
    Returns the number of the run, the (one-indexed) number of the epoch, the value
    associated with that run and epoch, and a dict of all the values used for
    comparison (mapping pair of run number and epoch number to value).
    """
    # Get the metrics, ignoring empty or nonexistent metrics.json files
    metrics = {run_num : import_metrics_json(models_path, run_num) for run_num in os.listdir(models_path)}
    metrics = {key : val for key, val in metrics.items() if val}  # Remove empties
    print(models_path, metric_name, metrics.keys()) ####

    # Get the best value
    best_run, best_epoch, best_val, all_vals = None, None, None, {}
    for run_num in metrics.keys():
        # print(metrics[run_num].keys()) ####
        try:
            # Find the best epoch within that run
            best_epoch_in_run, best_val_in_run = None, None
            for i, subarr in enumerate(metrics[run_num][metric_name]["values"]):
                if i == max_epoch:
                    break
                val = reduce_func(subarr)
                if best_val_in_run is None or compare_func(val, best_val_in_run):
                    best_epoch_in_run, best_val_in_run = i + 1, val
            all_vals[(run_num, best_epoch_in_run)] = best_val_in_run
            
            # If the best value in the best epoch of the run is best so far, update
            if best_val is None or compare_func(best_val_in_run, best_val):
                best_run, best_epoch, best_val = run_num, best_epoch_in_run, best_val_in_run
            # print(best_epoch) ####
        except Exception as e:
            print(e)
            print("Warning: Was not able to compute values for run %s" % run_num)
            continue
    return best_run, best_epoch, best_val, all_vals

def fetch_and_print_performance(models_path, metric_prefix, out_dir, run_id, max_epoch=None):
    """
    Given the path to a condition containing many runs, prints out the best validation
    losses for each run. If given, only consider up to `max_epoch` epochs total; anything
    afterward would be ignored.
    """
    val_key = f"{metric_prefix}_val_prof_corr_losses"
    
    out_path = os.path.join(out_dir, f"performance_{run_id}.txt")
    with open(out_path, "w") as out_file:
        print("Best validation loss overall:", file=out_file)
        best_run, best_epoch, best_val, all_vals = get_best_metric_at_best_epoch(
            models_path,
            val_key,
            lambda values: np.mean(values),
            lambda x, y: x < y,
            max_epoch
        )
        print("\tBest run: %s" % best_run, file=out_file)
        print("\tBest epoch in run: %d" % best_epoch if best_epoch is not None else -1, file=out_file)
        print("\tAssociated value: %f" % best_val if best_val is not None else np.nan, file=out_file)
        
        print("Best epoch in each run:", file=out_file)
        for key in sorted(all_vals.keys(), key=lambda p: int(p[0])):
            print("\tRun %s, epoch %d: %6.4f" % (key[0], key[1], all_vals[key]), file=out_file)
    return best_run, best_epoch, all_vals

def plot_test_metric_distributions(models_path, genome_prefix, nogenome_prefix, out_dir, nogenome_query_run=None, genome_query_run=None):
    # if model_type == "binary":
    #     metric_keys = [
    #         ("test_acc", "test accuracy", "less"),
    #         ("test_auroc", "test auROC", "less"),
    #         ("test_auprc", "test auPRC", "less"),
    #         ("test_corr_auprc", "estimated test auPRC", "less")
    #     ]
    # else:
    metric_keys = [
        ("summit_prof_nll", "test profile NLL", "greater"),
        ("summit_prof_jsd", "test profile JSD", "greater")
    ]
    
    # Get the metrics, ignoring empty or nonexistent metrics.json files
    model_metrics = {run_num : import_metrics_json(models_path, run_num) for run_num in os.listdir(models_path)}
    model_metrics = {key : val for key, val in model_metrics.items() if val}  # Remove empties
    
    vals_to_return = []
    
    # print(out_dir) ####
    for metric_key, metric_name, test_alternative in metric_keys:
        # print(out_dir) ####
        plt_path = os.path.join(out_dir, f"metric_{metric_key}.svg")
        # print(out_dir) ####
        nogenome_key = f"{nogenome_prefix}_{metric_key}"
        # print([[k for j in i for k in j] for metrics in model_metrics.values() for i in metrics.get(nogenome_key, {}).get("values", [[],])]) ####
        nogenome_vals = []
        nogenome_mean = np.nan
        for run, metrics in model_metrics.items():
            if nogenome_key not in metrics:
                nogenome_vals.append(np.nan)
                continue
            vals = metrics[nogenome_key]["values"]
            mean = np.mean([j["value"] for i in vals for j in i])
            nogenome_vals.append(mean)
            if run == nogenome_query_run:
                nogenome_mean = mean
            # print(np.([j["values"] for i in metrics[nogenome_key] for j in i]))
        nogenome_vals = np.array(nogenome_vals)
        # nogenome_vals = np.array([
        #     np.mean(metrics[nogenome_key]["values"]) for metrics in model_metrics.values()
        # ])
        genome_key = f"{genome_prefix}_{metric_key}"
        genome_vals = []
        genome_mean = np.nan
        for run, metrics in model_metrics.items():
            if genome_key not in metrics:
                genome_vals.append(np.nan)
                continue
            vals = metrics[genome_key]["values"]
            mean = np.mean([j["value"] for i in vals for j in i])
            genome_vals.append(mean)
            if run == genome_query_run:
                genome_mean = mean
        genome_vals = np.array(genome_vals)
        # print(nogenome_vals, genome_vals) ####
        # genome_vals = np.array([
        #     np.mean(metrics[genome_key]["values"]) for metrics in model_metrics.values()
        # ])
        vals_to_return.append((metric_name, nogenome_vals, genome_vals))
        bin_num = 20
        fig, ax = plt.subplots(figsize=(12, 6))
        all_vals = np.concatenate([nogenome_vals, genome_vals])
        bins = np.linspace(np.min(all_vals), np.max(all_vals), bin_num)
        ax.hist(nogenome_vals, bins=bins, color="coral", label="No genome", alpha=0.7)
        ax.hist(genome_vals, bins=bins, color="slateblue", label="With genome", alpha=0.7)
        title = "Histogram of %s without/with genomes" % metric_name
        title += "\n%s, %d/%d %s models" % ("BPNet", len(nogenome_vals), len(genome_vals), "profile")
        if peak_retention != "all":
            title += "\nTraining on %s peaks" % peak_retention
        ax.set_title(title)
        ax.set_xlabel(metric_name[0].upper() + metric_name[1:])
        plt.legend()
        # print(plt_path) ####
        plt.savefig(plt_path)

        # print(nogenome_vals, genome_vals, test_alternative) ####
        try:
            u, p = scipy.stats.mannwhitneyu(nogenome_vals, genome_vals, alternative=test_alternative)
        except ValueError as e:
            print(e)
            u, p = np.nan, np.nan

        txt_path = os.path.join(out_dir, f"metric_{metric_key}.txt")
        with open(txt_path, "w") as txt_file:
            print("Mean without genomes: %f" % np.mean(nogenome_vals), file=txt_file)
            print("Mean with genomes: %f" % np.mean(genome_vals), file=txt_file)
            print("Standard error without genomes: %f" % scipy.stats.sem(nogenome_vals), file=txt_file)
            print("Standard error with genomes: %f" % scipy.stats.sem(genome_vals), file=txt_file)
            print("One-sided Mann-Whitney U-test: U = %f, p = %f" % (u, p), file=txt_file)
            if nogenome_query_run is not None and genome_query_run is not None:
                # nogenome_mean = np.mean(model_metrics[nogenome_query_run][nogenome_key]["values"])
                # genome_mean = np.mean(model_metrics[genome_query_run][genome_key]["values"])
                print("No genome %s (run %s): %f" % (metric_name, nogenome_query_run, nogenome_mean), file=txt_file)
                print("With genome %s (run %s): %f" % (metric_name, genome_query_run, genome_mean), file=txt_file)
    
    return vals_to_return  # List of metric name, nogenome values, and genome values

def create_violin_pair(ax, nogenome_data, genome_data, metric_name, out_dir):
    print(nogenome_data, genome_data) ####
    all_data = np.stack([nogenome_data, genome_data], axis=0)
    # Define the quartiles
    q1, med, q3 = np.percentile(all_data, [25, 50, 70], axis=1)
    iqr = q3 - q1
    print(all_data[0], all_data[1]) ####
    plot_parts = ax.violinplot(
        [np.sort(all_data[0]), np.sort(all_data[1])], showmeans=False, showmedians=False, showextrema=False
    )
    violin_parts = plot_parts["bodies"]
    violin_parts[0].set_facecolor("coral")
    violin_parts[0].set_edgecolor("coral")
    violin_parts[0].set_alpha(0.7)
    violin_parts[1].set_facecolor("slateblue")
    violin_parts[1].set_edgecolor("slateblue")
    violin_parts[1].set_alpha(0.7)

    inds = np.arange(1, 3)
    ax.vlines(inds, q1, q3, color="black", linewidth=5, zorder=1)
    ax.scatter(inds, med, marker="o", color="white", s=30, zorder=2)
    ax.set_xticks(inds)
    ax.set_xticklabels(["No genome", "With genome"])
    ax.set_title(metric_name)

def plot_violin(test_metrics, nogenome_vals, genome_vals, peak_retention, out_dir):
    num_violin_plots = 1 + len(test_metrics)
    fig, ax = plt.subplots(1, num_violin_plots, figsize=(7 * num_violin_plots, 8))

    create_violin_pair(ax[0], nogenome_vals, genome_vals, "Validation profile NLL loss", out_dir)
    for i in range(0, len(test_metrics)):
        create_violin_pair(ax[i + 1], test_metrics[i][1], test_metrics[i][2], test_metrics[i][0][0].upper() + test_metrics[i][0][1:], out_dir)
        
    title = "Model performance without/with genomes"
    title += "\n%s, %d/%d %s models" % ("BPNet", len(nogenome_vals), len(genome_vals), "profile")
    plt.subplots_adjust(top=0.85)
    if peak_retention != "all":
        title += "\nTraining on %s peaks" % peak_retention
        plt.subplots_adjust(top=0.80)
    fig.suptitle(title)
    plt.savefig(os.path.join(out_dir, "violin.svg"))

def plot_loss_hist(nogenome_vals, genome_vals, peak_retention, out_dir):
    bin_num = 30
    fig, ax = plt.subplots(figsize=(12, 6))
    all_vals = np.concatenate([nogenome_vals, genome_vals])
    print(all_vals) ####
    bins = np.linspace(np.min(all_vals), np.max(all_vals), bin_num)
    ax.hist(nogenome_vals, bins=bins, color="coral", label="No genome", alpha=0.7)
    ax.hist(genome_vals, bins=bins, color="slateblue", label="With genome", alpha=0.7)
    title = "Histogram of validation profile NLL loss without/with genomes"
    title += "\n%s, %d/%d %s models" % ("BPNet", len(nogenome_vals), len(genome_vals), "profile")
    if peak_retention != "all":
        title += "\nTraining on %s peaks" % peak_retention
    ax.set_title(title)
    ax.set_xlabel("Validation profile NLL loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_hist.svg"))


def write_loss_stats(nogenome_vals, genome_vals, out_dir):
    # print(nogenome_vals, genome_vals) ####
    try:
        u, p = scipy.stats.mannwhitneyu(nogenome_vals, genome_vals, alternative="greater")
    except ValueError as e:
        print(e)
        u, p = np.nan, np.nan
    out_path = os.path.join(out_dir, "loss_stats.txt")
    with open(out_path, "w") as out_file:
        print("Mean without genomes: %f" % np.mean(nogenome_vals), file=out_file)
        print("Mean with genomes: %f" % np.mean(genome_vals), file=out_file)
        print("Standard error without genomes: %f" % scipy.stats.sem(nogenome_vals), file=out_file)
        print("Standard error with genomes: %f" % scipy.stats.sem(genome_vals), file=out_file)
        print("One-sided Mann-Whitney U-test: U = %f, p = %f" % (u, p), file=out_file)

def plot_stats(models_path, genome_prefix, nogenome_prefix, out_dir, peak_retention):
    nogenome_best_run, nogenome_best_epoch, nogenome_val_losses = fetch_and_print_performance(models_path, nogenome_prefix, out_dir, "nogenome")
    genome_best_run, genome_best_epoch, genome_val_losses = fetch_and_print_performance(models_path, nogenome_prefix, out_dir, "genome")
    nogenome_vals, genome_vals = np.array(list(nogenome_val_losses.values())), np.array(list(genome_val_losses.values()))
    plot_loss_hist(nogenome_vals, genome_vals, peak_retention, out_dir)
    write_loss_stats(nogenome_vals, genome_vals, out_dir)
    # print(out_dir) ####
    test_metrics = plot_test_metric_distributions(models_path, genome_prefix, nogenome_prefix, out_dir, nogenome_query_run=nogenome_best_run, genome_query_run=genome_best_run)
    # plot_violin(test_metrics, nogenome_vals, genome_vals, peak_retention, out_dir)

if __name__ == '__main__':
    models_path = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/trained_models/profile/misc/"
    # models_path = "/users/atwang/transfer/models/trained_models/profile/misc/"
    peak_retention = "all"
    cell_types = ["K562", "HepG2"]
    for i in cell_types:
        for j in cell_types:
            genome_prefix = f"{i}_from_{j}"
            nogenome_prefix = f"{i}_from_{j}_aux"
            # nogenome_prefix = f"{i}_from_{i}" ####
            out_dir = "/users/atwang/results/domain_adapt/dnase_models/"
            plot_stats(models_path, genome_prefix, nogenome_prefix, out_dir, peak_retention)



