#!/usr/bin/env python
# coding: utf-8

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import scipy.stats
import pandas as pd 
import seaborn as sns

# Plotting defaults
font_manager.fontManager.ttflist.extend(
    font_manager.createFontList(
        font_manager.findSystemFonts(fontpaths="/users/amtseng/modules/fonts")
    )
)

# plot_params = {
#     "figure.titlesize": 22,
#     "axes.titlesize": 22,
#     "axes.labelsize": 20,
#     "legend.fontsize": 18,
#     "xtick.labelsize": 16,
#     "ytick.labelsize": 16,
#     "font.family": "Roboto",
#     "font.weight": "bold"
# }
# plt.rcParams.update(plot_params)


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
    # print(models_path, metric_name, metrics.keys()) ####

    # Get the best value
    best_run, best_epoch, best_val, all_vals = None, None, None, {}
    for run_num in metrics.keys():
        # print(metrics[run_num].keys()) ####
        # try:
        #     a = metrics[run_num][metric_name] ####
        # except KeyError as e:
        #     print(metrics[run_num].keys())
        #     print(e)
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

def plot_metric_individual(models_path, prefix, query_run, loader_name, metric_key, metric_name, metrics_genome_dict, metrics_nogenome_dict, plt_dir):
    metrics_path = os.path.join(models_path, f"{prefix}_{query_run}", "metrics.pickle")
    metrics_genome = metrics_genome_dict.setdefault(query_run, pd.read_pickle(metrics_path)) 
    metrics_path_nogenome = os.path.join(models_path, f"{prefix}_{query_run}", "metrics_aux.pickle")
    metrics_nogenome = metrics_nogenome_dict.setdefault(query_run, pd.read_pickle(metrics_path_nogenome))

    name_genome = f"{metric_name} With Genome"
    name_nogenome = f"{metric_name} Without Genome"
    name_differential = f"{metric_name} Differential"

    # print(metrics_genome.keys()) ####
    dict_key, arr_idx = metric_key
    data_genome = metrics_genome[dict_key]
    if arr_idx is not None:
        data_genome = data_genome[:,:,arr_idx]
    data_genome = data_genome[:,0]
    coords_genome = metrics_genome["coords"]
    counts_to = metrics_genome["counts_to"]
    counts_from = metrics_genome["counts_from"]
    counts_diff = counts_to - counts_from
    print(coords_genome) ####
    print(counts_diff) ####
    print(data_genome) ####
    arr_genome = np.stack((coords_genome, counts_diff, data_genome), axis=1)
    df_genome = pd.DataFrame(arr_genome, columns=["Coordinates", "Enrichment Differential", name_genome])

    data_nogenome = metrics_nogenome[dict_key]
    if arr_idx is not None:
        data_nogenome = data_nogenome[:,:,arr_idx]
    data_nogenome = data_nogenome[:,0]
    coords_nogenome = metrics_nogenome["coords"]
    arr_nogenome = np.stack((coords_nogenome, data_nogenome))
    df_nogenome = pd.DataFrame(arr_nogenome, columns=["Coordinates", metric_name])

    df_merged = df_genome.merge(df_nogenome, on="Coordinates", suffixes=(None, " Without Genome"))
    df_merged[name_differential] = df_merged[name_genome] - df_merged[name_nogenome]

    sns.set(style="whitegrid", font="Roboto")

    sns.scatterplot(data=df_merged, x="Enrichment Differential", y=name_genome)
    plt.title(f"{name_genome} vs. Peak Specificity")
    plt_path = os.path.join(plt_dir, f"individual_genome.svg")
    plt.savefig(plt_path, bbox_inches='tight')
    plt.clf()

    sns.scatterplot(data=df_merged, x="Enrichment Differential", y=name_nogenome)
    plt.title(f"{name_nogenome} vs. Peak Specificity")
    plt_path = os.path.join(plt_dir, f"individual_nogenome.svg")
    plt.savefig(plt_path, bbox_inches='tight')
    plt.clf()

    sns.scatterplot(data=df_merged, x="Enrichment Differential", y=name_diff)
    plt.title(f"{name_diff} vs. Peak Specificity")
    plt_path = os.path.join(plt_dir, f"individual_diff.svg")
    plt.savefig(plt_path, bbox_inches='tight')
    plt.clf()

def plot_test_metric_distributions(models_path, genome_prefix, nogenome_prefix, out_dir, nogenome_query_run=None, genome_query_run=None):
    # if model_type == "binary":
    #     metric_keys = [
    #         ("test_acc", "test accuracy", "less"),
    #         ("test_auroc", "test auROC", "less"),
    #         ("test_auprc", "test auPRC", "less"),
    #         ("test_corr_auprc", "estimated test auPRC", "less")
    #     ]
    # else:
    # print(nogenome_query_run, genome_query_run) ####
    loaders = [
        "summit_union",
        "summit_to_sig",
        "summit_from_sig",
        "summit_to_sig_from_sig",
        # "summit_to_insig_from_sig",
        # "summit_to_sig_from_insig",
        "test_summit_to_insig_from_sig",
        "test_summit_to_sig_from_insig",
    ]

    metric_keys = []
    for i in loaders:
        metric_keys_spec = [
            (i, "prof_nll", "Test Profile NLL", "greater", (0, 7000), True, ("nll", None)),
            (i, "prof_jsd", "Test Profile JSD", "greater", (0, 0.3), True, ("jsd", None)),
            (i, "prof_spearman_bin1", "Test Profile Spearman r, Bin 1", "greater", (0, 1), True, ("spearman_binned", 0)),
            (i, "prof_pearson_bin1", "Test Profile Pearson r, Bin 1", "greater", (0, 1), True, ("pearson_binned", 0)),
            (i, "prof_mse_bin1", "Test Profile MSE, Bin 1", "greater", (0, 5.5e-5), True, ("mse_binned", 0)),
            (i, "prof_spearman_bin4", "Test Profile Spearman r, Bin 4", "greater", (0, 1), True, ("spearman_binned", 1)),
            (i, "prof_pearson_bin4", "Test Profile Pearson r, Bin 4", "greater", (0, 1), True, ("pearson_binned", 1)),
            (i, "prof_mse_bin4", "Test Profile MSE, Bin 4", "greater", (0, 5.5e-5), True, ("mse_binned", 1)),
            (i, "prof_spearman_bin10", "Test Profile Spearman r, Bin 10", "greater", (0, 1), True, ("spearman_binned", 2)),
            (i, "prof_pearson_bin10", "Test Profile Pearson r, Bin 10", "greater", (0, 1), True, ("pearson_binned", 2)),
            (i, "prof_mse_bin10", "Test Profile MSE, Bin 10", "greater", (0, 5.5e-5), True, ("mse_binned", 2)),
            (i, "count_spearman", "Test Count Spearman r, Bin 10", "greater", (0, 1), False, None),
            (i, "count_pearson", "Test Count Pearson r, Bin 10", "greater", (0, 1), False, None),
            (i, "count_mse", "Test Count MSE, Bin 10", "greater", (0, None), False, None)
        ]
        metric_keys.extend(metric_keys_spec)
    
    # Get the metrics, ignoring empty or nonexistent metrics.json files
    model_metrics = {run_num : import_metrics_json(models_path, run_num) for run_num in os.listdir(models_path)}
    model_metrics = {key : val for key, val in model_metrics.items() if val}  # Remove empties
    
    vals_to_return = []

    metrics_genome_dict = {}
    metrics_nogenome_dict = {}
    
    # print(out_dir) ####
    for loader_name, metric_key, metric_name, test_alternative, bounds, plot_individual, ind_key in metric_keys:
        # print(out_dir) ####
        plt_dir = os.path.join(out_dir, metric_key)
        os.makedirs(plt_dir, exist_ok=True)
        plt_path = os.path.join(plt_dir, f"{loader_name}.svg")
        # print(out_dir) ####
        nogenome_key = f"{nogenome_prefix}_{loader_name}_{metric_key}"
        # print([[k for j in i for k in j] for metrics in model_metrics.values() for i in metrics.get(nogenome_key, {}).get("values", [[],])]) ####
        nogenome_vals = []
        nogenome_mean = np.nan
        for run, metrics in model_metrics.items():
            # print(metrics.keys()) ####
            if nogenome_key not in metrics:
                # print(run) ####
                # print(nogenome_key) ####
                # print(metrics.keys()) ####
                nogenome_vals.append(np.nan)
                continue
            vals = metrics[nogenome_key]["values"]
            mean = np.mean([j["value"] for i in vals for j in i])
            nogenome_vals.append(mean)
            if run == nogenome_query_run:
                # print(run, mean, run == nogenome_query_run) ####
                nogenome_mean = mean
            # print(np.([j["values"] for i in metrics[nogenome_key] for j in i]))
        # print(nogenome_mean) ####
        nogenome_vals = np.array(nogenome_vals)
        # nogenome_vals = np.array([
        #     np.mean(metrics[nogenome_key]["values"]) for metrics in model_metrics.values()
        # ])
        genome_key = f"{genome_prefix}_{loader_name}_{metric_key}"
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

        sns.set(style="whitegrid", font="Roboto")
        hist_data = np.stack([nogenome_vals, genome_vals], axis=-1)
        hist_df = pd.DataFrame(hist_data, columns=["No Genome", "With Genome"])
        bin_num = 20
        sns.histplot(data=hist_df, bins=bin_num)

        # vals_to_return.append((metric_name, nogenome_vals, genome_vals))
        # bin_num = 20
        # fig, ax = plt.subplots(figsize=(12, 6))
        # all_vals = np.concatenate([nogenome_vals, genome_vals])
        # bins = np.linspace(np.min(all_vals), np.max(all_vals), bin_num)
        # ax.hist(nogenome_vals, bins=bins, color="coral", label="No genome", alpha=0.7)
        # ax.hist(genome_vals, bins=bins, color="slateblue", label="With genome", alpha=0.7)
        title = "Histogram of %s without/with genome" % metric_name
        title += f"\nDNAse {loader_name}, {len(nogenome_vals)}/{len(genome_vals)} profile models" 
        if peak_retention != "all":
            title += "\nTraining on %s peaks" % peak_retention
        # ax.set_title(title)
        # ax.set_xlabel(metric_name[0].upper() + metric_name[1:])
        # plt.legend()
        # print(plt_path) ####
        left, right = bounds
        if left is not None:
            plt.xlim(left=left)
        if right is not None:
            plt.xlim(right=right)
        plt.title(title)
        plt.savefig(plt_path, bbox_inches='tight')
        plt.clf()

        if loader_name == "summit_union" and plot_individual:
            plot_metric_individual(models_path, genome_prefix, genome_query_run, loader_name, ind_key, metric_name, metrics_genome_dict, metrics_nogenome_dict, plt_dir)

        # print(nogenome_vals, genome_vals, test_alternative) ####
        try:
            u, p = scipy.stats.mannwhitneyu(nogenome_vals, genome_vals, alternative=test_alternative)
        except ValueError as e:
            print(e)
            u, p = np.nan, np.nan

        txt_path = os.path.join(plt_dir, f"{loader_name}.txt")
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
    # print(nogenome_data, genome_data) ####
    all_data = np.stack([nogenome_data, genome_data], axis=0)
    # Define the quartiles
    q1, med, q3 = np.percentile(all_data, [25, 50, 70], axis=1)
    iqr = q3 - q1
    # print(all_data[0], all_data[1]) ####
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
    # print(all_vals) ####
    if len(all_vals) == 0:
        return
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
    models_path = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/trained_models/transfer_v4/"
    out_dir_base = "/users/atwang/results/domain_adapt_results/dnase_models/"
    # models_path = "/users/atwang/transfer/models/trained_models/profile/misc/"
    peak_retention = "all"
    cell_types = ["K562", "HepG2"]
    for i in cell_types:
        for j in cell_types:
            if i == j:
                continue
            genome_prefix = f"{i}_from_{j}"
            nogenome_prefix = f"{i}_from_{j}_aux"
            # nogenome_prefix = f"{i}_from_{i}" ####
            out_dir = os.path.join(out_dir_base, genome_prefix)
            os.makedirs(out_dir, exist_ok=True)
            plot_stats(models_path, genome_prefix, nogenome_prefix, out_dir, peak_retention)

    # models_path = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/trained_models/baseline/"
    # out_dir_base = "/users/atwang/results/domain_adapt_results/dnase_models_baseline/"
    # peak_retention = "all"
    # cell_types = ["K562", "HepG2"]
    # for i in cell_types:
    #     genome_prefix = f"{i}_dnase_base"
    #     nogenome_prefix = f"{i}_dnase_base"
    #     # nogenome_prefix = f"{i}_from_{i}" ####
    #     out_dir = os.path.join(out_dir_base, genome_prefix)
    #     os.makedirs(out_dir, exist_ok=True)
    #     plot_stats(models_path, genome_prefix, nogenome_prefix, out_dir, peak_retention)
    
    



