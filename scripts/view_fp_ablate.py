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

def construct_query_table(peaks):
    table = {}
    for peak in set(peaks):
        chrom, start, end = peak
        start_hash = start // 1000
        end_hash = end // 1000
        for pos_hash in range(start_hash, end_hash + 1):
            table.setdefault((chrom, pos_hash), set()).add((start, end),)

    return table

def query_peak(query_table, peak):
    chrom, start, end = peak
    begin_hash = int(start) // 1000
    end_hash = int(end) // 1000
    candidates = set()
    for pos_hash in range(begin_hash, end_hash + 1):
        candidates |= query_table.get((chrom, pos_hash,), set())

    intersects = {}
    for c in candidates:
        start_c, end_c = c
        if start <= end_c and start_c <= end:
            # intersects.append((chrom, start_c, end_c),)
            distance = np.abs((start + end) - (start_c + end_c)) / 2
            intersects[(chrom, start_c, end_c)] = distance

    return min(intersects, key=intersects.get) if intersects else None

def load_fp_results(results, metric_names):
    fps = []
    peaks = []
    avg_metrics = {}
    for batch in results:
        # print(batch["metrics"]) ####
        fps.extend(batch["footprints"])
        peaks.extend(batch["peaks"])
        for k, v in batch["metrics"].items():
            metrics_mean_b = np.mean(v, axis=0)
            avg_metrics.setdefault(k, []).extend(metrics_mean_b)

    peak_table = construct_query_table(peaks)
    results_dict = {"Footprint Coordinates": fps, "Peak Coordinates": peaks}
    # print(avg_metrics.keys()) ####
    metric_names_vars = []
    for k, v in metric_names.items():
        metrics = avg_metrics[k]
        metrics_mean = np.concatenate(metrics)
        if v is None:
            results_dict[k] = metrics_mean
            metric_names_vars.append(k)
        else:
            for pos, var in enumerate(v):
                name_var = k + var
                results_dict[name_var] = metrics_mean[:,pos]
                metric_names_vars.append(name_var)

    # print([(k, v.shape if isinstance(v, np.ndarray) else len(v)) for k, v in results_dict.items()]) ####
    results_df = pd.DataFrame(results_dict)

    return results_df, metric_names_vars, peak_table

def load_enrichments(results_df, prefix, models_path, query_run, peak_table):
    epsilon = 1.
    if prefix == "K562_from_HepG2":
        offset = np.log(1400599316) - np.log(1728009533)
    elif prefix == "HepG2_from_K562":
        offset = np.log(1728009533) - np.log(1400599316)

    metrics_path = os.path.join(models_path, f"{prefix}_{query_run}", "metrics.pickle")
    metrics_genome = pd.read_pickle(metrics_path)
    coords_genome = [query_peak(peak_table, i) for i in metrics_genome["coords"]]
    # print(coords_genome) ####
    counts_to = metrics_genome["counts_to"].sum(axis=-1)[:,0]
    counts_from = metrics_genome["counts_from"].sum(axis=-1)[:,0]
    counts_sum = np.clip(np.log(counts_to + epsilon) + np.log(counts_from + epsilon), 10, 17)
    counts_diff = np.log(counts_to + epsilon) - np.log(counts_from + epsilon) + offset
    dct_genome = {"Peak Coordinates": coords_genome, "Enrichment Difference": counts_diff, "Enrichment Sum": counts_sum}
    df_genome = pd.DataFrame(dct_genome)
    # arr_genome = np.stack((coords_genome, counts_diff, counts_sum), axis=1)
    # df_genome = pd.DataFrame(arr_genome, columns=["Peak Coordinates", "Enrichment Difference", "Enrichment Sum"])

    # print(sorted(results_df["Peak Coordinates"])[:100])
    # print(sorted(df_genome["Peak Coordinates"])[:100])
    df_merged = results_df.merge(df_genome, on="Peak Coordinates")
    # print(df_merged) ####
    print(results_df["Peak Coordinates"].shape) ####
    print(df_genome["Peak Coordinates"].shape) ####
    print(df_merged["Peak Coordinates"].shape) ####
    return df_merged

def plot_fps(results_df, metric_name, plt_path, sample_size=None):
    if sample_size is not None:
        # print(results_df) ####
        results_df = results_df.sample(n=sample_size)

    sns.set(style="whitegrid", font="Roboto")
    plt.figure(figsize=(7,5))

    sns.scatterplot(data=results_df, x="Enrichment Difference", y=metric_name, hue="Enrichment Sum", s=6)
    plt.title(f"{metric_name} vs. Peak Specificity")
    plt.xlim((-7, 7),)
    plt.savefig(plt_path, bbox_inches='tight')
    plt.clf()

def view_fp_ablate(results_path, plt_dir, models_path, model_query_run, prefix, metric_names):
    results = pd.read_pickle(results_path)["results"]
    results_df, metric_names_vars, peak_table = load_fp_results(results, metric_names)
    results_df = load_enrichments(results_df, prefix, models_path, model_query_run, peak_table)
    for metric_name in metric_names_vars:
        plt_path = os.path.join(plt_dir, f"{metric_name}.svg")
        plot_fps(results_df, metric_name, plt_path, sample_size=5000)

if __name__ == '__main__':
    models_path = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/trained_models/transfer_v5/"
    ablate_path = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/ablation/transfer_v5/"
    out_dir_base = "/users/atwang/results/domain_adapt_results/dnase_models/"
    # models_path = "/users/atwang/transfer/models/trained_models/profile/misc/"
    run_id = "1"
    metric_names = {
        'nll': None, 
        'jsd': None, 
        # 'auprc_binned': ["_bin1", "_bin4", "_bin10"], 
        'pearson_binned': ["_bin1", "_bin4", "_bin10"], 
        'spearman_binned': ["_bin1", "_bin4", "_bin10"], 
        'mse_binned': ["_bin1", "_bin4", "_bin10"], 
        'counts_diff': [""]
    }

    cell_types = ["K562", "HepG2"]
    for i in cell_types:
        for j in cell_types:
            if i == j:
                continue
            
            # nogenome_prefix = f"{i}_from_{j}_aux"
            # nogenome_prefix = f"{i}_from_{i}" ####

            res_names = [f"{i}_from_{j}_ablate_run_{i}", f"{i}_from_{j}_ablate_run_{j}"]
            prefixes = [ f"{j}_from_{i}", f"{i}_from_{j}",]
            for res_name, prefix in zip(res_names, prefixes):
                out_dir = os.path.join(out_dir_base, prefix)
                results_path = os.path.join(ablate_path, f"{res_name}.pickle")
                plt_dir = os.path.join(out_dir_base, prefix, res_name)
                os.makedirs(plt_dir, exist_ok=True)
                view_fp_ablate(results_path, plt_dir, models_path, run_id, prefix, metric_names)

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
    