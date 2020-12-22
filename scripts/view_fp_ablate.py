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

def load_fp_results(results):
    fps = []
    peaks = []
    avg_metrics = {}
    for batch in results:
        data = zip(*(batch[i] for i in ("footprints", "peaks", "metrics")))
        for fps_b, peaks_b, metrics_b in data:
            print(fps_b, peaks_b, metrics_b)
            fps.extend(fps_b)
            peaks.extend(peaks_b)
            for k, v in metrics_b.items():
                metrics_mean_b = np.mean(v, axis=0)
                avg_metrics.setdefault(k, []).append(metrics_mean_b)

    results_dict = {"Footprint Coordinates": fps, "Peak Coordinates": peaks}
    print(avg_metrics.keys()) ####
    for k, v in avg_metrics.items():
        metrics_mean = np.concatenate(v)
        results_dict[k] = metrics_mean

    results_df = pd.DataFrame(results_dict)

    return results_df

def load_enrichments(results_df, genome_prefix, metrics_path, models_path, model_query_run):
    epsilon = 1.
    if prefix == "K562_from_HepG2":
        offset = np.log(1400599316) - np.log(1728009533)
    elif prefix == "HepG2_from_K562":
        offset = np.log(1728009533) - np.log(1400599316)

    metrics_path = os.path.join(models_path, f"{prefix}_{query_run}", "metrics.pickle")
    metrics_genome = metrics_genome_dict.setdefault(query_run, pd.read_pickle(metrics_path)) 
    coords_genome = metrics_genome["coords"][:,1]
    counts_to = metrics_genome["counts_to"].sum(axis=-1)[:,0]
    counts_from = metrics_genome["counts_from"].sum(axis=-1)[:,0]
    counts_sum = np.clip(np.log(counts_to + epsilon) + np.log(counts_from + epsilon), 10, 17)
    counts_diff = np.log(counts_to + epsilon) - np.log(counts_from + epsilon) + offset
    arr_genome = np.stack((coords_genome, counts_diff, counts_sum), axis=1)
    df_genome = pd.DataFrame(arr_genome, columns=["Peak Coordinates", "Enrichment Difference", "Enrichment Sum"])

    df_merged = results_df.merge(df_nogenome, on="Peak Coordinates").astype(float)
    return df_merged

def plot_fps(results_df, metric_name, plt_path, sample_size=None):
    if sample_size is not None:
        results = results.sample(n=sample_size)

    sns.set(style="whitegrid", font="Roboto")
    plt.figure(figsize=(7,5))

    sns.scatterplot(data=results_df, x="Enrichment Difference", y=metric_name, hue="Enrichment Sum", s=6)
    plt.title(f"{metric_name} vs. Peak Specificity")
    plt_path = os.path.join(plt_dir, f"individual_genome.svg")
    plt.xlim((-7, 7),)
    plt.savefig(plt_path, bbox_inches='tight')
    plt.clf()

def view_fp_ablate(results_path, plt_dir, models_path, model_query_run, prefix, metric_names):
    results = pd.read_pickle(results_path)["results"]
    results_df = load_fp_results(results)
    results_df = load_enrichments(results_df, prefix, models_path, model_query_run)
    for metric_name in metric_names:
        plt_path = os.path.join(plt_dir, f"{metric_name}.svg")
        plot_fps(results_df, metric_name, plt_path, sample_size=1000)

if __name__ == '__main__':
    models_path = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/trained_models/transfer_v5/"
    ablate_path = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/ablation/transfer_v5/"
    out_dir_base = "/users/atwang/results/domain_adapt_results/dnase_models/"
    # models_path = "/users/atwang/transfer/models/trained_models/profile/misc/"
    run_id = "1"
    metric_names = {

    }

    cell_types = ["K562", "HepG2"]
    for i in cell_types:
        for j in cell_types:
            if i == j:
                continue
            prefix = f"{i}_from_{j}"
            # nogenome_prefix = f"{i}_from_{j}_aux"
            # nogenome_prefix = f"{i}_from_{i}" ####
            out_dir = os.path.join(out_dir_base, prefix)

            res_names = [f"{i}_from_{j}_ablate_run_{i}", f"{i}_from_{j}_ablate_run_{j}"]
            for res_name in res_names:
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
    