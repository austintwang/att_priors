from model.util import place_tensor, restore_model
import model.profile_models as profile_models
import feature.make_profile_dataset as make_profile_dataset
import model.profile_performance as profile_performance
import extract.data_loading as data_loading
import torch
import numpy as np
import os
import sys
import tqdm
import sacred
import json
import scipy.special

# GPU = "4"
# place_tensor = lambda x: place_tensor_util(x, index=GPU)

DEVNULL = open(os.devnull, "w")
STDOUT = sys.stdout

# OUT_DIR = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/ablation/transfer_v5/"
OUT_DIR = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/ablation/test/"

ablate_ex = sacred.Experiment("motif_ablation", ingredients=[
    profile_performance.performance_ex
])
ablate_ex.observers.append(
    sacred.observers.FileStorageObserver.create(OUT_DIR)
)

def hide_stdout():
    sys.stdout = DEVNULL
def show_stdout():
    sys.stdout = STDOUT

@ablate_ex.config
def config():
    center_size_to_use = 2114
    prof_size = 1000
    num_runs = 5
    batch_size = 1024 
    gpu_id = "4"
    num_tasks = 1
    chrom_set = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

@ablate_ex.capture
def create_mask(peak_coord, fp_coords, center_size_to_use):
    # print(peak_coord) ####
    # print(fp_coords) ####
    chrom_s, start_s, end_s = peak_coord
    if center_size_to_use is not None:
        center = int(0.5 * (start_s + end_s))
        half_size = int(0.5 * center_size_to_use)
        start_s = center - half_size
        end_s = center + center_size_to_use - half_size

    starts_mask = []
    ends_mask = []
    prev_end = -1
    for fp_coord in sorted(fp_coords):
        chrom_f, start_f, end_f = fp_coord
        assert chrom_s == chrom_f ####
        start_rel = max(start_f - start_s, 0)
        end_rel = min(end_f - start_s, center_size_to_use)
        if start_rel > prev_end + 1:
            starts_mask.append(prev_end + 1)
            ends_mask.append(start_rel)
            prev_end = end_rel
        else:
            prev_end = max(prev_end, end_rel)

    starts_mask = np.array(starts_mask)
    lens_mask = np.array(ends_mask) - starts_mask

    return starts_mask, lens_mask

def get_fp_to_peak(peak_to_fps):
    min_dists = {}
    fp_to_peak = {}
    for peak, fps in peak_to_fps.items():
        for fp in fps:
            if fp in fp_to_peak:
                if fp in min_dists:
                    min_dist = min_dists[fp]
                else:
                    peak_prev = fp_to_peak[fp]
                    min_dist = np.abs((fp[1] + fp[2]) - (peak_prev[1] + peak_prev[2])) / 2
                dist = np.abs((fp[1] + fp[2]) - (peak[1] + peak[2])) / 2
                if dist < min_dist:
                    fp_to_peak[fp] = peak
                    min_dists[fp] = dist
                else:
                    min_dists[fp] = min_dist
            else:
                fp_to_peak[fp] = peak

    return fp_to_peak

def get_fp_to_seq_slice(fps, fp_to_peak, peak_to_seq_idx, center_size_to_use):
    fp_to_seq_slice = {}
    for fp in fps:
        peak = fp_to_peak[fp]
        chrom_s, start_s, end_s = peak
        if center_size_to_use is not None:
            center = int(0.5 * (start_s + end_s))
            half_size = int(0.5 * center_size_to_use)
            start_s = center - half_size
            # end_s = center + center_size_to_use - half_size

        chrom_f, start_f, end_f = fp
        start_rel = max(start_f - start_s, 0)
        end_rel = min(end_f - start_s, center_size_to_use)

        seq_idx = peak_to_seq_idx[peak]
        fp_to_seq_slice[fp] = (seq_idx, start_rel, end_rel)

    return fp_to_seq_slice

@ablate_ex.capture
def get_ablated_inputs(fps_in, seqs, profs_ctrls, fp_to_seq_slice, fp_to_peak, seq_masks, num_runs, profs_trans=None):
    seqs_out = []
    profs_ctrls_out = []
    profs_trans_out = []
    for fp in fps_in:
        seq_slice = fp_to_seq_slice[fp]
        # print(seq_slice) ####
        seq_idx, start, end = seq_slice
        fp_len = end - start + 1

        seq = seqs[seq_idx,:,:]
        seq_runs = np.broadcast_to(seq, (num_runs+1, seq.shape[0], seq.shape[1])).copy()

        mask_starts, mask_lens = seq_masks[fp_to_peak[fp]]
        # print(mask_starts) ####
        # print(mask_lens) ####
        mask_lens_allowed = np.clip(mask_lens - fp_len + 1, 0, None)
        mask_probs = mask_lens_allowed / sum(mask_lens_allowed)
        
        choices_1 = np.random.choice(len(mask_lens_allowed), size=num_runs, p=mask_probs)
        for run_num, interval_choice in enumerate(choices_1):
            slc_start = np.random.randint(mask_lens_allowed[interval_choice]) + mask_starts[interval_choice]
            slc_end = slc_start + fp_len
            # print(slc_start, slc_end) ####
            seq_runs[run_num+1, start:end+1, :] = seq[slc_start:slc_end, :]
        
        seqs_out.append(seq_runs)

        prof_ctrls = profs_ctrls[seq_idx,:,:,:]
        profs_ctrls_out.append(prof_ctrls)

        if profs_trans is not None:
            prof_trans = profs_trans[seq_idx,:,:,:]
            profs_trans_out.append(prof_trans)

    if prof_trans is not None:
        return np.stack(seqs_out), np.stack(profs_ctrls_out), np.stack(profs_trans_out)
    return np.stack(seqs_out), np.stack(prof_ctrls_out)
        
@ablate_ex.capture
def run_model(model_path, seqs, profs_ctrls, fps, gpu_id, model_args_extras=None, profs_trans=None):
    torch.set_grad_enabled(True)
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    if profs_trans is not None:
        model_class = profile_models.ProfilePredictorTransfer
    else:
        model_class = profile_models.ProfilePredictorWithMatchedControls
    model = restore_model(model_class, model_path, model_args_extras=model_args_extras)
    model.eval()
    model = model.to(device)

    # print(seqs[:5]) ####
    # print(seqs.shape) ####
    # print(profs_ctrls.shape) ####
    # print(profs_trans.shape) ####
    num_runs = seqs.shape[1]
    profs_preds_shape = (profs_ctrls.shape[0], seqs.shape[1], profs_ctrls.shape[1], profs_ctrls.shape[2], profs_ctrls.shape[3])
    profs_preds_logits = np.empty(profs_preds_shape)
    counts_preds_shape = (profs_ctrls.shape[0], seqs.shape[1], profs_ctrls.shape[1], profs_ctrls.shape[3])
    counts_preds = np.empty(counts_preds_shape)

    profs_ctrls_b = place_tensor(torch.tensor(profs_ctrls), index=gpu_id).float() 
    if profs_trans is not None:
        profs_trans_b = place_tensor(torch.tensor(profs_trans), index=gpu_id).float()

    for run in range(seqs.shape[1]):
        seqs_b = place_tensor(torch.tensor(seqs[:,run]), index=gpu_id).float() 
        if profs_trans is not None:
            profs_preds_logits_b, counts_preds_b = model(seqs_b, profs_ctrls_b, profs_trans_b)
        else:
            profs_preds_logits_b, counts_preds_b = model(seqs_b, profs_ctrls_b)

        profs_preds_logits_b = profs_preds_logits_b.detach().cpu().numpy()
        counts_preds_b = counts_preds_b.detach().cpu().numpy()
        profs_preds_logits[:,run] = profs_preds_logits_b
        counts_preds[:,run] = counts_preds_b

    return profs_preds_logits, counts_preds

@ablate_ex.capture
def get_metrics(profs_preds_logits, counts_preds, num_runs):
    profs_preds_logs = profs_preds_logits - scipy.special.logsumexp(profs_preds_logits, axis=3, keepdims=True)
    profs_preds_logs_o = profs_preds_logs[:,0]
    counts_preds_o = counts_preds[:,0]
    profs_preds_o = np.exp(profs_preds_logs_o) * np.broadcast_to(counts_preds_o[:,:,np.newaxis,:], profs_preds_logs_o.shape)
    # print(profs_preds_logs_o.shape) ####
    # print(counts_preds_o.shape) ####
    # print(profs_preds_o.shape) ####
    metrics = {}
    for i in range(1, num_runs + 1):
        profs_preds_logs_a = profs_preds_logs[:,i]
        counts_preds_a = counts_preds[:,i]
        # print(profs_preds_logs_o.shape) ####
        # print(profs_preds_logs_a.shape) ####
        metrics_run = profile_performance.compute_performance_metrics(profs_preds_o, profs_preds_logs_a, counts_preds_o, counts_preds_a, print_updates=False)
        for k, v in metrics_run.items():
            metrics.setdefault(k, []).append(v)

    return metrics

@ablate_ex.command
def run(files_spec, model_path, reference_fasta, model_class, out_path, num_runs, chrom_set, num_tasks, prof_size, center_size_to_use, batch_size, model_args_extras=None):
    print("Loading footprints...")
    peaks, peak_to_fp_prof, peak_to_fp_reg = data_loading.get_profile_footprint_coords(files_spec, prof_size=prof_size, region_size=center_size_to_use, chrom_set=chrom_set)
    masks = {k: create_mask(k, v) for k, v in peak_to_fp_reg.items()}
    fp_to_peak = get_fp_to_peak(peak_to_fp_prof)
    fps = list(fp_to_peak.keys())
    # peak_to_seq_idx = {val: ind for ind, val in enumerate(peaks)}
    # fp_to_seq_slice = get_fp_to_seq_slice(fp_to_peak, peak_to_seq_idx)

    if model_class == "prof_trans":
        input_func = data_loading.get_profile_trans_input_func(
            files_spec, center_size_to_use, prof_size, reference_fasta,
        )
    else:
        input_func = data_loading.get_profile_input_func(
            files_spec, center_size_to_use, prof_size, reference_fasta,
        )

    print("Computing metrics...")
    results = []
    fp_idx = {}
    fps = fps[:1000] ####
    for batch, i in enumerate(tqdm.tqdm(range(0, len(fps), batch_size))):
        j = min(i + batch_size, len(fps))
        fps_slice = fps[i:j]
        peaks_slice = list(set(fp_to_peak[i] for i in fps_slice))
        peak_to_seq_idx = {val: ind for ind, val in enumerate(peaks_slice)}
        fp_to_seq_slice = get_fp_to_seq_slice(fps_slice, fp_to_peak, peak_to_seq_idx, center_size_to_use)

        if model_class == "prof_trans":
            seqs, profiles, profs_trans = input_func(peaks_slice)
            profs_trans = profs_trans[:, :num_tasks]
            profs_ctrls = profiles[:, num_tasks:]
            # print(profs_trans.shape) ####
            # print(profs_ctrls.shape) ####
            seqs_in, profs_ctrls_in, profs_trans_in = get_ablated_inputs(fps_slice, seqs, profs_ctrls, fp_to_seq_slice, fp_to_peak, masks, num_runs, profs_trans=profs_trans)
            profs_preds_logits, counts_preds = run_model(model_path, seqs_in, profs_ctrls_in, fps, model_args_extras=model_args_extras, profs_trans=profs_trans_in)
        else:
            seqs, profiles = input_func(peaks_slice)
            profs_ctrls = profiles[:, num_tasks:]
            seqs_in, profs_ctrls_in = get_ablated_inputs(fps_slice, seqs, profs_ctrls, fp_to_seq_slice, fp_to_peak, masks, num_runs)
            profs_preds_logits, counts_preds = run_model(model_path, seqs_in, profs_ctrls_in, fps, model_args_extras=model_args_extras)

        metrics = get_metrics(profs_preds_logits, counts_preds, num_runs)
        result_b = {
            "footprints": fps_slice,
            "peaks": [fp_to_peak[i] for i in fps_slice],
            "metrics": metrics,
        }
        results.append(result_b)

        for ind, val in enumerate(fps_slice):
            fp_idx[val] = (batch, ind)

    export = {"results": results, "index": fp_idx}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as out_file:
        pickle.dump(export, out_file)

@ablate_ex.automain
def main():
    # main()

    model_class = "prof_trans"
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    

    models_dir = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/trained_models/transfer_v5/"
    hdf5_dir = "/mnt/lab_data2/atwang/att_priors/data/processed/ENCODE_DNase/profile/labels"
    peak_bed_dir = "/mnt/lab_data2/amtseng/share/austin/dnase"
    fp_bed_dir = "/users/amtseng/att_priors/data/raw/DNase_footprints/footprints_per_sample/"
    out_dir = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/ablation/transfer_v5/"

    cell_types = {
        "K562": ["ENCSR000EOT"],
        "HepG2": ["ENCSR149XIL"]
    }
    fp_beds = {
        "K562": ["K562-DS15363.bed.gz", "K562-DS16924.bed.gz"],
        "HepG2": ["h.HepG2-DS24838.bed.gz", "h.HepG2-DS7764.bed.gz"]
    }

    run_id = "1"

    for i, i_ex in cell_types.items():
        for j, j_ex in cell_types.items():
            if i == j:
                continue

            metrics_path = os.path.join(models_dir, run_id, "metrics.json")
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            best_epoch = metrics[f"{i}_from_{j}_best_epoch"]["values"][0]
            model_path = os.path.join(models_dir, f"{i}_from_{j}_{run_id}", f"model_ckpt_epoch_{best_epoch}.pt")

            files_spec = {
                "profile_hdf5": os.path.join(hdf5_dir, f"{i}/{i}_profiles.h5"),
                "profile_trans_hdf5": os.path.join(hdf5_dir, f"{j}/{j}_profiles.h5"),
                "peak_beds": [os.path.join(peak_bed_dir, f"DNase_{ex}_{i}_idr-optimal-peaks.bed.gz") for ex in i_ex],
                "footprint_beds": [os.path.join(fp_bed_dir, fex) for fex in fp_beds[i]]
            }
            out_path = os.path.join(out_dir, f"{i}_from_{j}_ablate.pickle")

            extras = {
                "prof_trans_conv_kernel_size": 15,
                "prof_trans_conv_channels": [5],
            }

            run(files_spec, model_path, reference_fasta, model_class, out_path, model_args_extras=extras)
