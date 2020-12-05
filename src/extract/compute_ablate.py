from model.util import place_tensor
import model.profile_models as profile_models
import model.binary_models as binary_models
from extract.dinuc_shuffle import dinuc_shuffle
import model.profile_performance as profile_performance
import torch
import numpy as np
import os
import sys
import tqdm

# GPU = "4"
# place_tensor = lambda x: place_tensor_util(x, index=GPU)

DEVNULL = open(os.devnull, "w")
STDOUT = sys.stdout

def hide_stdout():
    sys.stdout = DEVNULL
def show_stdout():
    sys.stdout = STDOUT

def create_mask(fp_coords, peak_coord, center_size_to_use=None):
    chrom_s, start_s, end_s = peak_coord
    if center_size_to_use is not None:
        center = int(0.5 * (start_s + end_s))
        half_size = int(0.5 * center_size_to_use)
        start_s = center - half_size
        end_s = center + center_size_to_use - half_size

    starts_mask = []
    ends_mask = []
    prev_end = 0
    for fp_coord in sorted(fp_coords):
        chrom_f, start_f, end_f = fp_coord
        assert chrom_s == chrom_f ####
        start_rel = max(start_f - start_s, 0)
        end_rel = min(end_f - end_s, center_size_to_use)
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

def get_fp_to_seq_slice(fp_to_peak, peak_to_seq_idx, center_size_to_use=None):
    fp_to_seq_slice = {}
    for fp, peak in fp_to_peak.items():
        chrom_s, start_s, end_s = peak
        if center_size_to_use is not None:
            center = int(0.5 * (start_s + end_s))
            half_size = int(0.5 * center_size_to_use)
            start_s = center - half_size
            end_s = center + center_size_to_use - half_size

        chrom_f, start_f, end_f = fp
        start_rel = max(start_f - start_s, 0)
        end_rel = min(end_f - end_s, center_size_to_use)

        seq_idx = peak_to_seq_idx[peak]
        fp_to_seq_slice[fp] = (seq_idx, start_rel, end_rel)

    return fp_to_seq_slice

def get_ablated_inputs(seqs, profiles, profs_ctrls, fp_to_seq_slice, fp_to_peak, seq_masks, num_runs, profs_trans=None):
    seqs_out = []
    profs_ctrls_out = []
    profs_trans_out = []
    profs_out = []
    fps = []
    for fp, seq_slice in fp_to_seq_slice.items():
        fps.append(fp)

        seq_idx, start, end = seq_slice
        fp_len = end - start + 1

        seq = seqs[seq_idx,:,:]
        seq_runs = np.broadcast_to(seq, (num_runs+1, seq.shape[0], seq.shape[1]))

        mask_starts, mask_lens = seq_masks[fp_to_peak[fp]]
        mask_lens_allowed = np.clip(mask_lens - fp_len + 1, 0, None)
        mask_probs = mask_lens_allowed / sum(mask_lens_allowed)
        
        choices_1 = np.random.choice(len(mask_lens_allowed), size=num_runs, p=mask_probs)
        for run_num, interval_choice in enumerate(choices_1):
            slc_start = np.random.randint(mask_lens_allowed[interval_choice]) + mask_starts[interval_choice]
            slc_end = slc_start + fp_len
            seq_runs[run_num+1, start:end+1, :] = seq[slc_start:slc_end, :]
        
        seqs_out.append(seq_runs)
        
        prof = profiles[seq_idx,:,:,:]
        profs_out.append(prof)

        prof_ctrls = profs_ctrls[seq_idx,:,:,:]
        profs_ctrls_out.append(prof_ctrls)

        if profs_trans is not None:
            prof_trans = profs_trans[seq_idx,:,:,:]
            profs_trans_out.append(profs_trans)

    if prof_trans is not None:
        return fps, np.stack(seqs_out), np.stack(prof_ctrls), np.stack(prof), np.stack(prof_trans)
    return fps, np.stack(seqs_out), np.stack(prof_ctrls), np.stack(prof)
        
def run_model(model_path, seqs, profs, profs_ctrls, fps, batch_size, gpu_id, model_args_extras=None, profs_trans=None):
    torch.set_grad_enabled(True)
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    if profs_trans is not None:
        model_class = profile_models.ProfilePredictorTransfer
    else:
        model_class = profile_models.ProfilePredictorWithMatchedControls
    model = model_util.restore_model(model_class, model_path, model_args_extras=model_args_extras)
    model.eval()
    model = model.to(device)

    num_runs = seqs.shape[1]
    profs_preds_logits = np.empty(profs_preds_shape)
    counts_preds = np.empty(counts_preds_shape)

    for i in tqdm.tqdm(range(0, len(fps), batch_size)):
        j = min(i + batch_size, len(fps))
        profs_b = util.place_tensor(torch.tensor(profs[i:j]), index=gpu_id).float() 
        profs_ctrls_b = util.place_tensor(torch.tensor(profs_ctrls[i:j]), index=gpu_id).float() 
        if profs_trans is not None:
            profs_trans_b = util.place_tensor(torch.tensor(profs_trans[i:j]), index=gpu_id).float()

        for run in range(num_runs):
            seqs_b = util.place_tensor(torch.tensor(seqs[i:j,run]), index=gpu_id).float() 
            if profs_trans is not None:
                profs_preds_logits_b, counts_preds_b = model(seqs_b, profs_ctrls_b, profs_trans_b)
            else:
                profs_preds_logits_b, counts_preds_b = model(seqs_b, profs_ctrls_b)

            profs_preds_logits_b = profs_preds_logits_b.detach().cpu().numpy()
            counts_preds_b = counts_preds_b.detach().cpu().numpy()
            profs_preds_logits[i:j,run] = profs_preds_logits_b
            counts_preds[i:j,run] = counts_preds_b

    return profs_preds_logits, counts_preds

def get_metrics(profs_preds_logits, counts_preds):
    


def ablate(seqs, seq_coords, fp_coords, fp_map, num_runs, center_size_to_use=None):
    seqs_ablated = []
    for seq, seq_coord, fp_coord in zip(seqs, seq_coords, fp_coords):
        chrom_s, start_s, end_s = seq_coord
        chrom_f, start_f, end_f = fp_coord
        assert chrom_s == chrom_f

        if center_size_to_use is not None:
            center = int(0.5 * (start_s + end_s))
            half_size = int(0.5 * center_size_to_use)
            start_s = center - half_size
            end_s = center + center_size_to_use - half_size

        start_rel = max(start_f - start_s, 0)
        end_rel = min(end_f - end_s, center_size_to_use)
        fp_len = end_rel - start_rel

        fps_all = fp_map[tuple(seq_coord)]

