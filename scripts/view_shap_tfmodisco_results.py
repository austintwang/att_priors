import sys
import os
sys.path.append(os.path.abspath("../src/"))
import plot.viz_sequence as viz_sequence
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def find_motifs(input_seqs, query_seq, center_slice):
    base_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
    rc_base_dict = {"A": 3, "C": 2, "G": 1, "T": 0}
    found = []
    seq = np.array([base_dict[base] for base in query_seq])
    rc_seq = np.array([rc_base_dict[base] for base in query_seq])
    for i in tqdm.notebook.trange(len(input_seqs)):
        input_seq = np.where(input_seqs[i][center_slice] == 1)[1]
        for j in range(0, len(input_seq) - len(seq)):
            if np.all(seq == input_seq[j : j + len(seq)]) or np.all(rc_seq == input_seq[j : j + len(seq)]):
                found.append(i)
                break
    return found

def pfm_info_content(pfm, background_freqs, pseudocount=0.001):
    """
    Given an L x 4 PFM, computes information content for each base and
    returns it as an L-array.
    """
    num_bases = pfm.shape[1]
    # Normalize track to probabilities along base axis
    pfm_norm = (pfm + pseudocount) / (np.sum(pfm, axis=1, keepdims=True) + (num_bases * pseudocount))
    ic = pfm_norm * np.log2(pfm_norm / np.expand_dims(background_freqs, axis=0))
    return np.sum(ic, axis=1)

def pfm_to_pwm(pfm, background_freqs, pseudocount=0.001):
    """
    Converts and L x 4 PFM into an L x 4 PWM.
    """
    num_bases = pfm.shape[1]
    # Incorporate pseudocount by adding it to every element and renormalizing
    pfm_norm = (pfm + pseudocount) / (np.sum(pfm, axis=1, keepdims=True) + (num_bases * pseudocount))
    return np.log2(pfm_norm / np.expand_dims(background_freqs, axis=0))

def plot_shap(shap_scores_path, out_dir):
    with h5py.File(shap_scores_path, "r") as f:
        hyp_scores = f["hyp_scores"][:]
        input_seqs = f["one_hot_seqs"][:]

    plt_dir = os.path.join(out_dir, "shap_weights")
    os.makedirs(plt_dir, exist_ok=True)
    for index in np.random.choice(hyp_scores.shape[0], size=5, replace=False):
        viz_sequence.plot_weights((hyp_scores[index] * input_seqs[index])[570:770], out_path=os.path.join(plt_dir, f"hyp_{index}.svg"), subticks_frequency=100)
        plt.clf()

def import_tfmodisco_motifs(
    tfm_results_hdf5, background_freqs, out_dir, min_seqlets=0, min_ic=0.6, ic_window=6, trim_flank_ic_frac=0.2,
    max_length=20, plot_all_motifs=False, plot_passed_motifs=True
):
    """
    Imports the TF-MoDISco motifs, and a final set of motifs, trimmed by info content.
    The motifs returned must have at least `min_seqlets` supporting them, and there must
    be a window of size `ic_window` with at IC at least `min_ic`. Finally, the resulting
    motifs are trimmed by cutting off flanks whose base-level IC is below
    `trim_flank_ic_frac` of the highest IC of the motif. If the remaining motif is over
    `max_length`, it is also deemed to not pass, because IC is not concentrated enough.
    This also only keeps motifs with overall positive contributions (i.e. no negative
    seqlets).
    Returns 2 parallel lists: a list of motif CWMs, and a list of motif PWMs.
    """
    cwms, pwms = [], []
    num_seqlets = []
    with h5py.File(tfm_results_hdf5, "r") as f:
        metaclusters = f["metacluster_idx_to_submetacluster_results"]
        num_metaclusters = len(metaclusters.keys())
        for metacluster_i, metacluster_key in enumerate(list(metaclusters.keys())):
            metacluster = metaclusters[metacluster_key]
            if plot_all_motifs:
                print("Metacluster: %s (%d/%d)" % (metacluster_key, metacluster_i + 1, num_metaclusters))
                print("==========================================")
            patterns = metacluster["seqlets_to_patterns_result"]["patterns"]
            num_patterns = len(patterns["all_pattern_names"][:])
            for pattern_i, pattern_name in enumerate(patterns["all_pattern_names"]):
                pattern_name = pattern_name.decode()
                pattern = patterns[pattern_name]
                seqlets = pattern["seqlets_and_alnmts"]["seqlets"]
                x = np.array([int(s.split(",")[0].split(":")[1]) for s in seqlets[:].astype(str)])
                print(np.max(x))
                
                if plot_all_motifs:
                    pattern_dir = os.path.join(out_dir, "all", metacluster_key, pattern_name)
                    os.makedirs(pattern_dir, exist_ok=True)
                    print("Pattern: %s (%d/%d)" % (pattern_name, pattern_i + 1, num_patterns))
                    print("--------------------------------------")

                    print("%d seqlets" % len(seqlets))
                    print("Sequence")
                    viz_sequence.plot_weights(pattern["sequence"]["fwd"][:], out_path=os.path.join(pattern_dir, "sequence.svg"))
                    print("Hypothetical contributions")
                    viz_sequence.plot_weights(pattern["task0_hypothetical_contribs"]["fwd"][:], out_path=os.path.join(pattern_dir, "hypothetical_contribs.svg"))
                    print("Contribution_scores")
                    viz_sequence.plot_weights(pattern["task0_contrib_scores"]["fwd"][:], out_path=os.path.join(pattern_dir, "contrib_scores.svg")) 
                    plt.clf()

                pfm = pattern["sequence"]["fwd"][:]
                # print(pfm) ####
                act_contribs = pattern["task0_contrib_scores"]["fwd"][:]
                
                # Check that the contribution scores are overall positive
                if np.sum(act_contribs) < 0:
                    continue
                
                # Check number of seqlets and IC
                if len(seqlets) < min_seqlets:
                    continue
                
                pwm = pfm_to_pwm(pfm, background_freqs)
                print(pwm) ####
                pwm_ic = pfm_info_content(pfm, background_freqs)
                max_windowed_ic = max(
                    np.sum(pwm_ic[i : (i + ic_window)]) for i in range(len(pwm_ic) - ic_window + 1)
                )
                if max_windowed_ic / ic_window < min_ic:
                    continue
                    
                # Cut off flanks from actual contribution scores and PWM based on IC of PWM
                ic_trim_thresh = np.max(pwm_ic) * trim_flank_ic_frac
                pass_inds = np.where(pwm_ic >= ic_trim_thresh)[0]
                trimmed_cwm = act_contribs[np.min(pass_inds): np.max(pass_inds) + 1]
                trimmed_pwm = pwm[np.min(pass_inds): np.max(pass_inds) + 1]
                
                # If too long after trimming, IC is not concentrated enough; toss out;
                # it is almost certainly a homopolymer repeat
                if len(trimmed_cwm) > max_length:
                    continue
                
                # Last check to make sure motif is overall positive
                if np.sum(trimmed_cwm) < 0:
                    continue

                cwms.append(trimmed_cwm)
                pwms.append(trimmed_pwm)
                num_seqlets.append(len(seqlets))

    if plot_passed_motifs:
        print("Final motifs: %d total" % len(cwms))
        print("==========================================")
        passed_dir = os.path.join(out_dir, "final")
        os.makedirs(passed_dir, exist_ok=True)
        for i in range(len(cwms)):
            print("Motif %d (%d seqlets)" % (i + 1, num_seqlets[i]))
            viz_sequence.plot_weights(cwms[i], out_path=os.path.join(passed_dir, f"cwm_{i}.svg"))
            viz_sequence.plot_weights(pwms[i], out_path=os.path.join(passed_dir, f"pwm_{i}.svg"))
            plt.clf()

    return cwms, pwms

if __name__ == '__main__':
    # Path to SHAP scores and TF-MoDISco results
    shap_dir = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/deepshap/transfer_v5/"
    tfm_dir = "/mnt/lab_data2/atwang/models/domain_adapt/dnase/modisco/transfer_v5/"
    out_dir_base = "/users/atwang/results/domain_adapt_results/dnase_models/"

    background_freqs = np.array([0.27, 0.23, 0.23, 0.27])

    cell_types = {
        "K562": ["ENCSR000EOT"],
        "HepG2": ["ENCSR149XIL"]
    }

    for i, i_ex in cell_types.items():
        for j, j_ex in cell_types.items():
            if i == j:
                continue

            shap_scores_path = os.path.join(shap_dir, f"{i}_from_{j}_shap.h5")
            tfm_results_path = os.path.join(tfm_dir, f"{i}_from_{j}_modisco.h5")

            out_dir = os.path.join(out_dir_base, f"{i}_from_{j}", "motifs")

            motifs = import_tfmodisco_motifs(tfm_results_path, background_freqs, out_dir, plot_all_motifs=True, plot_passed_motifs=True)

            # motifs = import_tfmodisco_motifs(tfm_results_path, background_freqs, min_seqlets=0, min_ic=0.6, trim_flank_ic_frac=0, max_length=100, plot_all_motifs=False, plot_passed_motifs=True)

            # viz_sequence.plot_weights(np.flip(motifs[0][9], axis=(0, 1)))

            plot_shap(shap_scores_path, out_dir)

