import os
import json
import tqdm
import torch
import numpy as np
import model.profile_models as profile_models 
import model.profile_performance_recompute as profile_performance 
import model.util as util
import feature.make_profile_dataset as make_profile_dataset


def get_model_path(metrics_json_path):
    """
    Returns the path to the saved model with the best validation loss.
    """
    with open(metrics_json_path, "r") as f:
        metrics = json.load(f)
    val_epoch_loss = metrics["val_epoch_loss"]["values"]
    best_epoch = np.argmin(val_epoch_loss) + 1
    return os.path.join(
        os.path.dirname(metrics_json_path),
        "model_ckpt_epoch_%d.pt" % best_epoch
    )

    
def model_loss(
    config, model, true_profs, log_pred_profs, log_pred_counts, epoch_num,
    input_grads=None, status=None
):
    counts_loss_weight = config["counts_loss_weight"]
    att_prior_loss_weight = config["att_prior_loss_weight"]
    att_prior_loss_weight_anneal_type = config["att_prior_loss_weight_anneal_type"]
    att_prior_loss_weight_anneal_speed = config["att_prior_loss_weight_anneal_speed"]
    att_prior_grad_smooth_sigma = config["att_prior_grad_smooth_sigma"]
    fourier_att_prior_freq_limit = config["fourier_att_prior_freq_limit"]
    fourier_att_prior_freq_limit_softness = config["fourier_att_prior_freq_limit_softness"]
    att_prior_loss_only = config["att_prior_loss_only"]

    corr_loss, prof_loss, count_loss = model.correctness_loss(
        true_profs, log_pred_profs, log_pred_counts, counts_loss_weight,
        return_separate_losses=True
    )
    
    if not att_prior_loss_weight:
        return corr_loss, (corr_loss, torch.zeros(1)), \
            (prof_loss, count_loss)
   
    att_prior_loss = model.fourier_att_prior_loss(
        status, input_grads, fourier_att_prior_freq_limit,
        fourier_att_prior_freq_limit_softness, att_prior_grad_smooth_sigma
    )
    
    if att_prior_loss_weight_anneal_type is None:
        weight = att_prior_loss_weight
    elif att_prior_loss_weight_anneal_type == "inflate":
        exp = np.exp(-att_prior_loss_weight_anneal_speed * epoch_num)
        weight = att_prior_loss_weight * ((2 / (1 + exp)) - 1)
    elif att_prior_loss_weight_anneal_type == "deflate":
        exp = np.exp(-att_prior_loss_weight_anneal_speed * epoch_num)
        weight = att_prior_loss_weight * exp

    if att_prior_loss_only:
        final_loss = att_prior_loss
    else:
        final_loss = corr_loss + (weight * att_prior_loss)

    return final_loss, (corr_loss, att_prior_loss), (prof_loss, count_loss)


def run_epoch(
    config, data_loader, mode, model, epoch_num, optimizer=None,
    return_data=False
):
    num_tasks = config["num_tasks"]
    controls = config["controls"]
    att_prior_loss_weight = config["att_prior_loss_weight"]
    batch_size = config["batch_size"]
    revcomp = config["revcomp"]
    input_length = config["input_length"]
    input_depth = config["input_depth"]
    profile_length = config["profile_length"]

    assert mode in ("train", "eval")
    if mode == "train":
        assert optimizer is not None
    else:
        assert optimizer is None 

    data_loader.dataset.on_epoch_start()  # Set-up the epoch
    num_batches = len(data_loader.dataset)
    t_iter = tqdm.tqdm(
        data_loader, total=num_batches, desc="\tLoss: ---"
    )

    if mode == "train":
        model.train()  # Switch to training mode
        torch.set_grad_enabled(True)

    batch_losses, corr_losses, att_losses = [], [], []
    prof_losses, count_losses = [], []
    if return_data:
        # Allocate empty NumPy arrays to hold the results
        num_samples_exp = num_batches * batch_size
        num_samples_exp *= 2 if revcomp else 1
        # Real number of samples can be smaller because of partial last batch
        profile_shape = (num_samples_exp, num_tasks, profile_length, 2)
        count_shape = (num_samples_exp, num_tasks, 2)
        all_log_pred_profs = np.empty(profile_shape)
        all_log_pred_counts = np.empty(count_shape)
        all_true_profs = np.empty(profile_shape)
        all_true_counts = np.empty(count_shape)
        all_input_seqs = np.empty((num_samples_exp, input_length, input_depth))
        all_input_grads = np.empty((num_samples_exp, input_length, input_depth))
        all_coords = np.empty((num_samples_exp, 3), dtype=object)
        num_samples_seen = 0  # Real number of samples seen

    for input_seqs, profiles, statuses, coords, peaks in t_iter:
        if return_data:
            input_seqs_np = input_seqs
        input_seqs = util.place_tensor(torch.tensor(input_seqs)).float()
        profiles = util.place_tensor(torch.tensor(profiles)).float()

        if controls is not None:
            tf_profs = profiles[:, :num_tasks, :, :]
            cont_profs = profiles[:, num_tasks:, :, :]  # Last half or just one
        else:
            tf_profs, cont_profs = profiles, None

        # Clear gradients from last batch if training
        if mode == "train":
            optimizer.zero_grad()
        elif att_prior_loss_weight > 0:
            # Not training mode, but we still need to zero out weights because
            # we are computing the input gradients
            model.zero_grad()

        if att_prior_loss_weight > 0:
            input_seqs.requires_grad = True  # Set gradient required
            logit_pred_profs, log_pred_counts = model(input_seqs, cont_profs)

            # Subtract mean along output profile dimension; this wouldn't change
            # softmax probabilities, but normalizes the magnitude of gradients
            norm_logit_pred_profs = logit_pred_profs - \
                torch.mean(logit_pred_profs, dim=2, keepdim=True)

            # Weight by post-softmax probabilities, but do not take the
            # gradients of these probabilities; this upweights important regions
            # exponentially
            pred_prof_probs = profile_models.profile_logits_to_log_probs(
                logit_pred_profs
            ).detach()
            weighted_norm_logits = norm_logit_pred_profs * pred_prof_probs

            input_grads, = torch.autograd.grad(
                weighted_norm_logits, input_seqs,
                grad_outputs=util.place_tensor(
                    torch.ones(weighted_norm_logits.size())
                ),
                retain_graph=True, create_graph=True
                # We'll be operating on the gradient itself, so we need to
                # create the graph
                # Gradients are summed across strands and tasks
            )
            if return_data:
                input_grads_np = input_grads.detach().cpu().numpy()
            input_grads = input_grads * input_seqs  # Gradient * input
            status = util.place_tensor(torch.tensor(statuses))
            status[status != 0] = 1  # Set to 1 if not negative example
            input_seqs.requires_grad = False  # Reset gradient required
        else:
            logit_pred_profs, log_pred_counts = model(input_seqs, cont_profs)
            status, input_grads = None, None

        loss, (corr_loss, att_loss), (prof_loss, count_loss) = model_loss(
            config, model, tf_profs, logit_pred_profs, log_pred_counts,
            epoch_num, status=status, input_grads=input_grads
        )

        if mode == "train":
            loss.backward()  # Compute gradient
            optimizer.step()  # Update weights through backprop

        batch_losses.append(loss.item())
        corr_losses.append(corr_loss.item())
        att_losses.append(att_loss.item())
        prof_losses.append(prof_loss.item())
        count_losses.append(count_loss.item())
        t_iter.set_description(
            "\tLoss: %6.4f" % loss.item()
        )

        if return_data:
            logit_pred_profs_np = logit_pred_profs.detach().cpu().numpy()
            log_pred_counts_np = log_pred_counts.detach().cpu().numpy()
            true_profs_np = tf_profs.detach().cpu().numpy()
            true_counts = np.sum(true_profs_np, axis=2)

            num_in_batch = true_counts.shape[0]
          
            # Turn logit profile predictions into log probabilities
            log_pred_profs = profile_models.profile_logits_to_log_probs(
                logit_pred_profs_np, axis=2
            )

            # Fill in the batch data/outputs into the preallocated arrays
            start, end = num_samples_seen, num_samples_seen + num_in_batch
            all_log_pred_profs[start:end] = log_pred_profs
            all_log_pred_counts[start:end] = log_pred_counts_np
            all_true_profs[start:end] = true_profs_np
            all_true_counts[start:end] = true_counts
            all_input_seqs[start:end] = input_seqs_np
            if att_prior_loss_weight:
                all_input_grads[start:end] = input_grads_np
            all_coords[start:end] = coords

            num_samples_seen += num_in_batch

    if return_data:
        # Truncate the saved data to the proper size, based on how many
        # samples actually seen
        all_log_pred_profs = all_log_pred_profs[:num_samples_seen]
        all_log_pred_counts = all_log_pred_counts[:num_samples_seen]
        all_true_profs = all_true_profs[:num_samples_seen]
        all_true_counts = all_true_counts[:num_samples_seen]
        all_input_seqs = all_input_seqs[:num_samples_seen]
        all_input_grads = all_input_grads[:num_samples_seen]
        all_coords = all_coords[:num_samples_seen]
        return batch_losses, corr_losses, att_losses, prof_losses, \
            count_losses, all_true_profs, all_log_pred_profs, \
            all_true_counts, all_log_pred_counts, all_coords, all_input_grads, \
            all_input_seqs
    else:
        return batch_losses, corr_losses, att_losses, prof_losses, count_losses


def get_data_loader(config):
    peaks_bed_paths = config["peak_beds"]
    profile_hdf5_path = config["profile_hdf5"]

    batch_size = config["batch_size"]
    revcomp = config["revcomp"]
    input_length = config["input_length"]
    profile_length = config["profile_length"]
    test_chroms = config["test_chroms"]
    reference_fasta = config["dataset"]["reference_fasta"]
    chrom_sizes_tsv = config["dataset"]["chrom_sizes_tsv"]
    negative_ratio = config["dataset"]["negative_ratio"]
    peak_tiling_stride = config["dataset"]["peak_tiling_stride"]
    peak_retention = config["dataset"]["peak_retention"]
    jitter_size = config["dataset"]["jitter_size"]
    num_workers = config["dataset"]["num_workers"]
    if "negative_seed" in config["dataset"]:
        negative_seed = config["dataset"]["negative_seed"]
    else:
        negative_seed = None
    shuffle_seed = config["dataset"]["shuffle_seed"]
    jitter_seed = config["dataset"]["jitter_seed"]

    test_loader = make_profile_dataset.create_data_loader(
        peaks_bed_paths, profile_hdf5_path, "SummitCenteringCoordsBatcher",
        batch_size, reference_fasta, chrom_sizes_tsv, input_length,
        profile_length, negative_ratio, peak_tiling_stride, None,
        num_workers, revcomp, jitter_size, negative_seed, shuffle_seed,
        jitter_seed, chrom_set=test_chroms, shuffle=True, return_coords=True
    )
    return test_loader


def recompute_test_metrics(config_json_path, model):
    with open(config_json_path, "r") as f:
        config = json.load(f)

    test_loader = get_data_loader(config)
    
    batch_losses, corr_losses, att_losses, prof_losses, count_losses, \
        true_profs, log_pred_profs, true_counts, log_pred_counts, coords, \
        input_grads, input_seqs = run_epoch(
            config, test_loader, "eval", model, 0, return_data=True
        )
    
    metrics = profile_performance.compute_performance_metrics(
        true_profs, log_pred_profs, true_counts, log_pred_counts
    )
    return batch_losses, corr_losses, att_losses, prof_losses, count_losses, \
        metrics


def update_metrics_json(
    metrics_json_path, batch_losses, corr_losses, att_losses, prof_losses,
    count_losses, test_metrics
):
    with open(metrics_json_path, "r") as f:
        metrics = json.load(f)

    metrics["test_summit_batch_losses"]["values"] = [batch_losses]
    metrics["test_summit_corr_losses"]["values"] = [corr_losses]
    metrics["test_summit_att_losses"]["values"] = [att_losses]
    metrics["test_summit_prof_corr_losses"]["values"] = [prof_losses]
    metrics["test_summit_count_corr_losses"]["values"] = [count_losses]

    # Before logging, condense the metrics into averages over the samples (when
    # appropriate)
    nll = np.nanmean(test_metrics["nll"], axis=0)  # T
    jsd = np.nanmean(test_metrics["jsd"], axis=0)  # T
    auprc_bin = np.nanmean(test_metrics["auprc_binned"][:, :, :, 0], axis=0)  # T x Z
    pears_bin = np.nanmean(test_metrics["pearson_binned"], axis=0)  # T x Z
    spear_bin = np.nanmean(test_metrics["spearman_binned"], axis=0)  # T x Z
    mse_bin = np.nanmean(test_metrics["mse_binned"], axis=0)  # T x Z
    pears_tot = test_metrics["pearson_total"]  # T
    spear_tot = test_metrics["spearman_total"]  # T
    mse_tot = test_metrics["mse_total"]  # T
    # At this point, these metrics are all extracted from the dictionary and are
    # either T-arrays or T x Z arrays (where T is the number of tasks and Z is
    # the number of bin sizes for a metric)

    metrics["summit_prof_nll"]["values"] = [list(nll)]
    metrics["summit_prof_jsd"]["values"] = [list(jsd)]
    bin_sizes = [1, 4, 10]
    for i, bin_size in enumerate(bin_sizes):
        metrics["summit_prof_auprc_bin%d" % bin_size]["values"] = [list(auprc_bin[:, i])]
        metrics["summit_prof_pearson_bin%d" % bin_size]["values"] = [list(pears_bin[:, i])]
        metrics["summit_prof_spearman_bin%d" % bin_size]["values"] = [list(spear_bin[:, i])]
        metrics["summit_prof_mse_bin%d" % bin_size]["values"] = [list(mse_bin[:, i])]
    metrics["summit_count_pearson"]["values"] = [list(pears_tot)]
    metrics["summit_count_spearman"]["values"] = [list(spear_tot)]
    metrics["summit_count_mse"]["values"] = [list(mse_tot)]

    new_metrics_path = os.path.join(
        os.path.dirname(metrics_json_path), "metrics_testupdate.json"
    )
    with open(new_metrics_path, "w") as f:
        json.dump(metrics, f, sort_keys=True, indent=2)


def recompute(run_path):
    metrics_json_path = os.path.join(run_path, "metrics.json")
    config_json_path = os.path.join(run_path, "config.json")

    with open(config_json_path, "r") as f:
        config = json.load(f)
    controls = config["controls"]

    if controls is None:
        model_class = profile_models.ProfilePredictorWithoutControls
    if controls == "matched":
        model_class = profile_models.ProfilePredictorWithMatchedControls
    elif controls == "shared":
        model_class = profile_models.ProfilePredictorWithSharedControls

    model_path = get_model_path(metrics_json_path)
    model = util.restore_model(model_class, model_path)
    model.cuda()
    
    batch_losses, corr_losses, att_losses, prof_losses, \
        count_losses, test_metrics = recompute_test_metrics(
            config_json_path, model
        )

    update_metrics_json(
        metrics_json_path, batch_losses, corr_losses, att_losses, prof_losses,
        count_losses, test_metrics
    )


if __name__ == "__main__":
    import sys
    run_path = sys.argv[1]
    recompute(run_path)
