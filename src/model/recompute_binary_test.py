import os
import json
import tqdm
import torch
import numpy as np
import model.binary_models as binary_models
import model.binary_performance as binary_performance
import model.util as util
import feature.make_binary_dataset as make_binary_dataset


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
    config, model, true_vals, logit_pred_vals, epoch_num, input_grads=None,
    status=None
):
    avg_class_loss = config["avg_class_loss"]
    att_prior_loss_weight = config["att_prior_loss_weight"]
    att_prior_loss_weight_anneal_type = config["att_prior_loss_weight_anneal_type"]
    att_prior_loss_weight_anneal_speed = config["att_prior_loss_weight_anneal_speed"]
    att_prior_grad_smooth_sigma = config["att_prior_grad_smooth_sigma"]
    fourier_att_prior_freq_limit = config["fourier_att_prior_freq_limit"]
    fourier_att_prior_freq_limit_softness = config["fourier_att_prior_freq_limit_softness"]
    att_prior_loss_only = config["att_prior_loss_only"]

    corr_loss = model.correctness_loss(
        true_vals, logit_pred_vals, avg_class_loss
    )
    
    if not att_prior_loss_weight:
        return corr_loss, (corr_loss, torch.zeros(1))
   
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

    return final_loss, (corr_loss, att_prior_loss)


def run_epoch(
    config, data_loader, mode, model, epoch_num, optimizer=None,
    return_data=False
):
    num_tasks = config["num_tasks"]
    att_prior_loss_weight = config["att_prior_loss_weight"]
    batch_size = config["batch_size"]
    revcomp = config["revcomp"]
    input_length = config["input_length"]
    input_depth = config["input_depth"]

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
    if return_data:
        # Allocate empty NumPy arrays to hold the results
        num_samples_exp = num_batches * batch_size
        num_samples_exp *= 2 if revcomp else 1
        # Real number of samples can be smaller because of partial last batch
        all_true_vals = np.empty((num_samples_exp, num_tasks))
        all_pred_vals = np.empty((num_samples_exp, num_tasks))
        all_input_seqs = np.empty((num_samples_exp, input_length, input_depth))
        all_input_grads = np.empty((num_samples_exp, input_length, input_depth))
        all_coords = np.empty((num_samples_exp, 3), dtype=object)
        num_samples_seen = 0  # Real number of samples seen

    for input_seqs, output_vals, statuses, coords in t_iter:
        if return_data:
            input_seqs_np = input_seqs
            output_vals_np = output_vals
        input_seqs = util.place_tensor(torch.tensor(input_seqs)).float()
        output_vals = util.place_tensor(torch.tensor(output_vals)).float()

        # Clear gradients from last batch if training
        if mode == "train":
            optimizer.zero_grad()
        elif att_prior_loss_weight > 0:
            # Not training mode, but we still need to zero out weights because
            # we are computing the input gradients
            model.zero_grad()

        if att_prior_loss_weight > 0:
            input_seqs.requires_grad = True  # Set gradient required
            logit_pred_vals = model(input_seqs)
            # Compute the gradients of the output with respect to the input
            input_grads, = torch.autograd.grad(
                logit_pred_vals, input_seqs,
                grad_outputs=util.place_tensor(
                    torch.ones(logit_pred_vals.size())
                ),
                retain_graph=True, create_graph=True
                # We'll be operating on the gradient itself, so we need to
                # create the graph
                # Gradients are summed across tasks
            )
            if return_data:
                input_grads_np = input_grads.detach().cpu().numpy()
            input_grads = input_grads * input_seqs  # Gradient * input
            status = util.place_tensor(torch.tensor(statuses))
            input_seqs.requires_grad = False  # Reset gradient required
        else:
            logit_pred_vals = model(input_seqs)
            status, input_grads = None, None

        loss, (corr_loss, att_loss) = model_loss(
            config, model, output_vals, logit_pred_vals, epoch_num,
            input_grads=input_grads, status=status
        )

        if mode == "train":
            loss.backward()  # Compute gradient
            optimizer.step()  # Update weights through backprop

        batch_losses.append(loss.item())
        corr_losses.append(corr_loss.item())
        att_losses.append(att_loss.item())
        t_iter.set_description(
            "\tLoss: %6.4f" % loss.item()
        )

        if return_data:
            logit_pred_vals_np = logit_pred_vals.detach().cpu().numpy()
          
            # Turn logits into probabilities
            pred_vals = binary_models.binary_logits_to_probs(logit_pred_vals_np)
            num_in_batch = pred_vals.shape[0]

            # Fill in the batch data/outputs into the preallocated arrays
            start, end = num_samples_seen, num_samples_seen + num_in_batch
            all_true_vals[start:end] = output_vals_np
            all_pred_vals[start:end] = pred_vals
            all_input_seqs[start:end] = input_seqs_np
            if att_prior_loss_weight:
                all_input_grads[start:end] = input_grads_np
            all_coords[start:end] = coords

            num_samples_seen += num_in_batch

    if return_data:
        # Truncate the saved data to the proper size, based on how many
        # samples actually seen
        all_true_vals = all_true_vals[:num_samples_seen]
        all_pred_vals = all_pred_vals[:num_samples_seen]
        all_input_seqs = all_input_seqs[:num_samples_seen]
        all_input_grads = all_input_grads[:num_samples_seen]
        all_coords = all_coords[:num_samples_seen]
        return batch_losses, corr_losses, att_losses, all_true_vals, \
            all_pred_vals, all_coords, all_input_grads, all_input_seqs
    else:
        return batch_losses, corr_losses, att_losses


def get_data_loader(config):
    labels_hdf5 = config["labels_hdf5"]
    bin_labels_npy = config["bin_labels_npy"]
    peak_signals_npy = config["peak_signals_npy"]
    bin_labels_array = np.load(bin_labels_npy, allow_pickle=True)
    peak_signals_array = np.load(peak_signals_npy) if peak_signals_npy else None

    batch_size = config["batch_size"]
    revcomp = config["revcomp"]
    input_length = config["input_length"]
    test_chroms = config["test_chroms"]
    reference_fasta = config["dataset"]["reference_fasta"]
    negative_ratio = config["dataset"]["negative_ratio"]
    peak_retention = config["dataset"]["peak_retention"]
    num_workers = config["dataset"]["num_workers"]
    if "negative_seed" in config["dataset"]:
        negative_seed = config["dataset"]["negative_seed"]
    else:
        negative_seed = None
    shuffle_seed = config["dataset"]["shuffle_seed"]

    test_loader = make_binary_dataset.create_data_loader(
        labels_hdf5, bin_labels_array, batch_size, reference_fasta,
        input_length, negative_ratio, None, num_workers, revcomp,
        negative_seed, shuffle_seed, 
        peak_signals_npy_or_array=peak_signals_array, chrom_set=test_chroms,
        shuffle=True, return_coords=True
    )
    return test_loader


def recompute_test_metrics(config_json_path, model):
    with open(config_json_path, "r") as f:
        config = json.load(f)

    test_loader = get_data_loader(config)
    
    batch_losses, corr_losses, att_losses, true_vals, pred_vals, coords, \
        input_grads, input_seqs = run_epoch(
            config, test_loader, "eval", model, 0, return_data=True
    )

    neg_upsample_factor = test_loader.dataset.bins_batcher.neg_to_pos_imbalance
    metrics = binary_performance.compute_performance_metrics(
        true_vals, pred_vals, neg_upsample_factor
    )
    return batch_losses, corr_losses, att_losses, metrics


def update_metrics_json(
    metrics_json_path, batch_losses, corr_losses, att_losses, test_metrics
):
    with open(metrics_json_path, "r") as f:
        metrics = json.load(f)

    metrics["test_batch_losses"]["values"] = [batch_losses]
    metrics["test_corr_losses"]["values"] = [corr_losses]
    metrics["test_att_losses"]["values"] = [att_losses]
    metrics["test_acc"]["values"] = [test_metrics["acc"]]
    metrics["test_pos_acc"]["values"] = [test_metrics["pos_acc"]]
    metrics["test_neg_acc"]["values"] = [test_metrics["neg_acc"]]
    metrics["test_auroc"]["values"] = [test_metrics["auroc"]]
    metrics["test_precis_score"]["values"] = [test_metrics["precis_score"]]
    metrics["test_recall_score"]["values"] = [test_metrics["recall_score"]]
    metrics["test_auprc"]["values"] = [test_metrics["auprc"]]
    metrics["test_corr_precis_score"]["values"] = [test_metrics["c_precis_score"]]
    metrics["test_corr_recall_score"]["values"] = [test_metrics["c_recall_score"]]
    metrics["test_corr_auprc"]["values"] = [test_metrics["c_auprc"]]

    new_metrics_path = os.path.join(
        os.path.dirname(metrics_json_path), "metrics_testupdate.json"
    )
    with open(new_metrics_path, "w") as f:
        json.dump(metrics, f, sort_keys=True, indent=2)


def recompute(run_path):
    metrics_json_path = os.path.join(run_path, "metrics.json")
    config_json_path = os.path.join(run_path, "config.json")

    model_class = binary_models.BinaryPredictor

    model_path = get_model_path(metrics_json_path)
    model = util.restore_model(model_class, model_path)
    model.cuda()
    
    batch_losses, corr_losses, att_losses, test_metrics = \
        recompute_test_metrics(config_json_path, model)

    update_metrics_json(
        metrics_json_path, batch_losses, corr_losses, att_losses, test_metrics
    )


if __name__ == "__main__":
    import sys
    run_path = sys.argv[1]
    recompute(run_path)
