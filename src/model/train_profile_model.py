import numpy as np
import sacred
import torch
import math
import tqdm
import os
import model.util as util
import model.profile_models as profile_models
import model.profile_performance as profile_performance
import feature.make_profile_dataset as make_profile_dataset

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/mnt/lab_data2/atwang/models/domain_adapt/dnase/trained_models/baseline_v2/"
)

train_ex = sacred.Experiment("train", ingredients=[
    make_profile_dataset.dataset_ex,
    profile_performance.performance_ex
])
train_ex.observers.append(
    sacred.observers.FileStorageObserver.create(MODEL_DIR)
)

@train_ex.config
def config(dataset):
    controls = "matched"

    # Number of dilating convolutional layers to apply
    num_dil_conv_layers = 9

    # Number of filters to use for each dilating convolutional layer (i.e.
    # number of channels to output)
    dil_conv_depth = 64

    # Type of annealing; can be None (constant/no annealing), "inflate" (follows
    # `2/(1 + e^(-c*x)) - 1`), or "deflate" (follows `e^(-c * x)`)
    att_prior_loss_weight_anneal_type = None

    # Annealing factor for attribution prior loss weight, c
    if att_prior_loss_weight_anneal_type is None:
        att_prior_loss_weight_anneal_speed = None
    elif att_prior_loss_weight_anneal_type == "inflate":
        att_prior_loss_weight_anneal_speed = 1
    elif att_prior_loss_weight_anneal_type == "deflate":
        att_prior_loss_weight_anneal_speed = 0.3

    params = {
        "gpu_id": "4",

        "prof_trans_conv_kernel_size": 3,
        "prof_trans_conv_channels": [10],

        # Number of dilating convolutional layers to apply
        "num_dil_conv_layers": num_dil_conv_layers,

        # Size of dilating convolutional filters to apply
        "dil_conv_filter_sizes": [21] + ([3] * (num_dil_conv_layers - 1)),

        # Stride for dilating convolutional layers
        "dil_conv_stride": 1,

        # Number of filters to use for each dilating convolutional layer (i.e.
        # number of channels to output)
        "dil_conv_depth": dil_conv_depth,
        "dil_conv_depths": [dil_conv_depth] * num_dil_conv_layers,

        # Dilation values for each of the dilating convolutional layers
        "dil_conv_dilations": [2 ** i for i in range(num_dil_conv_layers)],

        # Size of filter for large profile convolution
        "prof_conv_kernel_size": 75,

        # Stride for large profile convolution
        "prof_conv_stride": 1,

        # Number of prediction tasks
        "num_tasks": 1,

        # Number of strands; typically 1 (unstranded) or 2 (plus/minus strand)
        "num_strands": 1,

        # Type of control profiles (if any) to use in model; can be "matched" (each
        # task has a matched control), "shared" (all tasks share a control), or
        # None (no controls)
        "controls": controls,
        "share_controls": controls == "shared",

        # Amount to weight the counts loss within the correctness loss
        "counts_loss_weight": 20,

        # Weight to use for attribution prior loss; set to 0 to not use att. priors
        # "att_prior_loss_weight": 50,
        "att_prior_loss_weight": 0, ####

        # Type of annealing; can be None (constant/no annealing), "inflate" (follows
        # `2/(1 + e^(-c*x)) - 1`), or "deflate" (follows `e^(-c * x)`)
        "att_prior_loss_weight_anneal_type": att_prior_loss_weight_anneal_type,

        # Annealing factor for attribution prior loss weight, c
        "att_prior_loss_weight_anneal_speed": att_prior_loss_weight_anneal_speed,

        # Smoothing amount for gradients before computing attribution prior loss;
        # Smoothing window size is 1 + (2 * sigma); set to 0 for no smoothing
        "att_prior_grad_smooth_sigma": 3,

        # Maximum frequency integer to consider for a Fourier attribution prior
        "fourier_att_prior_freq_limit": 200,

        # Amount to soften the Fourier attribution prior loss limit; set to None
        # to not soften; softness decays like 1 / (1 + x^c) after the limit
        "fourier_att_prior_freq_limit_softness": 0.2,

        # Number of training epochs
        "num_epochs": 20,

        "num_epochs_prof": 5,

        # Learning rate
        "learning_rate": 0.001,

        # Whether or not to use early stopping
        "early_stopping": True,

        # Number of epochs to save validation loss (set to 1 for one step only)
        "early_stop_hist_len": 3,

        # Minimum improvement in loss at least once over history to not stop early
        "early_stop_min_delta": 0.001,

        # Training seed
        "train_seed": None,

        # If set, ignore correctness loss completely
        "att_prior_loss_only": False,

        # Imported from make_profile_dataset
        "batch_size": dataset["batch_size"],

        # Imported from make_profile_dataset
        "revcomp": dataset["revcomp"],

        # Imported from make_profile_dataset
        "input_length": dataset["input_length"],
        
        # Imported from make_profile_dataset
        "input_depth": dataset["input_depth"],

        # Imported from make_profile_dataset
        "profile_length": dataset["profile_length"],
        
        # Imported from make_profile_dataset
        "negative_ratio": dataset["negative_ratio"],
    }


# @train_ex.capture
# def create_model(
#     controls, input_length, input_depth, profile_length, num_tasks, num_strands,
#     num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
#     dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size, prof_conv_stride
# ):
#     """
#     Creates a profile model using the configuration above.
#     """
#     if controls is None:
#         predictor_class = profile_models.ProfilePredictorWithoutControls
#     if controls == "matched":
#         predictor_class = profile_models.ProfilePredictorWithMatchedControls
#     elif controls == "shared":
#         predictor_class = profile_models.ProfilePredictorWithSharedControls

#     prof_model = predictor_class(
#         input_length=input_length,
#         input_depth=input_depth,
#         profile_length=profile_length,
#         num_tasks=num_tasks,
#         num_strands=num_strands,
#         num_dil_conv_layers=num_dil_conv_layers,
#         dil_conv_filter_sizes=dil_conv_filter_sizes,
#         dil_conv_stride=dil_conv_stride,
#         dil_conv_dilations=dil_conv_dilations,
#         dil_conv_depths=dil_conv_depths,
#         prof_conv_kernel_size=prof_conv_kernel_size,
#         prof_conv_stride=prof_conv_stride
#     )

#     return prof_model

@train_ex.capture
def create_model(**kwargs):
    """
    Creates a profile model using the configuration above.
    """

    return profile_models.ProfilePredictorWithControlsKwargs(**kwargs)

@train_ex.capture
def model_loss(
    model, true_profs, log_pred_profs, log_pred_counts, epoch_num, params,
    input_grads=None, status=None
):
    """
    Computes the loss for the model.
    Arguments:
        `model`: the model being trained
        `true_profs`: a B x T x O x S tensor, where B is the batch size, T is
            the number of tasks, O is the length of the output profiles, and S
            is the number of strands (1 or 2); this contains true profile read
            counts (unnormalized)
        `log_pred_profs`: a B x T x O x S tensor, consisting of the output
            profile predictions as logits
        `log_pred_counts`: a B x T x S tensor consisting of the log counts
            predictions
        `epoch_num`: a 0-indexed integer representing the current epoch
        `input_grads`: a B x I x D tensor, where I is the input length and D is
            the input depth; this is the gradient of the output with respect to
            the input, times the input itself; only needed when attribution
            prior loss weight is positive
        `status`: a B-tensor, where B is the batch size; each entry is 1 if that
            that example is to be treated as a positive example, and 0
            otherwise; only needed when attribution prior loss weight is
            positive
    Returns a scalar Tensor containing the loss for the given batch, a pair
    consisting of the correctness loss and the attribution prior loss, and a
    pair for the profile loss and the counts loss.
    If the attribution prior loss is not computed at all, then 0 will be in its
    place, instead.
    """
    att_prior_loss_weight_anneal_type = params["att_prior_loss_weight_anneal_type"]
    att_prior_loss_weight_anneal_speed = params["att_prior_loss_weight_anneal_speed"]
    att_prior_grad_smooth_sigma = params["att_prior_grad_smooth_sigma"]
    fourier_att_prior_freq_limit = params["fourier_att_prior_freq_limit"]
    fourier_att_prior_freq_limit_softness = params["fourier_att_prior_freq_limit_softness"]
    att_prior_loss_only = params["att_prior_loss_only"]
    counts_loss_weight = params["counts_loss_weight"]
    att_prior_loss_weight = params["att_prior_loss_weight"]

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


@train_ex.capture
def run_epoch(
    data_loader, mode, model, epoch_num, params, optimizer=None, return_data=False
):
    """
    Runs the data from the data loader once through the model, to train,
    validate, or predict.
    Arguments:
        `data_loader`: an instantiated `DataLoader` instance that gives batches
            of data; each batch must yield the input sequences, profiles,
            statuses, coordinates, and peaks; if `controls` is "matched",
            profiles must be such that the first half are prediction (target)
            profiles, and the second half are control profiles; if `controls` is
            "shared", the last set of profiles is a shared control; otherwise,
            all tasks are prediction profiles
        `mode`: one of "train", "eval"; if "train", run the epoch and perform
            backpropagation; if "eval", only do evaluation
        `model`: the current PyTorch model being trained/evaluated
        `epoch_num`: 0-indexed integer representing the current epoch
        `optimizer`: an instantiated PyTorch optimizer, for training mode
        `return_data`: if specified, returns the following as NumPy arrays:
            true profile raw counts (N x T x O x S), predicted profile log
            probabilities (N x T x O x S), true total counts (N x T x S),
            predicted log counts (N x T x S), coordinates used (N x 3 object
            array), input gradients (N x I x 4), and input sequences (N x I x4);
            if the attribution prior is not used, the gradients will be garbage
    Returns lists of overall losses, correctness losses, attribution prior
    losses, profile losses, and count losses, where each list is over all
    batches. If the attribution prior loss is not computed, then it will be all
    0s. If `return_data` is True, then more things will be returned after these.
    """
    num_tasks = params["num_tasks"]
    controls = params["controls"]
    att_prior_loss_weight = params["att_prior_loss_weight"]
    batch_size = params["batch_size"]
    revcomp = params["revcomp"]
    input_length = params["input_length"]
    input_depth = params["input_depth"]
    profile_length = params["profile_length"]
    gpu_id = params.get("gpu_id")

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
        input_seqs = util.place_tensor(torch.tensor(input_seqs), index=gpu_id).float()
        profiles = util.place_tensor(torch.tensor(profiles), index=gpu_id).float()

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
                    torch.ones(weighted_norm_logits.size(), index=gpu_id)
                ),
                retain_graph=True, create_graph=True
                # We'll be operating on the gradient itself, so we need to
                # create the graph
                # Gradients are summed across strands and tasks
            )
            if return_data:
                input_grads_np = input_grads.detach().cpu().numpy()
            input_grads = input_grads * input_seqs  # Gradient * input
            status = util.place_tensor(torch.tensor(statuses), index=gpu_id)
            status[status != 0] = 1  # Set to 1 if not negative example
            input_seqs.requires_grad = False  # Reset gradient required
        else:
            logit_pred_profs, log_pred_counts = model(input_seqs, cont_profs)
            status, input_grads = None, None

        loss, (corr_loss, att_loss), (prof_loss, count_loss) = model_loss(
            model, tf_profs, logit_pred_profs, log_pred_counts, epoch_num,
            status=status, input_grads=input_grads
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


@train_ex.capture
def train_model(
    train_loader, val_loader, test_summit_loader, test_peak_loader,
    test_genome_loader, trans_id, params, _run
):
    """
    Trains the network for the given training and validation data.
    Arguments:
        `train_loader` (DataLoader): a data loader for the training data
        `val_loader` (DataLoader): a data loader for the validation data
        `test_summit_loader` (DataLoader): a data loader for the test data, with
            coordinates centered at summits
        `test_peak_loader` (DataLoader): a data loader for the test data, with
            coordinates tiled across peaks
        `test_genome_loader` (DataLoader): a data loader for the test data, with
            summit-centered coordinates augmented with sampled negatives
    Note that all data loaders are expected to yield the 1-hot encoded
    sequences, profiles, statuses, source coordinates, and source peaks.
    """
    num_epochs = params["num_epochs"]
    num_epochs_prof = params["num_epochs_prof"]
    learning_rate = params["learning_rate"]
    early_stopping = params["early_stopping"]
    early_stop_hist_len = params["early_stop_hist_len"]
    early_stop_min_delta = params["early_stop_min_delta"]
    train_seed = params["train_seed"]


    run_num = _run._id
    output_dir = os.path.join(MODEL_DIR, f"{trans_id}_{run_num}")
    os.makedirs(output_dir, exist_ok=True)
    
    if train_seed:
        torch.manual_seed(train_seed)

    device = torch.device(f"cuda:{params['gpu_id']}") if torch.cuda.is_available() \
        else torch.device("cpu")

    # torch.backends.cudnn.enabled = False ####
    # torch.backends.cudnn.benchmark = True ####

    model = create_model(**params)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if early_stopping:
        val_epoch_loss_hist = []

    best_val_epoch_loss, best_model_state = float("inf"), None
    best_model_epoch = None

    for epoch in range(num_epochs):
        if torch.cuda.is_available:
            torch.cuda.empty_cache()  # Clear GPU memory

        t_batch_losses, t_corr_losses, t_att_losses, t_prof_losses, \
            t_count_losses = run_epoch(
                train_loader, "train", model, epoch, optimizer=optimizer
        )
        train_epoch_loss = np.nanmean(t_batch_losses)
        print(
            "Train epoch %d: average loss = %6.10f" % (
                epoch + 1, train_epoch_loss
            )
        )
        _run.log_scalar(f"{trans_id}_train_epoch_loss", train_epoch_loss)
        _run.log_scalar(f"{trans_id}_train_batch_losses", t_batch_losses)
        _run.log_scalar(f"{trans_id}_train_corr_losses", t_corr_losses)
        _run.log_scalar(f"{trans_id}_train_att_losses", t_att_losses)
        _run.log_scalar(f"{trans_id}_train_prof_corr_losses", t_prof_losses)
        _run.log_scalar(f"{trans_id}_train_count_corr_losses", t_count_losses)

        v_batch_losses, v_corr_losses, v_att_losses, v_prof_losses, \
            v_count_losses = run_epoch(
                val_loader, "eval", model, epoch
        )
        val_epoch_loss = np.nanmean(v_batch_losses)
        print(
            "Valid epoch %d: average loss = %6.10f" % (
                epoch + 1, val_epoch_loss
            )
        )
        _run.log_scalar(f"{trans_id}_val_epoch_loss", val_epoch_loss)
        _run.log_scalar(f"{trans_id}_val_batch_losses", v_batch_losses)
        _run.log_scalar(f"{trans_id}_val_corr_losses", v_corr_losses)
        _run.log_scalar(f"{trans_id}_val_att_losses", v_att_losses)
        _run.log_scalar(f"{trans_id}_val_prof_corr_losses", v_prof_losses)
        _run.log_scalar(f"{trans_id}_val_count_corr_losses", v_count_losses)

        # Save trained model for the epoch
        savepath = os.path.join(
            output_dir, "model_ckpt_epoch_%d.pt" % (epoch + 1)
        )
        util.save_model(model, savepath)

        # Save the model state dict of the epoch with the best validation loss
        if val_epoch_loss < best_val_epoch_loss:
            best_val_epoch_loss = val_epoch_loss
            best_model_state = model.state_dict()
            best_model_epoch = epoch

        # If losses are both NaN, then stop
        if np.isnan(train_epoch_loss) and np.isnan(val_epoch_loss):
            break

        # Check for early stopping
        if early_stopping:
            if len(val_epoch_loss_hist) < early_stop_hist_len + 1:
                # Not enough history yet; tack on the loss
                val_epoch_loss_hist = [val_epoch_loss] + val_epoch_loss_hist
            else:
                # Tack on the new validation loss, kicking off the old one
                val_epoch_loss_hist = \
                    [val_epoch_loss] + val_epoch_loss_hist[:-1]
                best_delta = np.max(np.diff(val_epoch_loss_hist))
                if best_delta < early_stop_min_delta:
                    break  # Not improving enough

    _run.log_scalar(f"{trans_id}_best_epoch", best_model_epoch)

    # Compute evaluation metrics and log them
    for data_loader, prefix in [
        (test_summit_loader, "summit"), # (test_peak_loader, "peak"),
        # (test_genome_loader, "genomewide")
    ]:
        print("Computing test metrics, %s:" % prefix)
        # Load in the state of the epoch with the best validation loss first
        model.load_state_dict(best_model_state)
        batch_losses, corr_losses, att_losses, prof_losses, count_losses, \
            true_profs, log_pred_profs, true_counts, log_pred_counts, coords, \
            input_grads, input_seqs = run_epoch(
                data_loader, "eval", model, 0, return_data=True
        )
        _run.log_scalar(f"{trans_id}_test_{prefix}_batch_losses", batch_losses)
        _run.log_scalar(f"{trans_id}_test_{prefix}_corr_losses", corr_losses)
        _run.log_scalar(f"{trans_id}_test_{prefix}_att_losses", att_losses)
        _run.log_scalar(f"{trans_id}_test_{prefix}_prof_corr_losses", prof_losses)
        _run.log_scalar(f"{trans_id}_test_{prefix}_count_corr_losses", count_losses)

        metrics = profile_performance.compute_performance_metrics(
            true_profs, log_pred_profs, true_counts, log_pred_counts
        )
        profile_performance.log_performance_metrics(metrics, f"{trans_id}_{prefix}",  _run)


@train_ex.command
def run_training(
    peak_beds, profile_hdf5, train_chroms, val_chroms, test_chroms, trans_id
):
    train_loader = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SamplingCoordsBatcher",
        return_coords=True, chrom_set=train_chroms
    )
    val_loader = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SamplingCoordsBatcher",
        return_coords=True, chrom_set=val_chroms, peak_retention=None
        # Use the whole validation set
    )
    test_summit_loader = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SummitCenteringCoordsBatcher",
        return_coords=True, revcomp=False, chrom_set=test_chroms
    )
    test_peak_loader = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "PeakTilingCoordsBatcher",
        return_coords=True, chrom_set=test_chroms
    )
    test_genome_loader = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SamplingCoordsBatcher", return_coords=True,
        chrom_set=test_chroms
    )
    train_model(
        train_loader, val_loader, test_summit_loader, test_peak_loader,
        test_genome_loader, trans_id
    )

@train_ex.automain
def main():
    import json
    # paths_json_path = "/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/profile/config/SPI1/SPI1_training_paths.json"
    # with open(paths_json_path, "r") as f:
    #     paths_json = json.load(f)
    # peak_beds = paths_json["peak_beds"]
    # profile_hdf5 = paths_json["profile_hdf5"]
    
    # splits_json_path = "/users/amtseng/att_priors/data/processed/chrom_splits.json"
    # with open(splits_json_path, "r") as f:
    #     splits_json = json.load(f)
    # train_chroms, val_chroms, test_chroms = \
    #     splits_json["1"]["train"], splits_json["1"]["val"], \
    #     splits_json["1"]["test"]

    # run_training(
    #     peak_beds, profile_hdf5, train_chroms, val_chroms, test_chroms
    # )

    splits_json_path = "/users/amtseng/att_priors/data/processed/chrom_splits.json"
    with open(splits_json_path, "r") as f:
        splits_json = json.load(f)
    train_chroms, val_chroms, test_chroms = \
        splits_json["1"]["train"], splits_json["1"]["val"], \
        splits_json["1"]["test"]

    bed_dir = "/mnt/lab_data2/amtseng/share/austin/dnase"
    hdf5_dir = "/mnt/lab_data2/atwang/att_priors/data/processed/ENCODE_DNase/profile/labels"

    cell_types = {
        "K562": ["ENCSR000EOT"],
        "HepG2": ["ENCSR149XIL"]
    }

    for i, i_ex in cell_types.items():
        bed_paths = [os.path.join(bed_dir, f"DNase_{ex}_{i}_idr-optimal-peaks.bed.gz") for ex in i_ex]
        hdf5_path = os.path.join(hdf5_dir, f"{i}/{i}_profiles.h5")
        run_training(
            bed_paths, hdf5_path, train_chroms, val_chroms, test_chroms, f"{i}_dnase_base"
        )
