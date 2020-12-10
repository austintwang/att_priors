import feature.util as feature_util
import feature.make_profile_dataset as make_profile_dataset
import feature.make_binary_dataset as make_binary_dataset
import pandas as pd
import numpy as np
import json
import h5py

def get_profile_input_func(
    files_spec_path, input_length, profile_length, reference_fasta
):
    """
    Returns a data function needed to run profile models. This data function
    will take in an N x 3 object array of coordinates, and return the
    corresponding data needed to run the model.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `input_length`: length of input sequence
        `profile_length`: length of output profiles
        `reference_fasta`: path to reference fasta
    Returns a function that takes in an N x 3 array of coordinates, and returns
    the following: the N x I x 4 one-hot encoded sequences, and the
    N x (T or T + 1 or 2T) x O x 2 profiles (perhaps with controls).
    """
    if isinstance(files_spec_path, str):
        with open(files_spec_path, "r") as f:
            files_spec = json.load(f)
    else:
        files_spec = files_spec_path

    # Maps coordinates to 1-hot encoded sequence
    coords_to_seq = feature_util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )
    
    # Maps coordinates to profiles
    coords_to_vals = make_profile_dataset.CoordsToVals(
        files_spec["profile_hdf5"], profile_length
    )
    
    def input_func(coords):
        input_seq = coords_to_seq(coords)
        profs = coords_to_vals(coords)
        return input_seq, np.swapaxes(profs, 1, 2)

    return input_func


def get_profile_trans_input_func(
    files_spec_path, input_length, profile_length, reference_fasta
):
    """
    Returns a data function needed to run profile models. This data function
    will take in an N x 3 object array of coordinates, and return the
    corresponding data needed to run the model.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `input_length`: length of input sequence
        `profile_length`: length of output profiles
        `reference_fasta`: path to reference fasta
    Returns a function that takes in an N x 3 array of coordinates, and returns
    the following: the N x I x 4 one-hot encoded sequences, and the
    N x (T or T + 1 or 2T) x O x 2 profiles (perhaps with controls).
    """
    if isinstance(files_spec_path, str):
        with open(files_spec_path, "r") as f:
            files_spec = json.load(f)
    else:
        files_spec = files_spec_path

    # Maps coordinates to 1-hot encoded sequence
    coords_to_seq = feature_util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )
    
    # Maps coordinates to profiles
    coords_to_vals = make_profile_dataset.CoordsToVals(
        files_spec["profile_hdf5"], profile_length
    )
    coords_to_vals_trans = make_profile_dataset.CoordsToVals(
        files_spec["profile_trans_hdf5"], profile_length
    )
    
    def input_func(coords):
        input_seq = coords_to_seq(coords)
        profs = coords_to_vals(coords)
        profs_trans = coords_to_vals_trans(coords)
        return input_seq, np.swapaxes(profs, 1, 2), np.swapaxes(profs_trans, 1, 2)

    return input_func
        

def get_binary_input_func(files_spec_path, input_length, reference_fasta):
    """
    Returns a data function needed to run binary models. This data function will
    take in an N-array of bin indices, and return the corresponding data needed
    to run the model.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `input_length`: length of input sequence
        `reference_fasta`: path to reference fasta
    Returns a function that takes in an N-array of bin indices, and returns the
    following: the N x I x 4 one-hot encoded sequences, the N x T array of
    output values, and the N x 3 object array of input sequence coordinates.
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)

    # Maps coordinates to 1-hot encoded sequence
    coords_to_seq = feature_util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )
    
    # Maps bin index to values
    bins_to_vals = make_binary_dataset.BinsToVals(files_spec["labels_hdf5"])
    
    def input_func(bin_inds):
        coords, output_vals = bins_to_vals(bin_inds) 
        input_seqs = coords_to_seq(coords)
        return input_seqs, output_vals, coords

    return input_func


def get_positive_profile_coords(files_spec_path, task_ind=None, chrom_set=None):
    """
    Gets the set of positive coordinates for a profile model from the files
    specs. The coordinates consist of peaks collated over all tasks.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `task_ind`: if specified, loads only the coordinates for the task with
            this index (0-indexed); by default loads for all tasks
        `chrom_set`: if given, limit the set of coordinates to these chromosomes
            only
    Returns an N x 3 array of coordinates.
    """
    if isinstance(files_spec_path, str):
        with open(files_spec_path, "r") as f:
            files_spec = json.load(f)
    else:
        files_spec = files_spec_path

    peaks = []
    peaks_beds = files_spec["peak_beds"][task_ind] if task_ind is not None \
        else files_spec["peak_beds"]
    for peaks_bed in peaks_beds:
        table = pd.read_csv(peaks_bed, sep="\t", header=None)
        if chrom_set is not None:
            table = table[table[0].isin(chrom_set)]
        peaks.append(table.values[:, :3])
    return np.concatenate(peaks)       

def query_peak(query_table, peak):
    chrom, start, end = peak[:3]
    begin_hash = int(start) // 1000
    end_hash = int(end) // 1000
    candidates = set()
    for pos_hash in range(begin_hash, end_hash + 1):
        candidates |= query_table.get((chrom, pos_hash,), set())

    intersects = []
    for c in candidates:
        start_c, end_c = c
        if start <= end_c and start_c <= end:
            intersects.append((chrom, start_c, end_c),)
            # distance = np.abs((start + end) - (start_c + end_c)) / 2
            # intersects[(chrom, start_c, end_c)] = distance

    return intersects
    # return min(intersects, key=intersects.get) if intersects else (None, None, None)

def get_profile_footprint_coords(files_spec_path, task_ind=None, footprint_ind=None, prof_size=None, region_size=None, chrom_set=None):
    """
    Gets the set of positive coordinates for a profile model from the files
    specs. The coordinates consist of peaks collated over all tasks.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `task_ind`: if specified, loads only the coordinates for the task with
            this index (0-indexed); by default loads for all tasks
        `chrom_set`: if given, limit the set of coordinates to these chromosomes
            only
    Returns an N x 3 array of coordinates.
    """
    if isinstance(files_spec_path, str):
        with open(files_spec_path, "r") as f:
            files_spec = json.load(f)
    else:
        files_spec = files_spec_path

    peaks_beds = files_spec["peak_beds"][task_ind] if task_ind is not None \
        else files_spec["peak_beds"]
    footprints_beds = files_spec["footprint_beds"][footprint_ind] if footprint_ind is not None \
        else files_spec["footprint_beds"]

    footprints_query = {}
    for footprints_bed in footprints_beds:
        table = pd.read_csv(footprints_bed, sep="\t", header=None)
        if chrom_set is not None:
            table = table[table[0].isin(chrom_set)]
        chroms = table[0]
        starts = table[1]
        ends = table[2]
        begin_hashes = starts.astype(int) // 1000
        end_hashes = ends.astype(int) // 1000
        for chrom, start, end, begin_hash, end_hash in zip(chroms, starts, ends, begin_hashes, end_hashes):
            for pos_hash in range(begin_hash, end_hash + 1):
                footprints_query.setdefault((chrom, pos_hash), set()).add((start, end),)

    peaks = []
    footprints_profile = {}
    footprints_region = {}
    for peaks_bed in peaks_beds:
        table = pd.read_csv(peaks_bed, sep="\t", header=None)
        if chrom_set is not None:
            table = table[table[0].isin(chrom_set)]
        chroms = table[0]
        starts = table[1]
        ends = table[2]
        if prof_size is None:
            prof_starts = starts
            prof_ends = ends
        else:
            center = (0.5 * (starts + ends)).astype(int)
            half_size = int(0.5 * prof_size)
            prof_starts = center - half_size
            prof_ends = center + prof_size - half_size

        if region_size is None:
            region_starts = starts
            region_ends = ends
        else:
            center = (0.5 * (starts + ends)).astype(int)
            half_size = int(0.5 * region_size)
            region_starts = center - half_size
            region_ends = center + region_size - half_size

        data = (chroms, starts, ends, prof_starts, prof_ends, region_starts, region_ends)
        for chrom, start, end, prof_start, prof_end, region_start, region_end in zip(*data):
            peak = (chrom, start, end)
            footprints_profile[peak] = query_peak(footprints_query, (chrom, prof_start, prof_end))
            footprints_region[peak] = query_peak(footprints_query, (chrom, region_start, region_end))
            peaks.append(np.array(peak))

    return np.vstack(peaks), footprints_profile, footprints_region


def get_positive_binary_bins(files_spec_path, task_ind=None, chrom_set=None):
    """
    Gets the set of positive bin indices from the files specs. This is all
    bins that contain at least one task which is positive in that bin.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `task_ind`: if specified, loads only the coordinates for the task with
            this index (0-indexed); by default loads for all tasks
        `chrom_set`: if given, limit the set bin indices to these chromosomes
            only
    Returns an N-array of bin indices.
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)

    if task_ind is None:
        # Load from the set of bin-level labels
        labels_array = np.load(files_spec["bin_labels_npy"], allow_pickle=True)
        mask = labels_array[:, 1] == 1
        if chrom_set is not None:
            chrom_mask = np.isin(labels_array[:, 0], np.array(chrom_set))
            mask = chrom_mask & mask
        return np.where(mask)[0]
    else:
        # Load from the full HDF5
        with h5py.File(files_spec["labels_hdf5"], "r") as f:
            vals = f["values"][:, task_ind]
            mask = vals == 1
            if chrom_set is not None:
                chroms = f["chrom"][:]
                chrom_mask = np.isin(chroms, np.array(chrom_set).astype(bytes))
                mask = chrom_mask & mask
        return np.where(mask)[0]
