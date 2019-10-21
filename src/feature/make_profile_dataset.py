import torch
import numpy as np
import pandas as pd
import sacred
from datetime import datetime
import tiledb
import feature.util as util

dataset_ex = sacred.Experiment("dataset")

@dataset_ex.config
def config():
    # Path to reference genome FASTA
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"

    # Path to chromosome sizes
    chrom_sizes = "/users/amtseng/genomes/hg38.canon.chrom.sizes"

    # The size of DNA sequences to fetch as input sequences
    input_length = 1346

    # The size of profiles to fetch for each coordinate
    profile_length = 1000

    # One-hot encoding has this depth
    input_depth = 4

    # Whether or not to perform reverse complement augmentation
    revcomp = True

    # Maximum size of jitter to the input for augmentation; set to 0 to disable
    jitter_size = 50

    # Batch size; will be multiplied by two if reverse complementation is done
    batch_size = 128

    # Sample X negatives randomly from the genome for every positive example
    negative_ratio = 1

    # Number of workers for the data loader
    num_workers = 10

    # Dataset seed (for shuffling)
    dataset_seed = None


class GenomeIntervalSampler:
    """
    Samples a random interval from the genome. The sampling is performed
    uniformly at random (i.e. longer chromosomes are more likely to be sampled
    from).
    Arguments:
        `chrom_sizes`: Path to 2-column TSV listing sizes of each chromosome
        `sample_length`: Length of sampled sequence
        `chroms_keep`: An iterable of chromosomes that specifies which
            chromosomes to keep from the sizes; sampling will only occur from
            these chromosomes
    """
    def __init__(self, chrom_sizes, sample_length, chroms_keep=None, seed=None):
        self.sample_length = sample_length
       
        # Create DataFrame of chromosome sizes
        chrom_table = pd.read_csv(
            chrom_sizes, sep="\t", header=None, names=["chrom", "size"]
        )
        if chroms_keep:
            chrom_table = chrom_table[chrom_table["chrom"].isin(chroms_keep)]
        chrom_table["size"] -= sample_length  # Cut off sizes to avoid overrun
        chrom_table["weight"] = chrom_table["size"] / chrom_table["size"].sum()
        self.chrom_table = chrom_table

        if seed:
            np.random.seed(seed)

    def sample_intervals(self, num_intervals):
        """
        Returns a 2D NumPy array of randomly sampled coordinates. Returns
        `num_intervals` intervals, uniformly randomly sampled from the genome.
        """
        chrom_sample = self.chrom_table.sample(
            n=num_intervals,
            replace=True,
            weights=self.chrom_table["weight"]
        )
        chrom_sample["start"] = \
            (np.random.rand(num_intervals) * chrom_sample["size"]).astype(int)
        chrom_sample["end"] = chrom_sample["start"] + self.sample_length

        return chrom_sample[["chrom", "start", "end"]].values.astype(object)


class CoordsToVals:
    """
    From the paths to several TileDB databases containing BigWig tracks, this
    creates an object that maps a list of coordinates to a NumPy array of
    profiles. The profiles are returned as an S x 2 x L array, where S is the
    number of provided paths, and L is the length of the returned profile (both
    plus and minus strands are given, in that order).
    Arguments:
        `tiledb_paths`: List of TileDB databases, each split by chromosome
        `center_size_to_use`: For each genomic coordinate, center it and pad it
            on both sides to this length to get the final profile; if this is
            smaller than the coordinate interval given, then the interval will
            be cut to this size by centering
    """
    def __init__(self, tiledb_paths, center_size_to_use):
        self.tiledb_paths = tiledb_paths 
        self.center_size_to_use = center_size_to_use

    def _get_profile(self, chrom, start, end, tiledb_context):
        """
        Fetches the profeile for the given coordinates, with an instantiated
        TileDB Context. Returns the profile as a NumPy array of numbers. This
        may pad or cut from the center to a specified length.
        """
        # Pad or cut
        if self.center_size_to_use:
            center = int(0.5 * (start + end))
            half_size = int(0.5 * self.center_size_to_use)
            left = center - half_size
            right = center + self.center_size_to_use - half_size
        else:
            left, right = start, end
       
        # Read from each database
        profiles = []
        for path in self.tiledb_paths:
            shard_path = "%s.%s" % (path, chrom)
            arr = tiledb.DenseArray(shard_path, mode="r", ctx=tiledb_context)
            plus = arr[left:right]["count_bigwig_plus_5p"]
            minus = np.flip(arr[left:right]["count_bigwig_minus_5p"])
            profiles.append(np.stack([plus, minus]))
        return np.stack(profiles)
        
    def _get_ndarray(self, coords):
        """
        From an iterable of coordinates, retrieves a 2D NumPy array of
        corresponding profile values. Note that all coordinate intervals need
        to be of the same length. The returned array is of shape N x S x 2 x L,
        where N is the number of coordinates given, S is the number of TileDB
        databases given at instantiation, and L is the desired length.
        """
        # To be thread-safe, create a new context
        context = tiledb.Ctx()
        return np.stack([
            self._get_profile(chrom, start, end, context) \
            for chrom, start, end in coords
        ])

    def __call__(self, coords):
        return self._get_ndarray(coords)


class CoordsBatcher(torch.utils.data.sampler.Sampler):
    """
    Creates a batch producer that batches positive coordinates and samples
    negative coordinates. Each batch will have some positives and negatives
    according to `neg_ratio`. When multiple sets of positive coordinates are
    given, the coordinates are all pooled together and drawn from uniformly.
    Arguments:
        `pos_coords_beds`: List of paths to gzipped BED files containing the
            sets of positive coordinates for various tasks
        `batch_size`: Number of samples per batch
        `neg_ratio`: Number of negatives to select for each positive example
        `jitter`: Random amount to jitter each positive coordinate example by
        `genome_sampler`: A GenomeIntervalSampler instance, which samples
            intervals randomly from the genome
        `shuffle_before_epoch`: Whether or not to shuffle all examples before
            each epoch
    """
    def __init__(
        self, pos_coords_beds, batch_size, neg_ratio, jitter, genome_sampler,
        shuffle_before_epoch=False, seed=None
    ):
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.jitter = jitter
        self.genome_sampler = genome_sampler
        self.shuffle_before_epoch = shuffle_before_epoch

        # Read in the positive coordinates and make N x 4 array, where the
        # 4th column is the identifier of the source BED
        pos_coords = []
        for i, pos_coords_bed in enumerate(pos_coords_beds):
            pos_coords_table = pd.read_csv(
                pos_coords_bed, sep="\t", header=None, compression="gzip"
            )
            coords = pos_coords_table.values.astype(object)
            coords = np.concatenate(
                [coords, np.tile(i + 1, (len(coords), 1))], axis=1
            )
            pos_coords.append(coords)
        self.pos_coords = np.concatenate(pos_coords)

        # Number of positives and negatives per batch
        self.num_pos = len(self.pos_coords)
        self.neg_per_batch = int(batch_size * neg_ratio / (neg_ratio + 1))
        self.pos_per_batch = batch_size - self.neg_per_batch

        if shuffle_before_epoch:
            self.rng = np.random.RandomState(seed)

    def __getitem__(self, index):
        """
        Fetches a full batch of positive and negative coordinates by filling the
        batch with some positive coordinates, and sampling randomly from the
        rest of the genome for the negatives. Returns a 2D NumPy array of
        coordinates, along with a parallel NumPy array of status. Status is 0
        for negatives, and [1, n] for positives, where the status is 1 plus the
        index of the coordinate BED file it came from.
        This method may also perform jittering for dataset augmentation.
        """
        pos_coords = self.pos_coords[
            index * self.pos_per_batch : (index + 1) * self.pos_per_batch
        ]

        # If specified, apply random jitter to each positive coordinate
        if self.jitter:
            jitter_vals = np.random.randint(
                -self.jitter, self.jitter + 1, size=len(pos_coords)
            )
            pos_coords[:,1] += jitter_vals
            pos_coords[:,2] += jitter_vals

        neg_coords = self.genome_sampler.sample_intervals(self.neg_per_batch)
        status = np.concatenate(
            [pos_coords[:,3], np.zeros(len(neg_coords), dtype=int)]
        )
        return np.concatenate([pos_coords[:,:3], neg_coords]), status

    def __len__(self):
        return int(np.ceil(self.num_pos / float(self.pos_per_batch)))
   
    def on_epoch_start(self):
        if self.shuffle_before_epoch:
            self.pos_coords = self.rng.permutation(self.pos_coords)


class CoordDataset(torch.utils.data.IterableDataset):
    """
    Generates single samples of a one-hot encoded sequence and value.
    Arguments:
        `coords_batcher (CoordsBatcher): Maps indices to batches of
            coordinates (split into positive and negative binding)
        `coords_to_seq (CoordsToSeq)`: Maps coordinates to 1-hot encoded
            sequences
        `coords_to_vals (CoordsToVals)`: Maps coordinates to a set of profiles
        `revcomp`: Whether or not to perform revcomp to the batch; this will
            double the batch size implicitly
        `return_coords`: If True, each batch returns the set of coordinates for
            that batch along with the 1-hot encoded sequences and values
    """
    def __init__(
        self, coords_batcher, coords_to_seq, coords_to_vals, revcomp=False,
        return_coords=False
    ):
        self.coords_batcher = coords_batcher
        self.coords_to_seq = coords_to_seq
        self.coords_to_vals = coords_to_vals
        self.revcomp = revcomp
        self.return_coords = return_coords

    def get_batch(self, index):
        """
        Returns a batch, which consists of an N x L x 4 NumPy array of 1-hot
        encoded sequence, an N x S x 2 x L NumPy array of profiles, and a 1D
        length-N NumPy array of statuses. The profile for each of the S tracks
        in `coords_to_vals_list` is returned, in the same order as in this list,
        and each task contains 2 tracks, for the plus and minus strand,
        respectively.
        """
        # Get batch of coordinates for this index
        coords, status = self.coords_batcher[index]

        # Map this batch of coordinates to 1-hot encoded sequences
        seqs = self.coords_to_seq(coords, revcomp=self.revcomp)

        # Map this batch of coordinates to the associated profiles
        profiles = self.coords_to_vals(coords)

        # If reverse complementation was done, double sizes of everything else
        if self.revcomp:
            profiles = np.concatenate(
                # To reverse complement, we must swap the strands AND the
                # directionality of each strand (i.e. we are assigning the other
                # strand to be the plus strand, but still 5' to 3')
                [profiles, np.flip(profiles, axis=(2, 3))]
            )
            status = np.concatenate([status, status])

        if self.return_coords:
            if self.revcomp:
                coords_ret = np.concatenate([coords.values, coords.values])
            return coords_ret, seqs, profiles, status
        else:
            return seqs, profiles, status

    def __iter__(self):
        """
        Returns an iterator over the batches. If the dataset iterator is called
        from multiple workers, each worker will be give a shard of the full
        range.
        """
        worker_info = torch.utils.data.get_worker_info()
        num_batches = len(self.coords_batcher)
        if worker_info is None:
            # In single-processing mode
            start, end = 0, num_batches
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shard_size = int(np.ceil(num_batches / num_workers))
            start = shard_size * worker_id
            end = min(start + shard_size, num_batches)
        return (self.get_batch(i) for i in range(start, end))

    def __len__(self):
        return len(self.coords_batcher)
    
    def on_epoch_start(self):
        """
        This should be called manually before the beginning of every epoch (i.e.
        before the iteration begins).
        """
        self.coords_batcher.on_epoch_start()


@dataset_ex.capture
def data_loader_from_beds_and_bigwigs(
    peaks_bed_paths, profile_tiledb_paths, batch_size, reference_fasta,
    chrom_sizes, input_length, profile_length, negative_ratio, num_workers,
    revcomp, jitter_size, dataset_seed, shuffle=True, return_coords=False
):
    """
    From a list of paths to gzipped BED files containing coordinates of positive
    peaks, and a list of paired paths to BigWigs containing reads mapped to each
    location in the genome, returns an IterableDataset object. Each entry in
    `profile_bigwig_paths` needs to be a pair of two paths, corresponding to
    the profile of the plus and minus strand, respectively. If `shuffle` is
    True, shuffle the dataset before each epoch.
    """
    # Maps set of coordinates to profiles
    coords_to_vals = CoordsToVals(profile_tiledb_paths, profile_length)

    # Randomly samples from genome
    genome_sampler = GenomeIntervalSampler(
        chrom_sizes, input_length, seed=dataset_seed
    )
    
    # Coordinate batcher, yielding batches of positive and negative coordinates
    coords_batcher = CoordsBatcher(
        peaks_bed_paths, batch_size, negative_ratio, jitter_size, genome_sampler,
        shuffle_before_epoch=shuffle, seed=dataset_seed
    )

    # Maps set of coordinates to 1-hot encoding, padded
    coords_to_seq = util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )

    # Dataset
    dataset = CoordDataset(
        coords_batcher, coords_to_seq, coords_to_vals, revcomp=revcomp,
        return_coords=return_coords
    )

    # Dataset loader: dataset is iterable and already returns batches
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        collate_fn=lambda x: x
    )

    return loader


data = None
loader = None
@dataset_ex.automain
def main():
    global data, loader
    import os
    import tqdm

    base_path = "/users/amtseng/att_priors/data/processed/ENCODE/profile/SPI1/"

    peaks_bed_files = [
        os.path.join(base_path, ending) for ending in [
            "tf_chipseq_peaks/SPI1_ENCSR000BGQ_GM12878_holdout_peakints.bed.gz",
            "tf_chipseq_peaks/SPI1_ENCSR000BGW_K562_holdout_peakints.bed.gz",
            "tf_chipseq_peaks/SPI1_ENCSR000BIJ_GM12891_holdout_peakints.bed.gz",
            "tf_chipseq_peaks/SPI1_ENCSR000BUW_HL-60_holdout_peakints.bed.gz"
        ]
    ]
            
    profile_bigwig_files = [
        os.path.join(base_path, ending) for ending in [
            "tf_chipseq_profiles/SPI1_ENCSR000BGQ_GM12878",
            "tf_chipseq_profiles/SPI1_ENCSR000BGW_K562",
            "tf_chipseq_profiles/SPI1_ENCSR000BIJ_GM12891",
            "tf_chipseq_profiles/SPI1_ENCSR000BUW_HL-60",
            "cont_chipseq_profiles/control_ENCSR000BGG_K562",
            "cont_chipseq_profiles/control_ENCSR000BGH_GM12878",
            "cont_chipseq_profiles/control_ENCSR000BIH_GM12891",
            "cont_chipseq_profiles/control_ENCSR000BVU_HL-60"
        ]
    ]

    loader = data_loader_from_beds_and_bigwigs(
        peaks_bed_files, profile_bigwig_files,
        reference_fasta="/users/amtseng/genomes/hg38.fasta"
    )
    loader.dataset.on_epoch_start()
    start_time = datetime.now()
    for batch in tqdm.tqdm(loader, total=len(loader.dataset)):
        data = batch
        break
    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)

    k = 2
    rc_k = int(len(data[0]) / 2) + k

    seqs, profiles, statuses = data
    
    seq, prof, status = seqs[k], profiles[k], statuses[k]
    rc_seq, rc_prof, rc_status = seqs[rc_k], profiles[rc_k], statuses[rc_k]
    
    print(util.one_hot_to_seq(seq))
    print(util.one_hot_to_seq(rc_seq))

    print(status, rc_status)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    task_ind = 3
    ax[0].plot(prof[task_ind][0])
    ax[0].plot(prof[task_ind][1])

    ax[1].plot(rc_prof[task_ind][0])
    ax[1].plot(rc_prof[task_ind][1])

    plt.show()
    
