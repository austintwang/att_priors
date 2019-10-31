import torch
import numpy as np
import pandas as pd
import sacred
from datetime import datetime
import feature.util as util

dataset_ex = sacred.Experiment("dataset")

@dataset_ex.config
def config():
    # When computing model output metrics, ignore outputs of this value
    output_ignore_value = -1

    # Path to reference genome FASTA
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"

    # For each input sequence in the raw data, center it and pad to this length 
    input_length = 1000

    # One-hot encoding has this depth
    input_depth = 4

    # Whether or not to perform reverse complement augmentation
    revcomp = True

    # Batch size; will be multiplied by two if reverse complementation is done
    batch_size = 128

    # Sample every X positives
    positive_stride = 1

    # Sample every X negatives
    negative_stride = 13

    # Number of workers for the data loader
    num_workers = 10

    # Dataset seed (for shuffling)
    dataset_seed = None


class CoordsToVals():
    """
    From a single gzipped BED file containing genomic coordinates and more
    columns of values for each coordinate, this creates an object that maps a
    list of coordinates (tuples) to a NumPy array of values at those coordinates.
    Arguments:
        `gzipped_bed_file`: Path to gzipped BED file containing a set of
            coordinates, with the extra columns being output values; coordinates
            are assumed to be on the positive strand
        `hastitle`: Whether or not the BED file has a header
    """
    def __init__(self, gzipped_bed_file, hastitle=False):
        self.gzipped_bed_file = gzipped_bed_file

        header = 0 if hastitle else None
        print("\tReading in BED file...", end=" ", flush=True)
        start = datetime.now()
        coord_val_table = pd.read_csv(
            gzipped_bed_file, sep="\t", header=header, compression="gzip"
        )
        print(str((datetime.now() - start).seconds) + "s")

        # Set MultiIndex of chromosome, start, and end
        print("\tSetting index...", end=" ", flush=True)
        start = datetime.now()
        coord_val_table.set_index(
            list(coord_val_table.columns[:3]), inplace=True
        )
        print(str((datetime.now() - start).seconds) + "s")

        self.coord_val_table = coord_val_table

    def _get_ndarray(self, coords):
        """
        From a partial Pandas MultiIndex or list of coordinates (tuples),
        retrieves a 2D NumPy array of corresponding values.
        """
        return self.coord_val_table.loc[coords].values

    def __call__(self, coords):
        return self._get_ndarray(coords)


class CoordsDownsampler(torch.utils.data.sampler.Sampler):
    """
    Creates a batch producer that evenly samples negatives and/or positives.
    Arguments:
        `coords`: A Pandas MultiIndex containing coordinate tuples
        `pos_coord_inds`: A NumPy array of which indices of `coords` are
            positive examples
        `neg_coord_inds`: A NumPy array of which indices of `coords` are
            negative examples
        `batch_size`: Number of samples per batch
        `revcomp`: Whether or not to add revcomp of coordinates in batch
        `shuffle_before_epoch`: Whether or not to shuffle all examples before
            each epoch
        `pos_row_fn`: A function that, given a Coordinate, returns whether or
            not it's a positive example
        `neg_stride`: Sample every `neg_stride` negatives before each epoch
        `pos_stride`: Sample every `pos_stride` positives before each epoch
    """
    def __init__(
        self, coords, pos_coord_inds, neg_coord_inds, batch_size,
        shuffle_before_epoch=False, neg_stride=1, pos_stride=1, seed=None
    ):
        self.pos_coord_inds = pos_coord_inds
        self.neg_coord_inds = neg_coord_inds
        self.batch_size = batch_size
        self.shuffle_before_epoch = shuffle_before_epoch
        self.neg_stride = neg_stride
        self.pos_stride = pos_stride
        self.neg_offset = 0  # Offset for striding
        self.pos_offset = 0  # Offset for striding

        # Sort the Pandas Index of coordinates
        print("\tSorting coordinates for downsampling...", end=" ", flush=True)
        start = datetime.now()
        self.coords = coords.sort_values()
        print(str((datetime.now() - start).seconds) + "s")

        if shuffle_before_epoch:
            self.rng = np.random.RandomState(seed)

    def _downsample_coords(self):
        """
        Creates a single list of downsampled indices, made according to the
        specified strides across the positive and negative coordinate indices.
        """
        start = datetime.now()
        ds_pos_coord_inds = self.pos_coord_inds[
            self.pos_offset::self.pos_stride
        ]
        ds_neg_coord_inds = self.neg_coord_inds[
            self.neg_offset::self.neg_stride
        ]
        ds_coord_inds = np.concatenate(
            [ds_pos_coord_inds, ds_neg_coord_inds]
        ).astype(int)
        return ds_coord_inds
    
    def __getitem__(self, index):
        return self.coords[self.ds_coord_inds[
            index * self.batch_size : (index + 1) * self.batch_size
        ]]

    def __len__(self):
        num_pos = len(
            range(self.pos_offset, len(self.pos_coord_inds), self.pos_stride)
        )
        num_neg = len(
            range(self.neg_offset, len(self.neg_coord_inds), self.neg_stride)
        )
        total_len = num_pos + num_neg
        return int(np.ceil(total_len / float(self.batch_size)))
   
    def _shuffle_downsample(self):
        """
        Downsamples the indices and shuffles them. The offsets used to
        downsample are incremented afterward.
        """
        self.ds_coord_inds = self._downsample_coords()
        self.pos_offset = (self.pos_offset + 1) % self.pos_stride
        self.neg_offset = (self.neg_offset + 1) % self.neg_stride
        if (self.shuffle_before_epoch):
            self.rng.shuffle(self.ds_coord_inds)

    def on_epoch_start(self):
        self._shuffle_downsample()


class CoordDataset(torch.utils.data.IterableDataset):
    """
    Generates single samples of a one-hot encoded sequence and value.
    Arguments:
        `coords_batcher (CoordsDownsampler): Maps indices to batches of
            coordinates
        `coords_to_seq (CoordsToSeq)`: Maps coordinates to 1-hot encoded
            sequences
        `coords_to_vals (CoordsToVals)`: Maps coordinates to values to predict
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
        Returns a batch, which is a pair of sequences and values, both 2D NumPy
        arrays. Takes in a batch of coordinates (i.e. a Pandas MultiIndex)
        """
        # Get batch of coordinates for this index
        coords_batch = self.coords_batcher[index]

        # Map this batch of coordinates to 1-hot encoded sequences
        seqs = self.coords_to_seq(coords_batch, revcomp=self.revcomp)

        # Map this batch of coordinates to the associated values
        vals = self.coords_to_vals(coords_batch)
        if self.revcomp:
            vals = np.concatenate([vals, vals])
        vals = vals.astype(np.int)

        if self.return_coords:
            coords = coords_batch.values
            if self.revcomp:
                coords = np.concatenate([coords, coords])
            return coords, seqs, vals
        else:
            return seqs, vals

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
def data_loader_from_bedfile(
    bedfile_path, batch_size, reference_fasta, input_length,
    positive_stride, negative_stride, num_workers, revcomp, dataset_seed,
    hastitle=True, shuffle=True, return_coords=False
):
    """
    From the path to a gzipped BED file containing coordinates and state labels,
    returns an IterableDataset object. If `shuffle` is True, shuffle the dataset
    before each epoch.
    """
    # Maps set of coordinates to state values, imported from a BED file
    coords_to_vals = CoordsToVals(
        bedfile_path, hastitle=hastitle
    )

    # Get the set of coordinates as a Pandas MultiIndex and pass it to the
    # coordinate batcher
    coord_val_table = coords_to_vals.coord_val_table
    coords = coord_val_table.index 

    print("\tGetting positive and negative rows...", end=" ", flush=True)
    start = datetime.now()
    val_matrix = coord_val_table.values
    pos_coord_bools = np.any(val_matrix == 1, axis=1)
    neg_coord_bools = np.all(val_matrix == 0, axis=1)
    pos_coord_inds = np.where(pos_coord_bools)[0]
    neg_coord_inds = np.where(neg_coord_bools)[0]
    print(str((datetime.now() - start).seconds) + "s")

    # For statistical purposes, compute the number of positives and negatives
    print("\tGetting positive and negative counts...", end=" ", flush=True)
    start = datetime.now()
    num_pos = len(np.where(val_matrix == 1)[0])
    num_neg = len(np.where(val_matrix == 0)[0])
    print(str((datetime.now() - start).seconds) + "s")
    print(
        "\tTotal coordinate counts by entry: %d + and %d -" % \
        (num_pos, num_neg)
    )
    print(
        "\tTotal coordinate counts by row: %d + and %d -" % \
        (len(pos_coord_inds), len(neg_coord_inds))
    )
    print(
        "\tEstimated downsampled counts by row: %d + and %d -" % \
        (
            int(len(pos_coord_inds) / positive_stride),
            int(len(neg_coord_inds) / negative_stride)
        )
    )

    coords_batcher = CoordsDownsampler(
        coords, pos_coord_inds, neg_coord_inds, batch_size,
        shuffle_before_epoch=shuffle, neg_stride=negative_stride,
        pos_stride=positive_stride, seed=dataset_seed
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

    tfname = "SPI1"

    base_path = "/users/amtseng/att_priors/data/"

    bedfile = os.path.join(
        base_path,
        "processed/ENCODE/binary/labels/{0}/{0}_holdout_labels.bed.gz".format(tfname)
    )

    print(tfname)

    loader = data_loader_from_bedfile(
        bedfile, reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    )
    loader.dataset.on_epoch_start()
    start_time = datetime.now()
    for batch in tqdm.tqdm(loader, total=len(loader.dataset)):
        data = batch
        break  # Just one batch to test
    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)
