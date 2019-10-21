tfname=$1
indir=/users/amtseng/att_priors/data/interim/ENCODE/profile/$tfname/
outdir=/users/amtseng/att_priors/data/processed/ENCODE/profile/$tfname/

# Get the set of TF/cell-line combos for the peak files
stems=$(find $indir -name *_peakints.bed.gz -exec basename {} \; | awk -F "_" '{print $1 "_" $2 "_" $3}' | sort -u)  # Each entry is like TFNAME_EXPID_CELLINE

peakoutdir=$outdir/tf_chipseq_peaks
mkdir -p $peakoutdir

for stem in $stems
do
	zcat $indir/$stem\_peakints.bed.gz | awk '$1 ~ /(chr1|chr8|chr21)/' | gzip > $peakoutdir/$stem\_holdout_peakints.bed.gz
	zcat $indir/$stem\_peakints.bed.gz | awk '$1 !~ /(chr1|chr8|chr21)/' | gzip > $peakoutdir/$stem\_train_peakints.bed.gz
done
