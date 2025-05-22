```bash
#!/bin/sh
#SBATCH --mail-user=ben.sakdinan@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --mem=200g
#SBATCH --cpus-per-task=32
#SBATCH --account=ctb-shapiro
#SBATCH --gpus-per-node a100:2
#SBATCH --job-name=basecalling
#SBATCH --output=%x-%j.out


# sbatch --account=ctb-shapiro basecalling.sh

module load gcc/13.3
module load dorado/0.9.5

working_dir=/home/bens/projects/ctb-shapiro/bens/sequencing_pilot

dorado basecaller --min-qscore 10 -rv --kit-name SQK-RBK114-96 /home/bens/projects/ctb-shapiro/bens/software/nanopore/dorado-0.9.6-linux-x64/bin/dna_r10.4.1_e8.2_400bps_sup@v5.0.0 $working_dir/pod5_files_pooled > $working_dir/dorado.EN03.WGAtest.bam
dorado demux --emit-fastq  --output-dir $working_dir/fastq --no-classify $working_dir/dorado.EN03.WGAtest.bam

# Pool barcodes together
for i in $(seq -w 1 96); do
  barcode="barcode${i}"
  matching_files=(*_${barcode}.fastq)

  # Check if files matching the barcode exist
  if ls ${matching_files[@]} 1> /dev/null 2>&1; then
    cat "${matching_files[@]}" > "combined_${barcode}.fastq"
    echo "Combined files for ${barcode}"
  else
    echo "No files found for ${barcode}, skipping."
  fi
done

mkdir $working_dir/corrected-fastq
parallel -j 1 'dorado correct -m /home/bens/projects/ctb-shapiro/bens/software/nanopore/dorado-0.9.6-linux-x64/bin/herro-v1 $working_dir/fastq/combined_barcode{}.fastq > corrected-fastq/barcode{}.corrected.fasta' ::: {01..96} 
```
