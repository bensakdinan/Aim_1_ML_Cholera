Kraken 
```bash
#!/bin/bash

# Write header
echo -e "Sample\tVibrionaceae_percent" > "vibrio_proportions.tsv"

# Loop through each subdirectory
for dir in */; do
    report="${dir}report.txt"
    
    if [[ -f "$report" ]]; then
        # Get sample name from directory
        sample=$(basename "$dir")

        # Extract percent for Vibrionaceae at family level (taxonomic rank "F")
        vib_percent=$(grep -F "Vibrionaceae" "$report" | awk '$4 == "F" {print $1}' | head -n 1)
        vib_percent=${vib_percent:-0.00}

        # Append to output file
        echo -e "${sample}\t${vib_percent}" >> "vibrio_proportions.tsv"
    fi
done
```

Bowtie
```
#!/bin/bash

# Set input/output
input="bowtie_mapping.out"
output="alignment_summary.tsv"

echo -e "Sample\tOverall_Alignment_Rate" > "$output"

awk '
  /Running bowtie2 for/ {
    # Split line by space and take 4th field, strip trailing "..."
    sample = $4
    sub(/\.\.\.$/, "", sample)
  }
  /overall alignment rate/ {
    print sample "\t" $1
  }
' "$input" >> "$output"

```

convert sam files to bam files;
```
for sam in barcode*.sam; do
    base=$(basename "$sam" .sam)
    samtools view -bS "$sam" > "${base}.bam"
done
```
