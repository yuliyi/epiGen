## Distance-dependent and external HiChIP analysis

This tutorial focuses on validating the biological relevance of the gene–peak associations inferred by epiGen. It does so by examining genomic distance decay (how accessibility changes relate to distance from the transcription start site) and by cross-referencing predictions with external 3D chromatin interaction data (HiChIP).

### Distance-dependent analysis from TSS

First, examine the relationship between the perturbation-derived accessibility changes (ΔX) and their genomic distance to transcription start sites (TSSs).

Run the following script to perform the calculations:

```
nohup python tss_peak_perturbation.py > tss.log 2>&1 &
```

Note: The generated result files will be saved in your data directory under: `{data_dir}/perturbation/TSS/`

### Visualize TSS results

Once the distance-dependent calculations are complete, you can generate the corresponding decay plots.

Open and run the provided Jupyter Notebook: `tss_peak_perturbation.ipynb`

### HiChIP analysis

This step assesses whether the high-scoring, epiGen-derived regulatory scores are concordant with true distal enhancer–gene associations supported by external 3D chromatin interaction data.

Open and execute the analysis pipeline in this notebook: `HiChIP_distal_perturbations.ipynb`

### Visualize the results

Finally, visualize the validation metrics and concordance plots from the HiChIP analysis to include in your evaluations.

Open and run the following notebook: `HiChIP_plots.ipynb`