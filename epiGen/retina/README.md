## Retina regulatory analysis

This section explores cell type-specific regulatory programs within retina spatial transcriptomics.

### 1. Visualize the spatial distribution of retinal cell types alongside the perturbation scores

Visualize spatial enrichment within the corresponding retinal layer domains, and quantitative compare perturbation scores between target cell spots and non-target spots for each cell type.

Open and run the following notebook: `deal_st_plot.ipynb`

### 2. In silico perturbation for cell type-specific cells

This pipeline illustrates regulatory effects for specific target genes.

Step1: Target Gene Perturbation

Calculate the delta scores for your target gene in a specific tissue and state.

```
nohup python gene_peak_regulation_perturbation_ST.py --ct RGCs --cuda 5 --gene_name GAP43 > perturbation_RGCs_GAP43.log 2>&1 &
```

Step 2: Calculate association scores and evaluate AUPRC

Compute the final gene–peak association scores and evaluate the Area Under the Precision-Recall Curve (AUPRC).

```
nohup python gene_peak_regulation_retina_top_gene.py --ct RGCs > RGCs.log 2>&1 &
```

Parameters for Steps 1–2:

--ct: Cell state. Choose from [RGCs, RPCs, MCs, ACs, BCs, PCs].

--gene_name: The specific gene to perturb.

--cuda: The specific GPU device ID to use.

Step 3: Visualize Results

Open and run the visualization notebook: `gene_peak_regulation_retina_top_gene_plot.ipynb`

### 3. Aggregating peaks into 10 kb genomic regions

Step 1: Calculate regional association scores and evaluate AUPRC

Aggregate nearby accessible peaks into 10 kb genomic regions, compute the region-level association scores, and evaluate the AUPRC of the perturbation-derived associations on this broader regional level.

```
nohup python gene_peak_regulation_retina_top_gene_merge.py --ct RGCs > RGCs_merge.log 2>&1 &
```

Step 2: Visualize regional results

Open and run the notebook: `gene_peak_regulation_retina_top_gene_plot_merge.ipynb`

### 4. Motif enrichment analysis

This pipeline identifies key transcription factor (TF) binding motifs by comparing target regulatory peaks against background peaks to uncover cell type-associated regulatory programs.

Step 1: Extract significant peaks
Based on the previously calculated scores, extract the peaks where the delta score exceeds the background score for specific cell state.

```
nohup python gene_peak_regulation_retina_delta_peak.py --ct RGCs > RGCs_delta_peak.log 2>&1 &
```

Step 2: Split target and background peaks
Open the following Jupyter Notebook: `gene_peak_regulation_retina_delta_peak.ipynb`

Run the first section titled *Split target peaks and background peaks*. This will format and export the data into the target and background `.bed` files required for downstream enrichment tools.

Step 3: Run HOMER motif enrichment

Perform the motif enrichment analysis using `HOMER`. Run the following command in your terminal, ensuring you replace the placeholder path with your actual local file locations:

```
findMotifsGenome.pl /path/perturbation_retina/gene_peak_links/RGCs_target_peaks.bed hg38 \
/path/perturbation_retina/gene_peak_links/RGCs_validation \
-bg /path/perturbation_retina/gene_peak_links/RGCs_background_peaks.bed -size given -p 8
```

Step 4: Identify regulatory programs

Return to the `gene_peak_regulation_retina_delta_peak.ipynb` notebook and sequentially execute the remaining sections. This final step filters the enrichment results to identify and isolate candidate cell type-associated TF-CRE-gene regulatory programs.
