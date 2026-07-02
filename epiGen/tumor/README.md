## TME regulatory analysis

This section explores tissue- and tumor-specific regulatory programs within multi-tissue tumor microenvironments (TME).

### 1. Tissue-specific accessibility patterns

Compare the accessibility patterns of tissue-specific top differentially accessible (DA) peaks between observed data and epiGen-predicted pseudo-bulk profiles across six normal tissues.

Open and run the following notebook: `atlas_tumor.ipynb`

### 2. In silico perturbation for tissue- and state-specific cells

This pipeline illustrates regulatory effects for specific target genes (e.g., ALCAM).

Step1: Target Gene Perturbation

Calculate the delta scores for your target gene in a specific tissue and state.

```
nohup python gene_peak_regulation_perturbation_tumor_normal.py --tis pancreas --cuda 5 --ct Tumor --gene_name ALCAM > perturbation_tumor_pancreas_ALCAM.log 2>&1 &
```

Step2: Permutation-based background scoring

Calculate the background scores via random permutations.

```
nohup python gene_peak_regulation_perturbation_permutation_tumor_normal.py --tis pancreas --cuda 4 --ct Tumor --gene_name ALCAM > perturbation_permutation_tumor_pancreas_ALCAM.log 2>&1 &
```

Step 3: Calculate association scores and evaluate AUPRC

Compute the final gene–peak association scores and evaluate the Area Under the Precision-Recall Curve (AUPRC).

```
nohup python gene_peak_regulation_tumor_normal_top_gene.py --tis pancreas --ct Tumor > pancreas_tumor.log 2>&1 &
```

Parameters for Steps 1–3:

--tis: Target tissue. Choose from [pancreas, uterus, colon, breast, skin, tongue].

--ct: Cell state. Choose from [Tumor, Normal].

--gene_name: The specific gene to perturb.

--cuda: The specific GPU device ID to use.

Step 4: Visualize Results

Open and run the visualization notebook: `gene_peak_regulation_tumor_normal_top_gene_plot.ipynb`

### 3. Aggregating peaks into 10 kb genomic regions

Step 1: Calculate regional association scores and evaluate AUPRC

Aggregate nearby accessible peaks into 10 kb genomic regions, compute the region-level association scores, and evaluate the AUPRC of the perturbation-derived associations on this broader regional level.

```
nohup python gene_peak_regulation_tumor_normal_top_gene_merge.py --tis pancreas --ct Tumor > pancreas_tumor_merge.log 2>&1 &
```

Step 2: Visualize regional results

Open and run the notebook: `gene_peak_regulation_tumor_normal_top_gene_plot_merge.ipynb`

### 4. Motif enrichment analysis

This pipeline identifies key transcription factor (TF) binding motifs by comparing target regulatory peaks against background peaks to uncover tumor-associated regulatory programs.

Step 1: Extract significant peaks
Based on the previously calculated scores, extract the peaks where the delta score exceeds the background score for specific cell state.

```
nohup python gene_peak_regulation_tumor_normal_delta_peak.py --tis pancreas --ct Tumor > pancreas_tumor_delta_peak.log 2>&1 &
```

Step 2: Split target and background peaks
Open the following Jupyter Notebook: `gene_peak_regulation_tumor_delta_peak.ipynb`

Run the first section titled *Split target peaks and background peaks*. This will format and export the data into the target and background `.bed` files required for downstream enrichment tools.

Step 3: Run HOMER motif enrichment

Perform the motif enrichment analysis using `HOMER`. Run the following command in your terminal, ensuring you replace the placeholder path with your actual local file locations:

```
findMotifsGenome.pl /path/perturbation_tumor/gene_peak_links/pancreas_target_peaks.bed hg38 \
/path/perturbation_tumor/gene_peak_links/pancreas_validation \
-bg /path/perturbation_tumor/gene_peak_links/pancreas_background_peaks.bed -size given -p 8
```

Step 4: Identify regulatory programs

Return to the `gene_peak_regulation_tumor_delta_peak.ipynb` notebook and sequentially execute the remaining sections. This final step filters the enrichment results to identify and isolate candidate tumor-associated TF-peak-gene regulatory programs.

### 5. State classification

Evaluate whether the chromatin accessibility profiles predicted by epiGen provide complementary information that improves tumor-versus-normal cell classification models.

Open and run the following notebook: `tumor_normal_classifier.ipynb`