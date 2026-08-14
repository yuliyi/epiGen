## Cell-type-specific gene–peak regulatory associations

This tutorial outlines the pipeline for inferring cell-type-specific gene–peak regulatory networks using epiGen's in silico perturbation approach. The process involves calculating perturbation scores, establishing background distributions via permutation, evaluating accuracy, and visualizing specific gene cases.

### Step1: In silico perturbation

Run the following command to compute the delta scores for cell-type-specific target genes.

```
nohup python gene_peak_regulation_perturbation.py --data_dir /data_path --save_dir /model_path --ct T > perturb.log 2>&1 &
```

Parameters:

--ct: The target cell type. Choose from [T, B, NK].

### Step2: Caculate the gene-peak association score, and evaluate the AUPRC of perturbation-derived associations

Compute the final gene–peak association scores and evaluate the overall predictive performance. This script calculates the Area Under the Precision-Recall Curve (AUPRC) to quantify the accuracy of the perturbation-derived associations.

```
nohup python gene_peak_regulation_top_gene.py --data_dir /data_path --ct T > evaluation.log 2>&1 &
```

Parameters:

--ct: The target cell type. Choose from [T, B, NK].

### Step3: Visualize the results

Once the evaluations are complete, you can generate visualizations of the association scores and model performance.

Please open and run the following Jupyter Notebook: `gene_peak_regulation_top_gene_plot.ipynb`

### Case study: Regulatory links of RPL36 in T cells

To practically validate the model's outputs, this case study assesses whether high-scoring, perturbation-derived gene–peak associations align with observed RNA–ATAC correlation patterns.

Open and run the following notebook to explore the specific regulatory landscape surrounding the RPL36 gene in T cells: `gene_peak_arc_T_RPL36.ipynb`
