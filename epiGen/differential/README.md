## Differential accessibility analysis

This tutorial guides you through evaluating differential accessibility (DA) across diverse cell types using the chromatin accessibility profiles predicted by epiGen.

The analysis is broken down into two main steps: calculating the correlation metrics and visualizing the results.

### Calculate log-fold change and correlation

Run the `logfd_pearson.py` script to calculate the Pearson Correlation Coefficient (PCC). This script evaluates the model's performance by comparing the DA log-fold change values computed from epiGen's predicted profiles against those from the observed, ground-truth accessibility profiles.

```
python logfd_pearson.py
```

### Visualize the Results

Once the log-fold change and PCC calculations are complete, you can generate visualizations to analyze the model's accuracy.

Open and run the provided Jupyter Notebook: `logfd_pearson.ipynb`