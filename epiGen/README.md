## Cross-omics prediction

This tutorial provides step-by-step instructions for predicting high-resolution chromatin accessibility profiles from single-cell RNA inputs using epiGen. You can choose to train a model from scratch, run zero-shot inference using our pretrained model, or fine-tune the pretrained model on your specific dataset.

### Data prepare

Before running the models, you need to format your RNA and ATAC datasets to match epiGen’s expected input structure. Follow the instructions in the `epiGen_data_preprocess.ipynb` notebook to process your data.

**Note**: Batch information must be stored in the `batch` column of the `obs` attribute in your AnnData objects.

Ensure your processed files are organized in your data directory ({data_dir}) using the following naming conventions:

atac file: {data_dir}/atac_data/atac_{chr}_{fold}.h5ad

rna file: {data_dir}/rna_data/rna_{chr}_{fold}.h5ad

train index file: {data_dir}/train_idx_{fold}.npy

test index file: {data_dir}/test_idx_{fold}.npy

### Training from scratch

To train a chromosome-specific prediction model from scratch, use the `epiGen_scratch.py` script.

```
nohup accelerate launch --main_process_port 29510 epiGen_scratch.py --data_dir /data_path --save_dir /model_path --chr chr1 --fold dataset_name > scratch.log 2>&1 &
```

Parameters:

--data_dir: The directory containing your preprocessed data.

--save_dir: The directory where the trained model weights will be saved.

--chr: The specific chromosome to predict (e.g., chr1 to chrY).

--fold: The identifier name for your dataset/fold.

--output_z: (Optional) Include this flag if you want to output and save the cell embeddings ({chr}_{fold}_embeds.npy).

Outputs:

Predictions: Saved to {data_dir}/prediction/{chr}_{fold}_preds.npy

Model Weights: Saved to {save_dir}/model_fold_{fold}/{chr}_decoder_{fold}.pt

### Inference without Fine-Tuning (Zero-shot)

To run direct inference using the pretrained epiGen model without any additional training on your dataset, set the `--mod` parameter to nofinetune.

**Prerequisite**: Ensure the pretrained epiGen model files are downloaded and placed in `{save_dir}/model_fold_pretrain/`.

```
nohup accelerate launch --main_process_port 29510 epiGen_finetune.py --data_dir /data_path --save_dir /model_path --chr chr1 --fold dataset_name --mod nofinetune > nofinetune.log 2>&1 &
```

Outputs:

Predictions: Saved to {data_dir}/prediction/{chr}_{fold}_preds_nofinetune.npy

### Fine-tune pretrained epiGen

To adapt the pretrained epiGen model to your specific dataset for improved accuracy, set the `--mod` parameter to finetune.

**Prerequisite**: As with inference, the pretrained epiGen model files must be located in `{save_dir}/model_fold_pretrain/`.

```
nohup accelerate launch --main_process_port 29510 epiGen_finetune.py --data_dir /data_path --save_dir /model_path --chr chr1 --fold dataset_name --ff_ratio 1.0 --new_ratio 1.0 --mod finetune > finetune.log 2>&1 &
```

Additional Parameters:

--ff_ratio: The ratio of the dataset to use during the fine-tuning training phase (e.g., 1.0 uses 100% of the training data).

--new_ratio: The proportion of new data in the mix of old and new data during fine-tuning.

Outputs:

Predictions: Saved to {data_dir}/prediction/{chr}_{fold}_preds_finetune.npy

Fine-tuned Model Weights: Saved to {save_dir}/model_fold_{fold}/{chr}_decoder_{fold}_finetune_{ff_ratio}_{new_ratio}.pt
