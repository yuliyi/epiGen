# epiGen
Genome-scale cross-modal pretraining generates chromatin accessibility and infers cis-regulatory programs from single-cell transcriptomes

We develop **epiGen**, a genome-scale cross-modal pretrained foundation model that bridges single-cell transcriptomics and epigenomics. Powered by [CellPLM](https://github.com/OmicsML/CellPLM) (a state-of-the-art Transformer-based single-cell RNA foundation model) as its core RNA encoder, epiGen is pretrained on 1.5 million paired single-cell multiomic profiles spanning 11 human tissues and induced pluripotent stem cells.

epiGen delivers two core capabilities:

* **Genome-wide cross-modal prediction:** Directly infers high-resolution chromatin accessibility profiles from single-cell RNA inputs, predicting nearly one million regulatory peaks.

* **Gene peak association inference:** Reconstructs downstream cis-regulatory landscapes to map fine-grained gene–peak regulatory associations across diverse tissue and disease states.

## Installation

Please note that the package requires Python version 3.9 or higher.

### Installation with conda environment

```
conda env create -n epiGen python=3.9
conda activate epiGen
```

### Packages

```
accelerate=0.32.1
anndata=0.10.8
numpy=1.26.3
pandas=2.3.3
python=3.9.19
torch=2.4.1
scikit-learn=1.5.1
scanpy=1.10.2
```

### Install CellPLM

Follow the installation instructions provided in the official [CellPLM](https://github.com/OmicsML/CellPLM) repository to deploy it within your epiGen environment. In addition, replace the `epiGen/cellformer.py` file with the same file located in your local `CellPLM/model` installation directory.

### Pretrained CellPLM model checkpoint

Please download the pre-trained CellPLM model checkpoints here. A model checkpoint consists of two files: a model config (`20230926_85M.config.json` file) and a torch ckpt (`20230926_85M.best.ckpt` file). The checkpoints can be found on [dropbox](https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h/ckpt/20230926_85M.best.ckpt?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0). Configuration json file is provided in `model` folder.

### epiGen model weight files

The epiGen weight for all chromosome can be acquired from our [Hugging Face](https://huggingface.co/yuliyi/epiGen).

### Distributed training setup

epiGen leverages Hugging Face's [accelerate](https://huggingface.co/docs/accelerate/index) library to efficiently scale and manage distributed multi-GPU environments. To enable parallel processing, please ensure accelerate is [installed and configured](https://huggingface.co/docs/accelerate/v1.14.0/en/basic_tutorials/install#configuration) on your system before running the model.

## Tutorials

Step-by-step tutorials for downstream experiments are provided in the repository. The table below details the specifications and location of each evaluation task:

| Tasks                                    | Description                                                                                         | Tuturial path              |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------- |
| Cross-omics prediction                   | Predicts high-resolution chromatin accessibility profiles from single-cell RNA inputs.              | `epiGen` folder              |
| Differential accessibility analysis      | Analyzes differential accessibility across diverse cell types utilizing epiGen-predicted profiles.  | `epiGen/differential` folder |
| Cell-type-specific gene–peak association | Infers cell-type-specific gene–peak regulatory associations via in sillico perturbation.            | `epiGen/gene-peak` folder    |
| Distance-dependent and HiChIP analysis   | Validates inferred gene–peak regulatory links based on genomic distance decay and HiChIP data.      | `epiGen/hichip` folder       |
| TME regulatory analysis                  | Dissects tissue- and cell-type-specific regulatory networks within tumor microenvironments.         | `epiGen/tumor` folder        |
| retina regulatory analysis               | Dissects cell-type-specific regulatory networks for retina layers.                                  | `epiGen/retina` folder       |

## Data Availability

The processed data used in this study are publicly available on Figshare: [https://doi.org/10.6084/m9.figshare.33228324](https://doi.org/10.6084/m9.figshare.33228324)

## Citation

```

```
