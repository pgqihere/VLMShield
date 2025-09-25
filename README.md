# VLMShield

A vision-language model shield for detecting and filtering harmful content in multimodal inputs.

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
- [Training](#training)

## Setup Instructions

### 1. Download Visualization Dataset
First, download the visualization embedding dataset from Google Drive: https://drive.google.com/drive/folders/1xn_sdBm3rzrv3lgV9Mr87Y2ZLmWAR169?usp=sharing

**Note**: Due to the large scale of the original datasets, the embedding conversion process requires substantial processing time, even though the conversion itself is computationally efficient. To facilitate quick reproduction of visualization results, we provide pre-processed embedding data in JSON format. This eliminates the lengthy data processing pipeline and allows you to directly proceed with visualization analysis.

The pre-processed files contain:
- Pre-computed MAFE embeddings for all dataset samples
- Ready-to-use data structures for immediate visualization
- Optimized format for faster loading and processing

### 2. Download CLIP Model
Download the CLIP-ViT-Large-Patch14 model from Hugging Face:
- Model URL: https://huggingface.co/openai/clip-vit-large-patch14
- Place the downloaded model in: `VLMShield_script/model/`

### 3. Create Directory Structure
Create the required directories in the project root:
```bash
mkdir Results
mkdir Results/visual_results
mkdir Results/experiment_results
mkdir MAFE_feature_visual/embedding_datasets
```
### 4. Setup Visualization Data
Place the downloaded visualization embedding dataset into: `MAFE_feature_visual/embedding_datasets/`

### 5. Environment Setup
Our code has been tested on Linux systems with NVIDIA GPUs. To ensure successful reproduction of results, please ensure you have access to a system with at least one GPU having 8GB or more VRAM.

We provide a pre-configured environment file for easy setup. Create the conda environment directly from the provided configuration:
```bash
conda env create -f VLMShield/environment.yml
conda activate vlmshield
```

## Project Structure
```markdown
├── VLMShield_script/
│   ├── model/                    # Model weights and CLIP model
│   │   └── *.pt                  # VLMShield weight files
│   ├── datasets/                 # Experimental datasets from paper
│   ├── classifier.py             # VLMShield classifier definition
│   ├── new_train.py              # Training script for VLMShield
│   ├── VLMShield.py              # Main VLMShield implementation
│   └── main.sh                   # Batch experiment runner
├── MAFE_feature_visual/
│   ├── embedding_datasets/       # Visualization embedding data
│   └── visual.py                 # Visualization script
└── Results/
    ├── visual_results/           # Visualization outputs
    └── experiment_results/       # Experimental results
```

## Experiments
### Running Visualization
To generate feature visualizations corresponding to the paper figures:
```bash
cd MAFE_feature_visual
python visual.py
```
The visualization results will be saved in `Results/visual_results/`:
- tsne_visualization.pdf - t-SNE visualization of embeddings (corresponding to `Figure 3` in the paper)
- pca_density_visualization.pdf - PCA density visualization of embeddings (corresponding to `Figure 5` in the paper)

These visualizations demonstrate the effectiveness of our embedding-based approach in distinguishing between safe and unsafe multimodal content.

### Running Experiments
To reproduce the key experimental results from our paper, run the batch evaluation script:
```bash
cd VLMShield_script
bash main.sh
```
This batch script will automatically process all datasets and generate the experimental results corresponding to:
- `Table 3`: Performance evaluation on AdvBench and VLSafe datasets
- `Table 4`: Evaluation results on MM-Vet dataset

The script performs the following operations:
- Process all datasets in the `datasets/ folder`
- Save individual results for each dataset in `Results/experiment_results/`
- Generate a `summary.json` file with aggregated statistics
    - Results for each dataset
    - Total processing time
    - Average processing time per sample

#### Understanding the Results
After running the batch script, experimental results are organized as follows:
```markdown
Results/experiment_results/
├── AdvBench_results.json
├── VLSafe_results.json
├── MM-Vet_results.json
├── ...
└── summary.json
```
Each dataset generates a separate JSON file with detailed metrics. For example, `VLSafe_results.json` contains:
```json
{
  "dataset_name": "VLSafe",
  "metrics": {
    "asr": {
      "asr": 0.016216216216216217,
      "attack_successes": 18,
      "total_unsafe_samples": 1110,
      "total_samples": 1110,
      "dataset_type": "all_unsafe"
    },
    "accuracy": {
      "accuracy": 0.9837837837837838,
      "correct_predictions": 1092,
      "total_samples": 1110
    }
  }
}
```
The `summary.json` file aggregates all experimental results and provides overall statistics:
```json
{
  "summary": {
    "timestamp": "2025-08-25T17:52:11+08:00",
    "total_datasets": 10,
    "successful_datasets": 10,
    "failed_datasets": 0,
    "total_samples_processed": 34772,
    "total_processing_time_seconds": 1182.234544754028,
    "total_wall_time_seconds": 1675.421589480,
    "average_time_per_sample_seconds": 0.033999,
    "datasets": [
      {
        "dataset_name": "JailbreakV_text",
        "metrics": {
          "asr": {
            "asr": 0.0,
            "attack_successes": 0,
            "total_unsafe_samples": 257,
            "total_samples": 257,
            "dataset_type": "all_unsafe"
          },
          "accuracy": {
            "accuracy": 1.0,
            "correct_predictions": 257,
            "total_samples": 257
          }
        }
      }
    ]
  }
}
```


## Training
To train a new VLMShield model:
```bash
cd VLMShield_script
python new_train.py --config your_config.json
```