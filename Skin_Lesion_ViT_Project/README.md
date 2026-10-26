# Data-Efficient Vision Transformers for Skin Lesion Classification (Limited + Imbalanced Data)

Author: Baranika V G (ID 1671212) – MAAI901 Deep Learning for Visual Data, University of Wollongong in Dubai

## What the program does
Compares model families as the amount of labelled data shrinks (10 % → 100 %):
1. **MobileNetV2** baseline (depthwise-separable convolutional architecture)
2. **ViT trained from scratch** (conventional Vision Transformer)
3. **ViT + masked-patch self-supervised pre-training** (data-efficient), with and without class-weighted loss

Hypothesis H1: with limited labels, the pre-trained ViT reaches a higher **test macro-F1** than the scratch ViT and competitive performance with MobileNetV2.
Everything (class-wise recall, confusion matrices, hypothesis check) is measured by the program –
nothing in the report/slides is typed in by hand.

## Easiest way
**Windows:** double-click `RUN_ME.bat`.  **macOS/Linux:** `sh run_me.sh`.
It installs everything, self-tests, runs the experiment, and writes the report, PDF and slides to `documents/`.
The size of the run is chosen automatically (GPU = full run, CPU = shorter run).
For real results on HAM10000: `RUN_ME.bat --dataset ham10000 --data-root data/HAM10000`.

## Manual setup (Windows / macOS / Linux)
```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Python 3.9-3.12 recommended. A GPU is optional but makes the standard run much faster
(free option: Google Colab, see `notebooks/run_in_colab.ipynb`).

## Run
| Goal | Command |
|---|---|
| **Interactive Streamlit Web App** | `streamlit run app.py` (or double-click `run_web_app.bat`) |
| Everything, auto-sized (synthetic data) | `python run_all.py` |
| Fastest smoke test | `python run_all.py --preset quick` |
| Install check only | `python selftest.py` |
| **Real study on HAM10000** | `python run_all.py --dataset ham10000 --data-root data/HAM10000` |
| Rebuild only documents | `python make_documents.py` |
| Classify one image | `python demo.py --image path/to/image.jpg` |

HAM10000: download from the Harvard Dataverse / Kaggle ("Skin Cancer MNIST: HAM10000"), accept its licence, and place
`HAM10000_metadata.csv` and all `.jpg` images (any sub-folders) under `data/HAM10000/`.

## Outputs
- `results/results.json` – every measurement (per method, label budget, seed)
- `results/figures/` – black/white figures
- `results/models/vit_ssl_best.pt` – trained model used by `demo.py`
- `documents/Project_Report_1671212.docx` and `.pdf` – full report
- `documents/Project_Presentation_1671212.pptx` – slides

## Honesty notes
- The synthetic dataset only verifies the pipeline. Its report is automatically stamped as such.
  Use the HAM10000 run for the real conclusions.
- The prototype is not a medical device.

## Layout
```
run_all.py  run_experiments.py  make_documents.py  demo.py
src/ data.py models.py train.py metrics.py figures.py
results/  documents/  notebooks/
```
