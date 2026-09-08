# Medical Disease Classification with ClinicalBERT

> NLP pipeline for automatic disease classification from clinical text narratives — M1 Data Science Project

---

## Overview

This project builds an end-to-end NLP pipeline that:
1. **Converts** a binary symptom dataset into realistic clinical narrative text
2. **Classifies** diseases from those narratives using **ClinicalBERT**, a BERT model pre-trained on medical literature

The goal is to simulate a realistic clinical NLP task: given a patient's textual description of symptoms, predict the most likely disease.

---

## Dataset

- **Source:** [Disease and Symptoms Dataset](https://www.kaggle.com/) (Kaggle)
- **Size:** 6,785 patient records × 377 symptoms
- **Classes:** Multiple disease categories (multi-class classification)
- **Format:** Binary symptom matrix → converted to clinical narrative text

**Example of generated clinical text:**
```
"The patient presents with chest tightness associated with dyspnea,
accompanied by depressive symptoms, palpitations."
→ Label: panic disorder
```

---

## Methodology

### Step 1 — Data Preprocessing (`ProjetNLP_VF.ipynb`)
- Loading and binarizing the symptom dataset
- Mapping symptoms to clinical terminology (e.g. `shortness of breath` → `dyspnea`)
- Generating clinical narrative text using randomized sentence templates

### Step 2 — Exploratory Data Analysis
- Class distribution analysis (highly imbalanced dataset)
- Text length statistics (words & characters)
- Vocabulary analysis & Pareto principle (80% data in ~20% classes)
- Word frequency & TF-IDF visualization

### Step 3 — Modeling with ClinicalBERT
- Pre-trained model: `emilyalsentzer/Bio_ClinicalBERT`
- Fine-tuning on the generated clinical narratives
- Class weights to handle severe class imbalance
- Stratified train/val/test split

### Step 4 — Evaluation
- Metrics: **Weighted F1-score**, **Macro F1-score**, Accuracy
- Confusion matrix analysis
- Per-class performance breakdown

---

## Key Results
ClinicalBERT
| Metric | Score |
|--------|-------|
| Weighted F1-score | *88,55%* |
| Macro F1-score | *92,9%* |
| Accuracy | *93%* |
BioBERT
| Metric | Score |
|--------|-------|
| Weighted F1-score | *93%* |
| Macro F1-score | *86,77%* |
| Accuracy | *88,66%* |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.x-darkblue?logo=pandas)

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, HuggingFace Transformers
- **NLP Model:** ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Platform:** Google Colab

---

##  How to Run

### Option 1 — Google Colab (recommended)
Open the notebook directly in Colab and run all cells. Upload the dataset when prompted.

### Option 2 — Local

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nlp-disease-classification.git
cd nlp-disease-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook ProjetNLP_VF.ipynb
```

> Note: The dataset file (`Disease and symptoms dataset.csv`) is not included due to size. Download it from Kaggle and place it in the root folder before running.

---

## Project Structure


nlp-disease-classification/
├── ProjetNLP_VF.ipynb       # Main notebook (full pipeline)
├── requirements.txt          # Python dependencies
├── .gitignore               # Files excluded from Git
└── README.md                # This file


---

## Author

Amani HENCHIRI — M1 Data Science Student 


## License

This project is licensed under the MIT License — feel free to use and adapt it.
