# 🏥 Medical Disease Classification with ClinicalBERT & BioBERT

> NLP pipeline for automatic disease classification from clinical text narratives — M1 Data Science Project

---

## 📌 Overview

This project builds an end-to-end NLP pipeline that:
1. **Converts** a binary symptom dataset into realistic clinical narrative text
2. **Classifies** diseases from those narratives using two biomedical BERT models: **ClinicalBERT** and **BioBERT**
3. **Compares** both models to identify the best approach for medical text classification

---

## 🗂️ Dataset

- **Source:** Disease and Symptoms Dataset (Medical Text Classification)
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

## 🔬 Methodology

### Step 1 — Data Preprocessing
- Loading and binarizing the symptom dataset
- Mapping symptoms to clinical terminology (e.g. `shortness of breath` → `dyspnea`)
- Generating clinical narrative text using randomized sentence templates

### Step 2 — Exploratory Data Analysis
- Class distribution analysis (highly imbalanced dataset)
- Text length statistics (words & characters)
- Vocabulary analysis & Pareto principle (80% data in ~20% classes)

### Step 3 — Modeling: Two Biomedical BERT Models

| Model | HuggingFace ID | Pre-trained on |
|-------|---------------|----------------|
| **ClinicalBERT** | `emilyalsentzer/Bio_ClinicalBERT` | Clinical notes (MIMIC-III hospital records) |
| **BioBERT** | `dmis-lab/biobert-v1.1` | Scientific articles (PubMed + PMC) |

Both models used as **feature extractors** (CLS token embeddings, 768 dimensions) combined with a **Logistic Regression** classifier.

### Step 4 — Evaluation
- Metrics: **Weighted F1-score**, **Macro F1-score**, Accuracy
- Confusion matrix on top 12 classes
- Per-class performance breakdown
- Comparison: ClinicalBERT vs BioBERT

---

## Key Results

| Model | Accuracy | Weighted F1 | Macro F1 |
|-------|----------|-------------|----------|
| ClinicalBERT + LogReg | *93%* |  *88,55%* |  *92,9%* |
| BioBERT + LogReg | *88,66%* | *93%* | **86,77%* |

> Results are consistent with published literature: BioBERT ~85-90% accuracy on biomedical tasks.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.x-darkblue?logo=pandas)

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, HuggingFace Transformers
- **NLP Models:** ClinicalBERT, BioBERT
- **Classifier:** Logistic Regression (scikit-learn)
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Platform:** Google Colab

---

## How to Run

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

> **Note:** The dataset file is not included due to size. Download it from Kaggle and place it in the root folder before running.

---

## 📁 Project Structure

```
nlp-disease-classification/
├── ProjetNLP_VF.ipynb       # Main notebook (full pipeline)
├── requirements.txt          # Python dependencies
├── .gitignore               # Files excluded from Git
└── README.md                # This file
```

---

## Author

**Amani HENCHIRI** — M1 Data Science Student  

---

## License

This project is licensed under the MIT License.
# Medical Disease Classification with ClinicalBERT

> NLP pipeline for automatic disease classification from clinical text narratives — M1 Data Science Project



