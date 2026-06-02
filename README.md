# 🏥 Biomedical NLP for Medical Specialty Triage

> A deep learning-powered biomedical NLP pipeline that classifies patient symptom descriptions and medical transcription text into the most relevant medical specialty, enabling faster and more consistent triage support.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM%20%7C%20Transformers-orange)
![NLP](https://img.shields.io/badge/NLP-Biomedical%20Text%20Classification-green)
![Transformers](https://img.shields.io/badge/Transformers-BioBERT%20%7C%20PubMedBERT-purple)
![Healthcare AI](https://img.shields.io/badge/Healthcare%20AI-Clinical%20Text%20Triage-red)

---

## 🚑 Problem Statement

Hospitals and healthcare systems receive large volumes of free-text patient complaints, symptom descriptions, and clinical notes. Routing a patient to the right specialty often depends on manually interpreting this text, which can be slow, inconsistent, and difficult to scale in high-pressure settings.

This project addresses that problem using **biomedical text classification**. Given a patient symptom description or medical transcription, the system predicts the most relevant medical specialty, such as **Cardiovascular / Pulmonary**, **Neurology**, **Orthopedic**, **Gastroenterology**, **Urology**, and other specialty categories.

The goal is not to replace clinicians, but to demonstrate how deep learning and domain-specific NLP models can support faster triage workflows and reduce manual routing effort.

---

## 💡 Solution Overview

This repository implements an end-to-end deep learning pipeline for medical specialty prediction from clinical text.

The pipeline includes:

- Biomedical text cleaning and preprocessing
- Specialty label encoding and class filtering
- Word-level sequence modeling using LSTM
- Transformer fine-tuning using BioBERT and PubMedBERT
- Multi-class classification evaluation
- Confusion matrix and class-wise performance analysis
- Model comparison across accuracy, precision, recall, and F1-score

The project compares a traditional sequential deep learning baseline against transformer-based biomedical language models to evaluate how much domain-specific pretraining improves clinical text understanding.

---

## 🧠 Models Implemented

### 1. LSTM Baseline

A TensorFlow/Keras-based sequence model used as the baseline architecture.

**Architecture components:**

- Word-level tokenization
- Embedding layer
- Bidirectional LSTM layer
- Dropout regularization
- Dense classification layers
- Softmax output for multi-class prediction

The LSTM baseline captures sequential patterns in text but has limited ability to model long-range context and domain-specific biomedical terminology.

---

### 2. BioBERT

BioBERT is a biomedical domain-adapted BERT model trained further on biomedical corpora. It is designed to capture contextual relationships in medical and scientific language more effectively than general-purpose BERT.

**Model used:**

```text
dmis-lab/biobert-base-cased-v1.1
```

**Why it matters:**

BioBERT is well-suited for clinical and biomedical NLP because it understands terms, abbreviations, and context patterns commonly found in medical text.

---

### 3. PubMedBERT

PubMedBERT is trained from scratch on biomedical literature, giving it a biomedical-specific vocabulary and representation space.

**Model used:**

```text
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
```

**Why it matters:**

PubMedBERT is useful for biomedical classification tasks because its tokenizer and learned representations are built specifically around biomedical language rather than general-domain text.

---

## 🗂️ Dataset

This project uses publicly available biomedical text datasets from Kaggle:

1. **Medical Transcriptions Dataset**
2. **Medical Speech, Transcription and Intent Dataset**

The main classification pipeline uses medical transcription text and corresponding medical specialty labels. After preprocessing and class filtering, the task is formulated as a **multi-class medical specialty classification problem** across 12 major specialty classes.

---

## 🔄 System Architecture

```text
Raw Medical Text
        ↓
Text Cleaning & Preprocessing
        ↓
Tokenization / Encoding
        ↓
Model Training
(LSTM / BioBERT / PubMedBERT)
        ↓
Evaluation & Error Analysis
        ↓
Predicted Medical Specialty
```

### 📸 Add Workflow Diagram Here

```markdown
![System Workflow](assets/system_workflow.png)
```

Recommended visual: a clean pipeline diagram showing:

```text
Raw Data → Preprocessing → Tokenization → Model Training → Evaluation → Specialty Prediction
```

You can use the workflow diagram from the report, but a cleaner recreated version will look more professional on GitHub.

---

## 🧹 Preprocessing Pipeline

The preprocessing stage prepares noisy clinical text for deep learning models.

Key steps include:

- Removing missing or irrelevant transcription records
- Converting text to lowercase
- Removing punctuation, special characters, and unnecessary symbols
- Normalizing whitespace
- Filtering medical specialty classes with sufficient samples
- Encoding target labels using `LabelEncoder`
- Performing train-test split
- Applying stratification where supported

For the **LSTM model**, text is converted into padded integer sequences.

For **BioBERT** and **PubMedBERT**, text is encoded using Hugging Face tokenizers with fixed maximum sequence lengths.

---

## ⚙️ Training Configuration

| Model | Framework | Tokenization | Max Length | Optimizer | Learning Rate |
|---|---|---:|---:|---|---:|
| LSTM | TensorFlow / Keras | Word-level tokenizer | 200 | Adam | 1e-3 |
| BioBERT | PyTorch + Hugging Face | WordPiece tokenizer | 256 | AdamW-style Trainer | 2e-5 |
| PubMedBERT | PyTorch + Hugging Face | Biomedical tokenizer | 384 | AdamW-style Trainer | 2e-5 |

Additional setup:

- Supervised multi-class classification
- Cross-entropy loss
- Class-weighted training for transformer models
- GPU-enabled training environment
- Evaluation using classification report and confusion matrix

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| LSTM | 0.6738 | 0.67 | 0.67 | 0.67 |
| BioBERT | **0.8134** | **0.82** | **0.81** | **0.81** |
| PubMedBERT | 0.7425 | 0.75 | 0.74 | 0.70 |

BioBERT achieved the strongest overall performance, showing the advantage of biomedical domain-specific pretraining for medical specialty classification.

### Key Insights

- Transformer-based models outperform the LSTM baseline on all major metrics.
- BioBERT provides the best balance of accuracy, precision, recall, and F1-score.
- PubMedBERT remains competitive because of its biomedical vocabulary and domain-specific pretraining.
- LSTM learns useful sequence patterns but struggles with complex clinical context and overlapping specialty language.
- Most errors occur between specialties with similar symptoms or related clinical vocabulary.

---

## 📈 Visual Results

### 1. Model Performance Comparison

```markdown
![Model Performance Comparison](assets/model_performance_comparison.png)
```

Add a grouped bar chart or line chart comparing **Accuracy** and **F1-score** across LSTM, BioBERT, and PubMedBERT.

---

### 2. Training Loss Curves

```markdown
![Training Loss Curves](assets/training_loss_curves.png)
```

Add a training loss comparison showing how quickly each model converges during training.

---

### 3. Confusion Matrix

```markdown
![Confusion Matrix](assets/confusion_matrix.png)
```

Add the confusion matrix for the best-performing model, preferably BioBERT. This helps show which specialties are classified well and where misclassifications happen.

---

### 4. Class-wise Performance

```markdown
![Class-wise Performance](assets/classwise_performance.png)
```

Add a class-wise F1-score chart to show performance variation across medical specialties.

---

### 5. Class Distribution

```markdown
![Class Distribution](assets/class_distribution.png)
```

Add a bar chart showing the number of samples per specialty. This is useful because class imbalance affects classification performance.

---

## 🧪 Sample Inference

Example input:

```text
Patient reports chest pain, shortness of breath, and discomfort while walking.
```

Expected output:

```text
Predicted Specialty: Cardiovascular / Pulmonary
```

The notebook includes preprocessing and model evaluation logic that can be extended into a reusable inference function for real-time prediction.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Matplotlib
- Seaborn
- Kaggle API
- Google Colab

---

## 📁 Repository Structure

```text
.
├── DL_Project_Code_Final.ipynb
├── README.md
├── assets/
│   ├── system_workflow.png
│   ├── model_performance_comparison.png
│   ├── training_loss_curves.png
│   ├── confusion_matrix.png
│   ├── classwise_performance.png
│   └── class_distribution.png
└── .gitignore
```

Recommended files to keep out of GitHub:

- Kaggle credentials
- Raw datasets
- Downloaded zip files
- Trained model checkpoints
- Generated model folders
- Notebook checkpoints
- Temporary logs and cache files

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Open the notebook

Run the notebook in Google Colab or Jupyter Notebook:

```text
DL_Project_Code_Final.ipynb
```

### 3. Add Kaggle API credentials

The notebook uses the Kaggle API for dataset access. Upload your `kaggle.json` file when prompted.

Do not commit `kaggle.json` to GitHub.

### 4. Run the pipeline

The notebook covers:

1. Dataset download and extraction
2. Data loading and preprocessing
3. Exploratory data analysis
4. LSTM training and evaluation
5. BioBERT fine-tuning and evaluation
6. PubMedBERT fine-tuning and evaluation
7. Metric comparison and visualization
8. Confusion matrix and class-wise analysis

---

## 🔐 Recommended `.gitignore`

```gitignore
# Credentials
kaggle.json

# Datasets
*.zip
dataset1/
dataset2/
medicaltranscriptions/
medical-speech-transcription-and-intent/

# Model artifacts
*.h5
*.pkl
*.pt
*.bin
*.safetensors
*_final/
*_final.zip
results/
logs/
wandb/

# Notebook checkpoints
.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc
```

---

## 🧩 Future Improvements

- Build a Streamlit or Gradio demo for interactive medical specialty prediction.
- Add explainability using SHAP, LIME, or attention-based visualizations.
- Improve handling of class imbalance using focal loss or advanced sampling strategies.
- Experiment with longer-context clinical transformer models.
- Package preprocessing and inference into reusable Python modules.
- Add model cards and dataset cards for better reproducibility.
- Evaluate on additional real-world clinical text datasets.

---

## ⚠️ Medical AI Disclaimer

This project is a machine learning prototype for biomedical text classification. It is intended for research, experimentation, and portfolio demonstration only. It should not be used for diagnosis, treatment decisions, or clinical deployment without proper validation, expert review, privacy safeguards, and regulatory compliance.

---

## 👥 Contributors

- Dhruv Gupta
- Lakshay Sawhney
- Shivam Kumar
