# Brand Sentiment Analysis using BERT

> *"Twitter’s Take on Your Brand: Positive or Negative?"*

---

## 🔍 Overview

This project focuses on classifying tweets about various brands into three sentiment categories:
- **Positive** (Positive)
- **Negative** (Negative)
- **Neutral** (Neutral)

I fine-tuned a pre-trained BERT model on a dataset of brand-related tweets, addressed class imbalance with augmentation, and evaluated performance using comprehensive metrics.

---

## 📁 Dataset

- Based on a dataset of **brand-related tweets** labeled as positive, negative, or neutral.
- Original dataset: [Dataset](https://www.kaggle.com/datasets/tusharpaul2001/brand-sentiment-analysis-dataset/data).
- Tweets include mentions of popular brands like Apple, Google, Android, etc.


###  BERT-based Modeling

- Used **BERT-base-uncased** from Hugging Face Transformers.
- Initially retained previous preprocessing and added **class weights**, resulting in slight F1-score improvement.
- Found that **preprocessing can degrade BERT performance**—removing it led to a **3–4% boost in accuracy and F1-scores**.

---

### 🧠 Final Model Strategy

- **Text Augmentation**  
  Applied NLP techniques using `nlpaug`:
  - Synonym Replacement
  
  Augmentation targeted underrepresented classes.


- **Performance Boost**  
  Doing **augmentation**  led to **drastic improvement** in both accuracy and per-class F1-scores.

- **Evaluation**
  - Achieved **~85% accuracy** and **84.6% F1 score** on the test set.
  - Used **macro and weighted F1-scores** for deeper insights.


---

## 📁 Folder Overview

| Folder           | Description                               |
|------------------|-------------------------------------------|
| `bert_model/`     | Contains the fine tuned BERT model       |
| `Main`            |   Contains notebook of the code        |
| `dataset`        | Contains dataset used     |
| `Streamlit`       | Implements a Streamlit-based user interface that loads and interacts with the fine-tuned BERT model saved in bert_model     |



## 📈 Results

Accuracy-85
F1score-84.6



---

## 🛠️ Setup & Usage

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets (HuggingFace)
- scikit-learn
- nlpaug
- wandb

### Installation

```bash
pip install -r requirements.txt
