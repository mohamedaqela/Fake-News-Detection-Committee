# Ensemble AI Framework for Fake News Detection

A multi-tiered ensemble machine learning and deep learning framework designed to detect fabricated news articles. The system processes textual data through parallel pipelines—utilizing statistical machine learning, dense neural networks, and transformer-based semantic architectures—to deliver a robust classification verdict based on a majority voting mechanism.

---

## System Architecture

The core engineering approach mitigates the risk of single-model bias by deploying a three-layered prediction architecture:

1. **Statistical Machine Learning Layer:** Extracted text features using a customized `TfidfVectorizer` (optimized for the top 10,000 features with standard English stop-words removal). Classification is handled via a high-precision `Logistic Regression` baseline.
2. **Deep Learning Layer:** Implemented a continuous `Keras Sequential Neural Network` incorporating Dense and Dropout layers, optimized via the Adam algorithm and binary cross-entropy loss to extract hidden spatial patterns in vector fields.
3. **Transformer Layer:** Integrated a specialized, context-aware `BERT-tiny` sequence classification model from Hugging Face (`mrm8488/bert-tiny-finetuned-fake-news-detection`) to preserve deep semantic relations and sentence token dependencies.
4. **Ensemble & Consensus Engine:** Built a real-time web interface using `Gradio` that acts as the coordinator. It routes input strings across all three models concurrently and executes a democratic majority vote (2-out-of-3) to produce the final system status.

---

## Technical Stack

* **Core Environment:** Python (Pandas, NumPy, Re, String)
* **Statistical Modeling & Features:** Scikit-Learn (TfidfVectorizer, Train-Test-Split, Classification Metrics)
* **Deep Learning Framework:** TensorFlow / Keras (Sequential API)
* **Pre-trained Architectures:** Hugging Face Transformers
* **Interface & Deployment:** Gradio Integration

---

## Engineering Workflow

* **Data Pipeline & Centralized Normalization:** Built a strict regular expression preprocessing function to strip HTML tags, punctuation arrays, URLs, digits, and uniform all raw inputs into absolute lowercase.
* **Validation Strategy:** Dynamic performance tracking via Seaborn confusion matrices and Scikit-Learn classification reports. The models achieved an operational validation accuracy within the range of 98-99%.
* **Testing Isolation:** Isolated a discrete subset of edge-case sequences prior to text vectorization (`manual_testing.csv`) to validate pipeline integrity and ensure rigorous integration testing on unseen operational parameters.

---

## Installation and Deployment

### 1. Clone the Repository
```bash
git clone [https://github.com/mohamedaqela/Fake-News-Detection-Committee.git](https://github.com/mohamedaqela/Fake-News-Detection-Committee.git)
cd Fake-News-Detection-Committee