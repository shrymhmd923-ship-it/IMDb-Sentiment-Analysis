# IMDb Movie Review Sentiment Analysis

A Natural Language Processing (NLP) project that uses **DistilBERT and PyTorch** to classify movie reviews as positive or negative.

The project demonstrates how Transformer-based language models can understand review text and predict its sentiment.

## Project Overview

Movie reviews contain valuable information about viewers' opinions and experiences. This project builds a sentiment classification model that analyzes review text and predicts whether the expressed opinion is positive or negative.

**Note:** This project uses the IMDb dataset available through Hugging Face. It does not scrape the live IMDb website or retrieve official movie ratings.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Model Architecture

The project uses **DistilBERT**, a lightweight Transformer model derived from BERT.

The model is adapted for binary sequence classification:

- **Input:** Movie review text
- **Tokenizer:** DistilBERT tokenizer
- **Transformer:** DistilBERT
- **Classification head:** Two output classes
- **Output:** Positive or Negative sentiment

## Dataset

The project uses a subset of the IMDb movie review dataset.

| Dataset split | Number of reviews |
|---|---:|
| Training | 1,000 |
| Validation | 200 |
| Testing | 300 |

The training and validation sets are selected from the training split, while the test subset comes from the separate test split.

## Training Results

In the initial recorded run, the model achieved:

| Metric | Result |
|---|---:|
| Training accuracy | 65.80% |
| Validation accuracy | 85.00% |
| Test subset accuracy | 82.00% |
| Validation loss | 0.3695 |

These results are from an earlier run and may differ from subsequent experiments. The test accuracy represents the selected 300-review subset, not the entire IMDb test dataset.

## Visualizations

### 1. Training and Validation Loss

This plot tracks the model's loss during training and validation. It helps assess learning progress and identify possible overfitting.

![Training and Validation Loss](training_validation_loss.png)

### 2. Training and Validation Accuracy

This plot compares training and validation accuracy across epochs, helping evaluate the model's ability to generalize to unseen reviews.

![Training and Validation Accuracy](training_validation_accuracy.png)

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd IMDb-Sentiment-Analysis
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Execute the Python script:

```bash
python sentiment_analysis.py
```

The script loads the dataset, tokenizes review text, trains the classification model, evaluates its performance, and generates training visualizations.

The first run may require an internet connection to download the dataset or pretrained model if they are not cached locally.

## Project Objectives

- Apply NLP concepts to a real-world text classification task.
- Use a pretrained Transformer model for sentiment analysis.
- Train and evaluate a neural network using PyTorch.
- Compare training and validation performance.
- Visualize model learning behavior.
- Practice reproducible machine learning workflows.

## Future Improvements

- Train on a larger subset of the IMDb dataset.
- Evaluate performance on the full test set.
- Add a confusion matrix and classification report.
- Experiment with learning rates, batch sizes, and sequence lengths.
- Build a simple web interface for custom review predictions.
- Compare DistilBERT with traditional machine learning approaches.

## Author

**Mohamed Mohsen Mohamed Soliman**

Artificial Intelligence Student | Machine Learning | Deep Learning | NLP

---

*This project was developed for educational and practical exploration of Transformer-based sentiment analysis.*
