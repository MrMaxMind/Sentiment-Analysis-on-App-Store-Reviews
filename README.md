# Sentiment Analysis on App Store Reviews

Welcome to the Sentiment Analysis on App Store Reviews project repository. This project aims to classify App Store reviews into positive, neutral, and negative sentiments using a BERT model. Below is an overview of the project, including features, code snippets, and instructions for running the code.

---

<div align="center">
  <img src="./Sentiment-Analysis.jpeg" alt="App Store Reviews" style="border:none;">
</div>

---

## Overview

This project focuses on analyzing and classifying App Store Reviews by leveraging the power of BERT, a state-of-the-art language model. The dataset comprises 12,495 reviews with various attributes, including review content, user details, and scores.

---

## Features

- **Data Cleaning**: Handling missing values in columns such as reviewCreatedVersion, replyContent, and repliedAt.
- **Data Processing**: Tokenizing reviews using BERT's tokenizer and converting them to numerical values.
- **Model Training**: Building and training a sentiment classifier using BERT.
- **Model Evaluation**: Evaluating model performance using metrics like accuracy, precision, recall, F1-score, and Confusion Matrix.

---

## Contents

- `Sentiment_Analysis.ipynb`: Jupyter notebook containing the code implementation and analysis.
- `README.md`: This file, providing an overview of the project.
- `review.csv`: Dataset used for training and testing the models.

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Sentiment-Analysis-on-App-Store-Reviews.git
   cd Sentiment-Analysis-on-App-Store-Reviews
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
3. **Run the Jupyter Notebook**:
   ```bash
   Sentiment_Analysis.ipynb

---

## Data Processing
### Data Cleaning
- Identified and handled missing values in relevant columns.
### Class Distribution
- **Positive**: Scores 4-5
- **Neutral**: Score 3
- **Negative**: Scores 1-2
### Tokenization
- **Model Used**: BERT (bert-base-cased)
- **Tokenizer**: Utilized BERT's tokenizer
- **Sequence Length**: Chose a maximum length of 160 tokens

---

## Dataset Preparation
- **Custom Dataset Class**: Created a GPReviewDataset class for handling the reviews and their sentiments.
- **Data Split**: Split the dataset into training (80%), validation (10%), and test (10%) sets.
- **DataLoader**: Used PyTorch DataLoader for batch processing.

---

## Model Training
- **Model Architecture**: Built a sentiment classifier on top of BERT with a dropout layer for regularization.
- **Optimizer**: AdamW optimizer.
- **Scheduler**: Linear scheduler with no warmup.
- **Loss Function**: CrossEntropyLoss.
## Training Loop
- **Epochs**: 1
- **Training Metrics**: Monitored training and validation accuracy and loss.
- **Model Saving**: Saved the best model based on validation accuracy.

---

## Model Evaluation
- **Test Accuracy**: Achieved a test accuracy of 75.12%.
- **Classification Report**: Evaluated precision, recall, and F1-score for each sentiment class.
- **Confusion Matrix**: Visualized model performance using a confusion matrix.

---

## Key Insights
- **Accuracy**: The model demonstrated high accuracy in classifying positive reviews, moderate accuracy for negative reviews, and struggled with neutral reviews.
- **Example Prediction**: For the review "I love completing my todos! Best app ever!!!", the model predicted a positive sentiment.

---

## Tools and Libraries
- `Pandas`: For data manipulation and analysis.
- `Matplotlib`: For creating static, animated, and interactive visualizations.
- `Seaborn`: For statistical data visualization.
- `PyTorch`: For deep learning and model building.
- `Transformers`: For using BERT and other transformer models.

---

## Contributing
- If you have suggestions or improvements, feel free to open an issue or create a pull request.

---

## Thank you for visiting! If you find this project useful, please consider starring the repository. Happy coding!

---
   
