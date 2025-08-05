# Amazon Product Review Sentiment Classification
This project analyzes customer product reviews on Amazon to classify sentiment using multiple machine learning models, including logistic regression, random forest, SVM, and BERT. We aimed to determine whether advanced transformer-based models like BERT outperform traditional NLP approaches in handling real-world, imbalanced review data.

> Final Project for LING 227: NLP Applications  
> Focus Areas: Sentiment Analysis and Data Imbalance  

---

## Background

Online product reviews are critical to how consumers make purchasing decisions, acting as a form of digital word-of-mouth. As e-commerce continues to grow, businesses increasingly rely on sentiment analysis to understand customer satisfaction, detect product issues, and improve recommendation systems. Our goal was to evaluate how well different models—both traditional and deep learning—could classify the sentiment of Amazon product reviews. Specifically, we asked the question: "Does fine-tuning a BERT model significantly improve sentiment classification accuracy compared to traditional ML models like logistic regression, random forest, and SVM?"

---
## How It Works
### 1. Data Preprocessing
- Source: Kaggle Amazon Product Reviews
- Cleaned with `BeautifulSoup` + RegEx
- Applied TF-IDF vectorization
- Sentiment labels grouped into:
  - 1–2 stars → Negative
  - 3 stars → Neutral
  - 4–5 stars → Positive
### 2. Models Implemented
- **BERT**: Fine-tuned on review text using HuggingFace Transformers and PyTorch
- **Logistic Regression**: With/without added features like helpful votes and time difference
- **Random Forest**: Tree-based ensemble classifier
- **SVM**: One-vs-Rest classifier with linear kernel and TF-IDF features
### 3. Evaluation Metrics
- Accuracy
- Precision / Recall / F1-score (Macro & Weighted)
- Confusion Matrix
- Class balance challenges addressed with SMOTE and undersampling (tested but not final)
---

## Results Overview
| Model             | Accuracy | F1-score (macro) | Notes                                      |
|------------------|----------|------------------|--------------------------------------------|
| **BERT**         | 82%      | High (5-star only)| Strong on majority class, weak on others   |
| Logistic Reg.    | 82.5%    | Low (imbalanced) | Best baseline, skewed toward 5-star class  |
| Random Forest    | 80.2%    | Lower            | Underperformed due to class imbalance       |
| SVM              | 83%      | Weak (1–4 stars) | Perfect on 5-star, poor on underrepresented|

---

## File Structure
- `amazon_review.csv`: Cleaned dataset of Amazon product reviews with ratings and metadata  
- `Implementation_of_BERT_amazon_dataset.ipynb`: Fine-tuning and evaluation of BERT on review text using PyTorch and HuggingFace Transformers  
- `Logistic_and_Random_Forest_Regression.ipynb`: Traditional ML models (logistic regression, random forest) using TF-IDF features  
- `SVM_amazon.ipynb`: Baseline SVM model using TF-IDF vectorization for multi-class classification  
- `SVM_amazon_addedFeatures.ipynb`: SVM with added features (e.g., review time, helpful votes) and improved preprocessing  
- `LING227_finalproject.pdf`: Final project report detailing methodology, results, and analysis

---

## Key Learnings
- BERT excels at contextual understanding but struggles with skewed data
- Traditional models are faster and interpretable, but performance is limited without balancing
- Sentiment classification in real-world datasets requires more than accuracy—understanding where the models fail is crucial

---

## Future Work
- Balance dataset with real synthetic samples (not undersampling)
- Fine-tune with additional metadata (e.g., verified purchase, review length)
- Explore ensemble or hybrid models (e.g., BERT + logistic layer)

---

## Collaborators
- Jin Wuk Lee
- Charlson Kim


