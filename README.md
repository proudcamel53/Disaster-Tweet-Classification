# Disaster Tweet Classification

This project focuses on classifying tweets as either disaster-related or non-disaster-related using Natural Language Processing (NLP) and machine learning techniques. The goal is to build models that can automatically identify whether a tweet is referring to a real disaster event or not.

## Project Overview

Social media platforms such as Twitter are often used during emergencies to share real-time information. However, not every tweet containing disaster-related words actually refers to a real disaster. This project applies text preprocessing, feature extraction, classical machine learning models, and deep learning models to solve this binary text classification problem.

The target labels are:

* `0` = Non-disaster tweet
* `1` = Disaster tweet

## Dataset

The project uses a disaster tweet classification dataset containing tweets and their corresponding labels.

### Training Data

The training dataset contains:

* `id`: Unique tweet ID
* `keyword`: Keyword extracted from the tweet
* `location`: Location information
* `text`: Original tweet text
* `target`: Classification label

### Dataset Summary

* Training samples: `7613`
* Test samples: `3263`
* Non-disaster tweets: `4342`
* Disaster tweets: `3271`

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* Gensim
* TensorFlow / Keras

## Project Workflow

### 1. Data Loading and Exploration

The dataset is loaded using Pandas. Basic exploratory data analysis is performed to understand:

* Dataset shape
* Missing values
* Unique values
* Target label distribution
* Keyword distribution
* Location distribution

### 2. Text Preprocessing

The tweet text is cleaned before model training. The preprocessing steps include:

* Removing HTML tags
* Converting text to lowercase
* Removing URLs
* Removing special characters
* Removing extra whitespace
* Tokenization
* Stopword removal
* Lemmatization

A new column named `clean_text` is created after preprocessing.

### 3. Train-Validation Split

The training data is split into:

* 80% training data
* 20% validation data

Stratified splitting is used to preserve the original class distribution.

### 4. Feature Extraction

Two types of text representation techniques are used:

#### TF-IDF Vectorization

TF-IDF is used for classical machine learning models.

Configuration:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
```

#### Word2Vec Embedding

Word2Vec Skip-gram is trained on the cleaned tweets and used for deep learning models.

Configuration:

```python
Word2Vec(
    vector_size=100,
    window=4,
    min_count=2,
    sg=1,
    negative=10,
    epochs=50,
    seed=42
)
```

## Models Implemented

The project implements both classical machine learning models and deep learning models.

### Classical Machine Learning Models

* Logistic Regression
* Multinomial Naive Bayes
* Random Forest Classifier

### Deep Learning Models

* Simple RNN
* Bidirectional Simple RNN
* LSTM
* Bidirectional LSTM

## Model Evaluation

The models are evaluated using:

* Accuracy
* Macro F1-score
* Classification report
* Confusion matrix
* ROC curve
* AUC score

## Validation Results

| Model                    | Validation Accuracy | Validation Macro F1-score |
| ------------------------ | ------------------: | ------------------------: |
| Logistic Regression      |              0.8030 |                    0.7990 |
| Multinomial Naive Bayes  |              0.8188 |                    0.8085 |
| Random Forest            |              0.7971 |                    0.7874 |
| Simple RNN               |              0.7919 |                    0.7771 |
| Bidirectional Simple RNN |              0.8089 |                    0.8022 |
| LSTM                     |              0.8181 |                    0.8108 |
| Bidirectional LSTM       |              0.8122 |                    0.7984 |

Based on the validation results, Multinomial Naive Bayes and LSTM achieved the strongest overall performance among the implemented models.

## Repository Structure

```text
Disaster-Tweet-Classification/
│
├── disaster tweet classification.ipynb
├── train.csv
├── test.csv
├── README.md
└── tfidf_vectorizer.pkl
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Disaster-Tweet-Classification.git
cd Disaster-Tweet-Classification
```

### 2. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn gensim tensorflow
```

### 3. Download NLTK Resources

```python
import nltk
nltk.download('all')
```

### 4. Run the Notebook

Open the notebook in Jupyter Notebook, Google Colab, or VS Code:

```bash
jupyter notebook "disaster tweet classification.ipynb"
```

Then run the cells sequentially.

## Results and Observations

The project shows that both traditional machine learning and deep learning models can perform well on disaster tweet classification. TF-IDF-based models provide strong baseline performance, while LSTM-based models are also effective for learning sequential text patterns.

Multinomial Naive Bayes performed very well among the classical models, while LSTM achieved the highest macro F1-score among the deep learning models. Random Forest showed high training accuracy but lower validation performance, indicating possible overfitting.

## Future Improvements

Possible improvements include:

* Using pre-trained embeddings such as GloVe or FastText
* Fine-tuning transformer-based models such as BERT
* Performing hyperparameter tuning
* Improving handling of noisy tweets, abbreviations, and hashtags
* Using keyword and location features along with tweet text
* Applying cross-validation for more stable evaluation

## Author

Syed Tahmidul Islam Tanmoy.
BRAC University

## License

This project is for academic and learning purposes.
