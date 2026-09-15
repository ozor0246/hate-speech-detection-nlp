# Hate Speech Detection Using NLP

A machine learning project for detecting toxic content in English-language comments using Natural Language Processing (NLP).

The project covers the complete machine learning workflow from text preprocessing and exploratory experimentation to model tuning, evaluation, model persistence, and deployment through a Streamlit web application.

## Project Overview

Online platforms contain large volumes of user-generated content, making manual moderation difficult at scale.

This project explores a machine learning approach for automatically classifying comments into:

* **Non-Toxic**
* **Toxic**

The final system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for text representation and **Logistic Regression** for binary classification.

A simple interactive **Streamlit application** allows users to enter English text and receive a toxicity prediction together with the model's estimated toxicity probability.

## Live Demo

The deployed Streamlit application is available here:

**[Launch Hate Speech Detector](https://ozor-hate-speech-detector.streamlit.app/)**

Source code: **[GitHub Repository](https://github.com/ozor0246/hate-speech-detection-nlp)**

## Dataset

The project uses the **Jigsaw Toxic Comment Classification** dataset.

The dataset contains approximately **159,000 comments** with toxicity-related labels.

For this project, the task was simplified into a binary classification problem using the `toxic` target:

* `0` → Non-Toxic
* `1` → Toxic

The dataset is imbalanced, with substantially more non-toxic comments than toxic comments. Because of this, model performance was evaluated using metrics beyond accuracy, particularly Precision, Recall, F1 Score, PR-AUC, and ROC-AUC.

## Project Workflow

The project follows this general workflow:

```text
Raw Comments
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Decision Threshold Analysis
      ↓
Model Persistence
      ↓
Streamlit Deployment
```

## Text Preprocessing

The preprocessing pipeline includes:

* Converting text to lowercase
* Removing HTML tags
* Removing URLs
* Removing user mentions
* Removing unnecessary special characters
* Removing extra whitespace
* Tokenization using NLTK
* English stopword removal
* Word lemmatization using WordNet

The same preprocessing logic is applied during deployment to maintain consistency between training and inference.

## Models Explored

Multiple classical machine learning approaches were explored during experimentation, including:

* Multinomial Naive Bayes
* Logistic Regression

Different text representation configurations were also investigated, including variations in:

* TF-IDF vocabulary size
* Minimum document frequency
* Maximum document frequency
* N-gram ranges
* Stopword removal
* Lemmatization

Hyperparameter tuning was performed using `GridSearchCV`.

## Final Model

The final classifier uses:

**Text representation:** TF-IDF
**Classifier:** Logistic Regression

The model was selected based on its overall classification performance and ability to produce class probabilities that could be used for threshold analysis.

## Final Model Performance

The final Logistic Regression model achieved the following performance on the held-out test set:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 95.86% |
| Precision | 85.80% |
| Recall    | 68.11% |
| F1 Score  | 75.94% |
| PR-AUC    | 84.54% |
| ROC-AUC   | 96.15% |

### Decision Threshold

Several probability thresholds were evaluated to investigate the trade-off between Precision and Recall.

The best tested threshold based on F1 Score was:

```text
Threshold: 0.50
F1 Score:  0.7594
Precision: 0.8580
Recall:    0.6811
```

Therefore, the final application uses a toxicity probability threshold of **0.50**.

The relatively high Precision means that comments classified as toxic are often correctly identified. Recall is lower, indicating that some toxic comments may still be missed.

This trade-off is important when considering the model for practical moderation scenarios.

## Streamlit Application

A Streamlit web application was developed to provide an interactive interface for the trained model.

Users can:

1. Enter an English-language comment.
2. Submit the comment for analysis.
3. View the predicted class.
4. View the estimated toxicity probability.
5. See a simple interpretation of the toxicity level.
6. View additional prediction details.

The application loads the previously trained model, TF-IDF vectorizer, and selected decision threshold rather than retraining the model when the application starts.

## Project Structure

```text
hate-speech-detection/
│
├── models/
│   ├── decision_threshold.joblib
│   ├── hate_speech_detector.joblib
│   └── tfidf_vectorizer.joblib
│
├── app.py
├── hate_speech_detection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

### Model Files

`hate_speech_detector.joblib`
Stores the trained Logistic Regression classifier.

`tfidf_vectorizer.joblib`
Stores the fitted TF-IDF vectorizer used to transform text into numerical features.

`decision_threshold.joblib`
Stores the selected probability threshold used to convert model probabilities into toxic/non-toxic predictions.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate into the project:

```bash
cd hate-speech-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application should then open in your browser.

## Requirements

The deployment application primarily depends on:

```text
streamlit
scikit-learn
joblib
nltk
```

Additional libraries may be used inside the development notebook for data analysis, visualization, and experimentation.

## Language Limitation

The model was trained and evaluated primarily on **English-language comments**.

Predictions involving:

* Nigerian Pidgin
* Other languages
* Code-switched text
* Regional dialects
* Unfamiliar slang

may be less reliable.

Although Nigerian Pidgin shares vocabulary with English and the model may correctly classify some Pidgin comments, the model was not specifically trained or evaluated for Nigerian Pidgin.

Therefore, successful individual predictions should not be interpreted as validated multilingual support.

## Other Limitations

This project has several additional limitations:

* Toxic language can depend heavily on context.
* Sarcasm and indirect insults can be difficult to classify.
* New or uncommon slang may not be represented effectively by the TF-IDF vocabulary.
* The model may miss toxic comments, as reflected by its lower Recall compared with Precision.
* Predictions may contain false positives and false negatives.
* TF-IDF does not understand language context in the same way as modern transformer-based language models.

The application is therefore intended as an **educational machine learning project** and should not be used as the sole basis for real-world content moderation decisions.

## Future Improvements

Possible future improvements include:

* Training with additional toxic-comment datasets.
* Improving minority-class representation.
* Experimenting with character-level features for misspellings and obfuscated toxic words.
* Evaluating additional classification algorithms.
* Experimenting with transformer models such as BERT or DistilBERT.
* Training specifically on Nigerian Pidgin and other language varieties.
* Supporting multilingual toxicity detection.
* Investigating translation-based multilingual classification.
* Performing more extensive threshold optimization.
* Adding model monitoring and feedback collection.

For multilingual support, future versions could either use multilingual training data or translate supported languages into English before classification. Translation-based classification would require separate evaluation because translation can change slang, sarcasm, cultural context, and toxic intent.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* TF-IDF
* Logistic Regression
* Multinomial Naive Bayes
* Joblib
* Streamlit
* Matplotlib / Seaborn
* Jupyter Notebook / Google Colab

## Key Learning Outcomes

This project demonstrates practical experience with:

* Natural Language Processing
* Text cleaning and preprocessing
* Tokenization
* Lemmatization
* TF-IDF feature engineering
* Imbalanced binary classification
* Model comparison
* Hyperparameter tuning
* Precision/Recall trade-offs
* Classification threshold analysis
* Model evaluation
* Model persistence
* Streamlit application development
* End-to-end ML deployment workflow

## Author

**Ozor Abiodun Isaac**

Machine Learning / AI Developer

GitHub-REPOSITORY-URL: https://github.com/ozor0246/hate-speech-detection-nlp

GitHub: https://github.com/ozor0246

LinkedIn: https://www.linkedin.com/in/ozor-abiodun-511789229/

## Disclaimer

This project was developed for educational and portfolio purposes.

Machine learning predictions can be incorrect and should not be used as the sole basis for real-world moderation, disciplinary, employment, legal, or other consequential decisions.
