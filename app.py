import re
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🛡️",
    layout="centered"
)


# =========================================================
# LOAD NLTK RESOURCES
# =========================================================

@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


setup_nltk()


# =========================================================
# LOAD MODEL FILES
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/hate_speech_detector.joblib"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.joblib"
    )

    threshold = joblib.load(
        "models/decision_threshold.joblib"
    )

    return model, vectorizer, threshold


try:
    model, vectorizer, threshold = load_model()

except FileNotFoundError:

    st.error(
        "Model files could not be found. "
        "Make sure all model files are inside the models folder."
    )

    st.stop()


# =========================================================
# PREPROCESSING
# =========================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text)

    # Remove HTML
    text = re.sub(
        r"<[^>]*>",
        " ",
        text
    )

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Keep letters and selected punctuation
    text = re.sub(
        r"[^a-zA-Z!?'\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_comment(text):

    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        return None, None, cleaned_text

    # Convert text to TF-IDF features
    text_vector = vectorizer.transform(
        [cleaned_text]
    )

    # Probability of toxic class
    toxic_probability = model.predict_proba(
        text_vector
    )[0][1]

    # Apply selected threshold
    prediction = int(
        toxic_probability >= threshold
    )

    return (
        prediction,
        toxic_probability,
        cleaned_text
    )


# =========================================================
# HEADER
# =========================================================

st.title("🛡️ Hate Speech Detector")


st.write(
    """
    An NLP machine learning application that analyzes English text
    and estimates whether it contains toxic language.
    """
)

st.info(
    """
    **Language Support:** This model was trained and evaluated primarily
    on standard English-language comments. Predictions for Nigerian Pidgin,
    other languages, dialects, code-switched text, or unfamiliar slang may
    be less reliable.
    """
)

st.caption(
    "Built using TF-IDF and Logistic Regression"
)

st.divider()


# =========================================================
# USER INPUT
# =========================================================

st.subheader("Analyze a Comment")

user_text = st.text_area(
    "Enter a comment below",
    height=150,
    placeholder=(
        "Example: Type a comment you would like "
        "the model to analyze..."
    )
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "Analyze Comment",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION RESULT
# =========================================================

if analyze:

    if not user_text.strip():

        st.warning(
            "Please enter a comment before running the analysis."
        )

    else:

        prediction, probability, cleaned_text = (
            predict_comment(user_text)
        )

        if prediction is None:
            st.warning(
        "The entered text does not contain enough meaningful words "
        "for the model to analyze.")
            st.stop()

        st.divider()

        st.subheader("Analysis Result")

        # -----------------------------------------
        # Main prediction
        # -----------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Potential Toxic Content Detected"
            )

        else:

            st.success(
                "✅ No Toxic Content Detected"
            )


        # -----------------------------------------
        # Probability
        # -----------------------------------------

        st.metric(
            label="Toxicity Probability",
            value=f"{probability:.1%}"
        )


        # -----------------------------------------
        # Probability progress bar
        # -----------------------------------------

        st.progress(
            min(
                max(float(probability), 0.0),
                1.0
            )
        )


        # -----------------------------------------
        # Risk interpretation
        # -----------------------------------------
        if probability < 0.30:
            st.info("Low toxicity probability.")

        elif probability < threshold:
            st.warning("Moderate toxicity probability, but below the "
                       "model's toxic classification threshold.")

        elif probability < 0.80:
            st.warning("High toxicity probability.")

        else:
            st.error("Very high toxicity probability.")



        # -----------------------------------------
        # Additional information
        # -----------------------------------------

        with st.expander(
            "View analysis details"
        ):

            st.write(
                "**Decision threshold:**",
                f"{threshold:.2f}"
            )

            st.write(
                "**Predicted class:**",
                "Toxic"
                if prediction == 1
                else "Non-Toxic"
            )

            st.write(
                "**Processed text:**"
            )

            st.code(
                cleaned_text
            )


# =========================================================
# ABOUT SECTION
# =========================================================

st.divider()

with st.expander("About this project"):

    st.write(
        """
        This project uses Natural Language Processing and
        machine learning to classify comments as toxic or
        non-toxic.

        **Model workflow:**

        Raw Comment → Text Preprocessing → TF-IDF →
        Logistic Regression → Toxicity Probability
        """
    )

    st.write(
        """
        The model was evaluated using metrics including
        precision, recall, F1-score, PR-AUC and ROC-AUC.
        """
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.caption(
    """
    ⚠️ Educational project: This model was developed using English-language
    training data and may not generalize reliably to Nigerian Pidgin, other
    languages, dialects, or unfamiliar slang. Machine learning models can
    also make incorrect predictions, so this application should not be used
    as the sole basis for real-world content moderation decisions.
    """
)