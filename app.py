import streamlit as st
import pandas as pd
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("college_chatbot_dataset.csv")
    return df

df = load_data()

# -------------------------------
# Train Model
# -------------------------------
# Combine all multilingual questions for training
questions = df["question_en"].tolist() + df["question_hi"].tolist() + df["question_mr"].tolist()
intents = df["intent"].tolist() * 3  # replicate intents

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Train classifier (Logistic Regression)
model = LogisticRegression()
model.fit(X, intents)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="College Chatbot", page_icon="🎓", layout="wide")

st.title("🎓 Multilingual College Chatbot")
st.write("Ask me about **Admissions, Exams, Timetable, Results, Fees** in English, Hindi, or Marathi.")

# User input
user_input = st.text_input("You:", "")

if user_input:
    # Detect language
    try:
        lang = detect(user_input)
    except:
        lang = "en"

    # Predict intent
    X_test = vectorizer.transform([user_input])
    intent = model.predict(X_test)[0]

    # Fetch answer (first matching row for intent)
    answer_row = df[df["intent"] == intent].iloc[0]

    if lang.startswith("hi"):
        bot_response = answer_row["answer_hi"]
    elif lang.startswith("mr"):
        bot_response = answer_row["answer_mr"]
    else:
        bot_response = answer_row["answer_en"]

    # Show chatbot reply
    st.markdown(f"**Chatbot ({lang.upper()}):** {bot_response}")
