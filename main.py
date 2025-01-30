import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import plotly.express as px

# 🔹 Download VADER Sentiment Lexicon (only required once)
nltk.download("vader_lexicon")

# 🔹 Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# 🔹 Set OpenAI API Key (Use Environment Variable for Security)
openai_api_key = "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"
openai.api_key = openai_api_key  # Set OpenAI API key

# 🔹 Initialize GPT-4 Model
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

# 🔹 Memory for Tracking Chat History
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# =========================
# 🔹 Function: Aspect-Based Sentiment Analysis
def preprocess_data(df):
    df['Comment'] = df['Rating'].astype(str) + ', ' + df['Comments']

    df.drop(['Date', 'Comments', 'Rating','Review Title'], axis=1, inplace=True)
    return df

# =========================
def analyze_aspect_based_sentiment(review):
    """
    Uses GPT-4 to analyze sentiment based on different aspects in the review.
    Returns structured output with aspect-wise sentiment classification.
    """
    prompt = f"""
    You are an AI performing Aspect-Based Sentiment Analysis (ABSA).
    Given a customer review, extract key aspects and classify their sentiment as Positive, Neutral, or Negative.

    Example:
    Review: "The food was amazing, but the service was slow. The ambiance was great though."
    Output: 
    - Food: Positive
    - Service: Negative
    - Ambiance: Positive

    Now analyze the following review:

    Review: "{review}"

    Output:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5
    )

    return response["choices"][0]["message"]["content"].strip()

# =========================
# 🔹 Function: Process CSV File
# =========================
def process_csv(df):
    """
    Processes the DataFrame by performing aspect-based sentiment analysis and creating two output files:
    - processed_reviews.csv (with sentiment analysis results)
    - loyal_customers.csv (with a special segment of loyal customers, based on criteria)
    """
    # pre_processing function called
    df=preprocess_data(df)
    print(df.head())
    try:

        # Apply aspect-based sentiment analysis to each review
        df["Aspect-Based Sentiment"] = df["Comment"].apply(analyze_aspect_based_sentiment)

        # Save processed reviews to a CSV file
        processed_file = "processed_reviews.csv"
        df.to_csv(processed_file, index=False, encoding="utf-8")

        # Example criteria for loyal customers (this can be customized)
        loyal_customers_df = df[df["Aspect-Based Sentiment"] == "Positive"]  # Example: Loyal customers are those with positive sentiment
        loyal_customers_file = "loyal_customers.csv"
        loyal_customers_df.to_csv(loyal_customers_file, index=False, encoding='latin-1')

        return df, processed_file, loyal_customers_file

    except Exception as e:
        print(f"Error processing DataFrame: {e}")
        return None, None, None


# =========================
# 🔹 Function: Chatbot for Sentiment Analysis
# =========================
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
    Given the conversation history and the user question, provide a helpful response.
    If the question is about sentiment analysis, refer to previous discussions.
    
    Chat History: {chat_history}
    User: {question}
    """
)

# 🔹 Create Chatbot Chain
chatbot = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

def chatbot_on_sentiment_analysis(user_query):
    """
    Allows the user to chat about sentiment analysis results.

    Parameters:
    - user_query (str): User's question related to sentiment analysis.

    Returns:
    - str: Chatbot's response.
    """
    return chatbot.predict(question=user_query)



def live_dashboard(df):
    """
    Displays a live dashboard for sentiment analysis results.
    
    Parameters:
    - df (DataFrame): The modified DataFrame with aspect-based sentiment scores.
    """
    st.title("📊 Sentiment Analysis Dashboard")

    if "Aspect-Based Sentiment" not in df.columns:
        st.error("The DataFrame must contain an 'Aspect-Based Sentiment' column.")
        return

    sentiment_counts = df["Aspect-Based Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Sentiment Distribution")
    st.plotly_chart(fig)

    st.write("### Sample Data")

    st.dataframe(df.head(10))
