import pandas as pd
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud
import io
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
import streamlit as st
from functools import lru_cache
import hashlib

# Load environment variables
load_dotenv()


# --- NLTK Resource Management ---

def download_nltk_resources():
    """Checks for and downloads all required NLTK resources."""
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "sentiment/vader_lexicon": "vader_lexicon",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name}...")
            nltk.download(name)


# --- Data Loading and Preprocessing ---

def find_comment_column(df):
    """Identifies the most likely column containing user comments."""
    common_columns = ['comments', 'review', 'text', 'feedback', 'comment', 'content']
    for col in df.columns:
        if col.lower() in common_columns:
            return col
    return None
def load_and_prepare_data(uploaded_file):
    """
    Loads data from an uploaded CSV file with robust parsing and prepares it for analysis.
    Tries different encodings and separators to handle various CSV formats.
    """
    # Ensure the file pointer is at the beginning
    uploaded_file.seek(0)
    
    # List of common encodings to try
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    df = None
    last_error = None
    
    for encoding in encodings_to_try:
        try:
            # Reset file pointer for each attempt
            uploaded_file.seek(0)
            # Try reading with standard comma separator
            temp_df = pd.read_csv(uploaded_file, encoding=encoding)
            # If it works and we get columns, assign and break the loop
            if not temp_df.empty and len(temp_df.columns) > 0:
                df = temp_df
                break
        except Exception as e1:
            last_error = e1
            try:
                # If comma fails, try to let pandas infer the separator
                uploaded_file.seek(0)
                temp_df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
                if not temp_df.empty and len(temp_df.columns) > 0:
                    df = temp_df
                    break
            except Exception as e2:
                last_error = e2
                continue # Try the next encoding

    if df is None or df.empty:
        error_message = f"Failed to parse CSV file. Please ensure it is a valid, non-empty CSV. Last error: {last_error}"
        raise ValueError(error_message)

    # Find the column with reviews/comments
    comment_col = None
    possible_cols = ['comment', 'review', 'text', 'content', 'feedback', 'review_text', 'body']
    for col in possible_cols:
        if col in df.columns:
            comment_col = col
            break
    
    if comment_col is None:
        # If no standard column is found, try to find a column with string data
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.len().mean() > 20: # Heuristic for text column
                comment_col = col
                break

    if comment_col is None:
        raise ValueError("Could not automatically identify the review/comment column. Please rename it to 'comment' or 'review'.")

    # Standardize the column name to 'comment'
    df.rename(columns={comment_col: 'comment'}, inplace=True)
    
    # Drop rows where the comment is missing and ensure it's a string
    df.dropna(subset=['comment'], inplace=True)
    df['comment'] = df['comment'].astype(str)
    
    return df


# --- Sentiment Analysis ---

@st.cache_data
def analyze_sentiment(df):
    """Analyzes sentiment using both TextBlob and VADER for comprehensive results."""
    # Create a copy to avoid modifying the original dataframe
    df_analyzed = df.copy()
    
    # Initialize VADER sentiment analyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    
    # Function to categorize sentiment based on polarity
    def categorize_sentiment(polarity):
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    # Apply sentiment analysis
    sentiments = []
    polarities = []
    subjectivities = []
    vader_scores = []
    
    for text in df_analyzed['comment']:
        if pd.isna(text) or str(text).strip() == '':
            sentiments.append('Neutral')
            polarities.append(0.0)
            subjectivities.append(0.0)
            vader_scores.append(0.0)
            continue
            
        # TextBlob analysis
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # VADER analysis
        vader_scores_dict = vader_analyzer.polarity_scores(str(text))
        vader_score = vader_scores_dict['compound']
        
        # Combine both approaches (weighted average)
        combined_polarity = (polarity + vader_score) / 2
        
        sentiments.append(categorize_sentiment(combined_polarity))
        polarities.append(combined_polarity)
        subjectivities.append(subjectivity)
        vader_scores.append(vader_score)
    
    # Add results to dataframe
    df_analyzed['sentiment_category'] = sentiments
    df_analyzed['polarity'] = polarities
    df_analyzed['subjectivity'] = subjectivities
    df_analyzed['vader_score'] = vader_scores
    
    return df_analyzed


# --- Visualization Generation ---

@st.cache_data
def create_sentiment_distribution_plot(df):
    """Creates a pie chart showing the distribution of sentiments."""
    sentiment_counts = df['sentiment_category'].value_counts()
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                 title="Sentiment Distribution")
    return fig

@st.cache_data
def create_polarity_vs_subjectivity_scatter(df):
    """Creates a scatter plot of polarity vs subjectivity."""
    fig = px.scatter(df, x='polarity', y='subjectivity', color='sentiment_category',
                     title="Polarity vs Subjectivity Scatter Plot")
    return fig

@st.cache_data
def generate_wordcloud_figure(df, sentiment_category):
    """Generates a wordcloud figure for the specified sentiment category."""
    if sentiment_category == 'All':
        text_data = ' '.join(df['comment'].astype(str))
    else:
        text_data = ' '.join(df[df['sentiment_category'] == sentiment_category]['comment'].astype(str))
    
    if not text_data.strip():
        return None
    
    # Create wordcloud
    wordcloud = WordCloud(width=400, height=300, background_color='white', 
                         max_words=100, colormap='viridis').generate(text_data)
    
    # Convert to plotly figure
    fig = px.imshow(wordcloud, title=f"Word Cloud - {sentiment_category} Reviews")
    fig.update_layout(showlegend=False)
    return fig


# --- AI-Powered Analysis (LangChain & OpenAI) ---

def get_ai_summary(df, api_key):
    """Generates a high-level summary of the reviews using OpenAI."""
    if not api_key:
        return "(AI features disabled. Please provide an OpenAI API key.)"

    llm = OpenAI(temperature=0.7, openai_api_key=api_key)
    
    sample_text = ' '.join(df['comment'].dropna().sample(min(len(df), 50)))

    prompt_template = PromptTemplate(
        input_variables=["reviews"],
        template="""As a business analyst, review the following customer feedback and provide a structured summary. Your summary should include:
        1.  **Key Positive Themes:** (List 2-3 main points)
        2.  **Key Negative Themes:** (List 2-3 main points)
        3.  **Actionable Suggestions:** (Provide one concrete suggestion for improvement based on the feedback.)

        Reviews:
        {reviews}

        Analysis:"""
    )
    
    summary_chain = LLMChain(llm=llm, prompt=prompt_template)
    summary = summary_chain.run(sample_text)
    return summary

def get_aspect_analysis(df, api_key):
    """Performs aspect-based sentiment analysis on a sample of the data."""
    if not api_key:
        df['aspects'] = "(AI features disabled)"
        return df

    llm = OpenAI(temperature=0, openai_api_key=api_key)
    
    prompt_template = PromptTemplate(
        input_variables=["review"],
        template="""From the review below, extract key aspects (e.g., battery, screen, price) and their sentiment (Positive, Negative, Neutral). If an aspect isn't mentioned, ignore it. 
        Example:
        Review: 'The screen is beautiful and the price was great, but the battery life is a disappointment.'
        Analysis: 'Screen: Positive, Price: Positive, Battery: Negative'

        Review: {review}

        Analysis:"""
    )
    
    aspect_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    sample_size = min(len(df), 10)
    df['aspects'] = ''
    for index, row in df.head(sample_size).iterrows():
        try:
            aspects = aspect_chain.run(row['comment'])
            df.at[index, 'aspects'] = aspects
        except Exception:
            df.at[index, 'aspects'] = "Error in analysis"
        
    return df

def initialize_chatbot(api_key, df):
    """Initializes the LangChain chatbot for interactive data analysis with full dataset context."""
    if not api_key:
        return None

    llm = OpenAI(temperature=0.3, openai_api_key=api_key, max_tokens=500)

    # Build a rich summary of the FULL dataset
    n_rows = len(df)
    n_cols = len(df.columns)
    col_names = ', '.join(df.columns)
    positive_count = len(df[df['sentiment_category'] == 'Positive'])
    negative_count = len(df[df['sentiment_category'] == 'Negative'])
    neutral_count = len(df[df['sentiment_category'] == 'Neutral'])
    avg_polarity = df['polarity'].mean()
    avg_subjectivity = df['subjectivity'].mean()
    sample_rows = df[['comment', 'sentiment_category', 'polarity']].head(5).to_string(index=False)

    data_summary = f"""
Dataset shape: {n_rows} rows × {n_cols} columns
Columns: {col_names}

Sentiment distribution:
- Positive: {positive_count}
- Negative: {negative_count}
- Neutral: {neutral_count}

Average polarity: {avg_polarity:.2f}
Average subjectivity: {avg_subjectivity:.2f}

Sample reviews (first 5):
{sample_rows}
"""

    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=f"""You are a helpful AI assistant analyzing customer feedback data. Use the provided dataset summary to answer questions. If you don't know, say so.

DATA SUMMARY:
{data_summary}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""
    )

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    chatbot_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=False
    )
    return chatbot_chain, data_summary


# --- Main Processing Function ---

@st.cache_data
def process_uploaded_file(uploaded_file, run_ai_analysis=False, api_key=None):
    """A single function to run the entire data processing pipeline with caching for performance."""
    df = load_and_prepare_data(uploaded_file)
    df_analyzed = analyze_sentiment(df)  # FULL dataset always

    ai_summary = None
    if run_ai_analysis and api_key:
        # Only run AI aspect analysis and summary on a sample for performance
        sample_size = min(len(df_analyzed), 50)
        df_sample = df_analyzed.head(sample_size)
        # Optionally, you can run get_aspect_analysis on df_sample and merge results if needed
        ai_summary = get_ai_summary(df_sample, api_key)

    return df_analyzed, ai_summary

