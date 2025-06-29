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

def analyze_sentiment(df):
    """Performs sentiment analysis using TextBlob and VADER."""
    analyzer = SentimentIntensityAnalyzer()
    
    # TextBlob Analysis
    df['polarity'] = df['comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['comment'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    # VADER Analysis
    df['vader_sentiment'] = df['comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # Categorize sentiment based on a combination or a primary method (e.g., TextBlob polarity)
    def categorize_sentiment(polarity):
        if polarity > 0.05:
            return 'Positive'
        elif polarity < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_category'] = df['polarity'].apply(categorize_sentiment)
    return df


# --- Visualization Generation ---

def create_sentiment_distribution_plot(df):
    """Creates a bar chart of sentiment distribution."""
    sentiment_counts = df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', title='Sentiment Distribution',
                 color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'grey'})
    return fig

def create_polarity_vs_subjectivity_scatter(df):
    """Creates a scatter plot of polarity vs. subjectivity."""
    fig = px.scatter(
        df, x='polarity', y='subjectivity', 
        color='sentiment_category', 
        title='Sentiment Polarity vs. Subjectivity',
        hover_data={'comment': True, 'polarity': ':.2f', 'subjectivity': ':.2f'},
        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'grey'}
    )
    return fig

def generate_wordcloud_figure(df, sentiment_category):
    """Generates a word cloud for a specific sentiment category and returns a Plotly figure."""
    if sentiment_category == 'All':
        text_data = df['comment']
    else:
        text_data = df[df['sentiment_category'] == sentiment_category]['comment']

    text = ' '.join(text_data.dropna())
    if not text:
        return None
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig = px.imshow(wordcloud, title=f'{sentiment_category} Reviews Word Cloud')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
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
    """Initializes the LangChain chatbot for interactive data analysis."""
    if not api_key:
        return None

    llm = OpenAI(temperature=0.5, openai_api_key=api_key)
    
    # Create a string representation of the dataframe sample to inject into the prompt
    data_context = df.head(100).to_string()

    memory = ConversationBufferMemory(memory_key="chat_history")
    
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question", "data_context"],
        template=f"""You are an AI assistant helping a user analyze customer feedback. You have access to the first 100 rows of their data.
        Answer the user's questions based on the data provided.

        Data Context:
        {{data_context}}

        Conversation History:
        {{chat_history}}

        User's Question: {{question}}

        Your Answer:"""
    )

    chatbot_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=False
    )
    # We pass the context during the call, not here
    return chatbot_chain, data_context


# --- Main Processing Function ---

def process_uploaded_file(uploaded_file, run_ai_analysis=False, api_key=None):
    """A single function to run the entire data processing pipeline."""
    df = load_and_prepare_data(uploaded_file)
    df_analyzed = analyze_sentiment(df)
    
    if run_ai_analysis and api_key:
        df_analyzed = get_aspect_analysis(df_analyzed, api_key)
        ai_summary = get_ai_summary(df_analyzed, api_key)
        return df_analyzed, ai_summary
    
    return df_analyzed, None

