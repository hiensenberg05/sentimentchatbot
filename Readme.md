# Sentiment Analysis Dashboard

This project is an **Advanced Sentiment Analysis Dashboard** built with Streamlit, leveraging NLP and AI to analyze customer feedback from CSV files. It provides interactive visualizations, word clouds, and AI-powered insights using OpenAI's GPT models via LangChain.

## Features
- **Upload CSV**: Robust CSV parsing for various formats and encodings.
- **Sentiment Analysis**: Uses TextBlob and VADER for polarity, subjectivity, and sentiment categorization.
- **Visualizations**: Interactive charts (Plotly) for sentiment distribution and polarity vs. subjectivity.
- **Word Clouds**: Generate word clouds for positive, negative, and all reviews.
- **AI Insights**: Summarize feedback and extract aspect-based sentiment using OpenAI (API key required).
- **Chatbot**: Chat with your data using a conversational AI assistant (OpenAI API key required).

## Getting Started

### 1. Clone the Repository
```powershell
git clone <your-repo-url>
cd mybot
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
- Copy `env_template.txt` to `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the App
```powershell
streamlit run app.py
```

### 5. Usage
- Upload a CSV file containing customer feedback (column name should be like `comment`, `review`, `text`, etc.).
- Explore sentiment metrics, visualizations, and word clouds.
- Enable AI features for advanced insights and chatbot (requires OpenAI API key).

## File Structure
- `app.py` - Streamlit dashboard UI and logic
- `main.py` - Data processing, sentiment analysis, AI integration
- `requirements.txt` - Python dependencies
- `env_template.txt` - Example environment variable file
- `Readme.md` - Project documentation

## Notes
- **Datasets** (CSV files) are ignored in version control for privacy and size reasons.
- **.env** file is ignored for security.
- **__pycache__/** is ignored.

## License
MIT License
