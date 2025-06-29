# 💬 Advanced Sentiment Analysis Dashboard

A powerful and interactive dashboard built with **Streamlit**, leveraging **TextBlob**, **VADER**, and **OpenAI's GPT models via LangChain** for deep sentiment and aspect-based analysis of customer feedback. Ideal for analyzing product reviews or service feedback from CSV files.

---

## 🚀 Features

- ✅ **Upload CSV**  
  Robust support for multiple CSV formats and encodings, with automatic column detection (e.g., `comment`, `review`, `text`, etc.).

- ✅ **Sentiment Analysis**  
  - **TextBlob** for polarity and subjectivity.  
  - **VADER** for compound sentiment scoring.  
  - Categorization into **Positive**, **Negative**, and **Neutral** sentiments.

- ✅ **Visual Insights**  
  - 📊 Sentiment distribution bar chart  
  - 🔬 Polarity vs. subjectivity scatter plot  
  - ☁️ Word clouds for positive, negative, and all reviews

- ✅ **AI-Powered Analysis** *(OpenAI Key required)*  
  - 🧠 Auto-generated summary of customer sentiment  
  - 🔍 Aspect-based sentiment extraction (first 10 reviews)

- ✅ **Conversational Chatbot** *(OpenAI Key required)*  
  Ask your data questions in natural language and get real-time insights powered by LangChain's conversational memory.

---

## 📦 File Structure

```
├── app.py                # Main Streamlit application (UI)
├── main.py               # Data processing, sentiment logic, OpenAI integration
├── requirements.txt      # Python dependencies
├── env_template.txt      # Template for .env (OpenAI API key)
├── .gitignore            # Excludes data, env files, and cache
└── README.md             # Project documentation
```

---

## 🛠️ Getting Started

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/hiensenberg05/sentimentchatbot.git
cd sentimentchatbot
```

### 2. 🧪 Create & Activate Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Unix or Mac
source venv/bin/activate
```

### 3. 📦 Install Requirements

```bash
pip install -r requirements.txt
```

### 4. 🔐 Set Up OpenAI API Key

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Or use the Streamlit sidebar to input your key dynamically.

### 5. 🚀 Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Upload a CSV** → App identifies the most relevant text column.
2. **NLP Analysis** → Runs TextBlob and VADER-based sentiment metrics.
3. **Visualizations** → Generates charts and word clouds.
4. **AI Summary** → OpenAI generates a concise, structured analysis.
5. **Chatbot** → LangChain-based assistant lets you query the data.

---

## 📂 Supported Dataset Format

Your CSV file must contain at least one column with customer feedback text. Supported column names include:

```
comment, review, text, feedback, content, review_text, body
```

---

## ⚠️ Notes

- Large datasets may take time to process with AI features enabled.
- `.env`, datasets, and `__pycache__` folders are ignored via `.gitignore`.

---

## 📄 License

MIT License. Feel free to use, modify, and share.

---
