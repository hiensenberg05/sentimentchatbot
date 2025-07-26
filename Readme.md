# 🤖 Advanced Sentiment Analysis Dashboard

A powerful **Streamlit-based sentiment analysis dashboard** that combines NLP techniques with AI to provide comprehensive insights into customer feedback data. Features interactive visualizations, AI-powered analysis, and a conversational chatbot.

## ✨ Key Features

### 📊 **Data Analysis**
- **Full Dataset Processing**: Analyzes entire datasets (not just samples)
- **Robust CSV Parsing**: Handles various formats, encodings, and separators
- **Smart Column Detection**: Automatically finds review/comment columns

### 🧠 **Sentiment Analysis**
- **Dual Engine**: TextBlob + VADER for comprehensive sentiment analysis
- **Multiple Metrics**: Polarity, subjectivity, and sentiment categorization
- **Real-time Processing**: Cached analysis for optimal performance

### 📈 **Visualizations**
- **Interactive Charts**: Plotly-powered sentiment distribution and scatter plots
- **Word Clouds**: Generate word clouds for positive, negative, and all reviews
- **Responsive Design**: Optimized for different screen sizes

### 🤖 **AI-Powered Features**
- **AI Insights**: Automated summary generation using OpenAI GPT models
- **Aspect Analysis**: Extract key aspects and their sentiment from reviews
- **Conversational Chatbot**: Chat with your data using natural language
- **Smart Context**: Chatbot understands full dataset characteristics

## 🚀 Quick Start

### 1. **Clone & Setup**
```bash
git clone <your-repo-url>
cd mybot
```

### 2. **Install Dependencies**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. **Configure Environment**
```bash
# Copy template and add your OpenAI API key
cp env_template.txt .env
# Edit .env and add: OPENAI_API_KEY=your_key_here
```

### 4. **Run the App**
```bash
streamlit run app.py
```

### 5. **Access Dashboard**
Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
mybot/
├── app.py              # Main Streamlit application
├── main.py             # Data processing & AI integration
├── requirements.txt    # Python dependencies
├── env_template.txt    # Environment variables template
├── Readme.md          # This documentation
├── .gitignore         # Git ignore rules
└── venv/              # Virtual environment (ignored)
```

## 🎯 Usage Guide

### **Upload Data**
- Supported formats: CSV files
- Column names: `comment`, `review`, `text`, `content`, `feedback`
- Auto-detection of text columns

### **Analysis Options**
- **Basic Analysis**: Sentiment metrics, visualizations, word clouds
- **AI Features**: Enable for advanced insights and chatbot (requires API key)
- **Performance**: Large files automatically optimized

### **Dashboard Tabs**
1. **🤖 AI Insights**: AI-generated summaries and aspect analysis
2. **📊 Visualizations**: Sentiment distribution and scatter plots
3. **☁️ Word Clouds**: Word frequency visualization by sentiment
4. **📄 Raw Data**: Complete analyzed dataset
5. **💬 Chatbot**: Interactive AI assistant for data queries

## 🔧 Technical Details

### **Performance Optimizations**
- **Caching**: All heavy computations are cached
- **Smart Sampling**: AI features use samples for speed
- **Memory Management**: Efficient conversation history handling
- **Error Handling**: Graceful handling of rate limits and timeouts

### **AI Integration**
- **OpenAI GPT Models**: For insights and chatbot
- **LangChain**: For conversation management
- **Context Management**: Rich dataset summaries for chatbot

### **Data Processing**
- **NLTK Resources**: Automatic download of required NLP resources
- **Multiple Encodings**: UTF-8, Latin1, ISO-8859-1, CP1252
- **Robust Parsing**: Handles various CSV formats

## 🛠️ Configuration

### **Environment Variables**
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### **File Size Limits**
- **Small files** (< 1MB): Full analysis
- **Large files** (> 1MB): Optimized processing with warnings
- **AI features**: Automatically sampled for performance

## 📊 Supported Data Formats

### **CSV Requirements**
- Text column with reviews/comments
- Any number of additional columns
- Various encodings supported
- Automatic separator detection

### **Example Data Structure**
```csv
comment,sentiment_category,polarity,subjectivity
"Great product!",Positive,0.8,0.6
"Not satisfied",Negative,-0.7,0.4
```

## 🔒 Security & Privacy

- **API Keys**: Stored in `.env` file (ignored by git)
- **Data Privacy**: CSV files are not uploaded to external servers
- **Local Processing**: All analysis done locally except AI features

## 🐛 Troubleshooting

### **Common Issues**
1. **Rate Limit Errors**: Wait 30 seconds and try again
2. **Large File Warnings**: Use smaller samples for AI features
3. **Encoding Issues**: App automatically tries multiple encodings

### **Performance Tips**
- Use smaller datasets for testing
- Disable word clouds for very large files
- Ask specific questions to the chatbot

## 📝 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Built with ❤️ using Streamlit, OpenAI, and LangChain**
