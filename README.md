# 💬 SentimentChatbot  
**Sentiment Analysis & Customer Engagement Dashboard (SACED)**  

---

## 🚀 Project Overview  

**SACED** is an AI-powered platform designed to enhance customer experience analysis. It performs **Aspect-Based Sentiment Analysis** on customer reviews and delivers valuable insights through an **interactive dashboard**. It also features an **AI chatbot assistant** to help interpret sentiment trends and identify **loyal customers** based on their feedback.

---

## 🎯 Key Features  

- ✅ **Aspect-Based Sentiment Analysis**  
  Extracts fine-grained sentiments from customer reviews using **GPT-4**, identifying opinions on multiple aspects like service, delivery, product quality, etc.

- ✅ **Chatbot Assistant**  
  An integrated AI chatbot allows users to ask questions about sentiment trends, customer satisfaction, and engagement patterns.

- ✅ **Real-Time Dashboard**  
  Visualizes sentiment scores, aspect distributions, and customer satisfaction trends with interactive graphs.

- ✅ **Loyal Customer Detection**  
  Automatically highlights repeat customers and those who leave consistently positive feedback.

- ✅ **CSV Upload and Export**  
  Easily upload review data in CSV format and export the processed file with sentiment annotations.

---

## 🛠️ Tech Stack  

| Layer         | Technologies Used                    |
|---------------|--------------------------------------|
| **Frontend**  | Streamlit                            |
| **Backend**   | Python (Flask or FastAPI)            |
| **AI Models** | GPT-4 via OpenAI API, LangChain      |
| **Data**      | Pandas                               |
| **Charts**    | Plotly, Matplotlib                   |

---

## 📦 Installation Guide  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/theRKworks/saced.git
cd saced
2️⃣ Create a Virtual Environment
python -m venv venv
Activate the environment:
On Windows:
venv\Scripts\activate
On Mac/Linux:
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Set Up OpenAI API Key
OPENAI_API_KEY=your_openai_api_key
5️⃣ Run the Application
streamlit run app.py
