import streamlit as st
import pandas as pd
from main import process_csv, chatbot_on_sentiment_analysis

# Streamlit App Title
st.set_page_config(page_title="Customer Sentiment Analysis", layout="wide")
st.title("📊 Customer Sentiment Analysis Tool")

# File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file containing customer reviews", type=["csv"])

if uploaded_file:
    st.success("✅ File Uploaded Successfully!")
    
    # Save uploaded file to disk (optional, depending on your needs)
    file_path = "uploaded_data.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Read the raw contents of the file for debugging
    raw_data = uploaded_file.getvalue()
    st.write("### Raw File Content Preview:")
    st.text(raw_data[:500])  # Display first 500 characters to check the file content

    try:
        # Load the file into a DataFrame uploaded_file
        try:
           df = pd.read_csv(uploaded_file, encoding='latin-1')  # Try 'latin-1' encoding
        except UnicodeDecodeError:
           df = pd.read_csv(uploaded_file, encoding='cp1252')  # Try 'cp1252' encoding if 'latin-1' fails

        # Log the column names and first few rows for debugging
        st.write("### 📂 CSV Columns:")
        st.write(df.columns)  # Display column names

        st.write("### 📂 Data Preview:")
        st.write(df.head())  # Show first few rows

        # Ensure 'Review' column exists
        if "Comments" not in df.columns:
            st.error("❌ The CSV file must contain a column named 'comments' for sentiment analysis.")
        else:
            st.success("✅ File processed successfully!")

            # Process the CSV (aspect-based sentiment analysis)
            df, processed_file, loyal_customers_file = process_csv(df)
            print(df.head())
            if df is not None:
                # Display processed results
                st.write("### Processed Data Preview:")
                st.dataframe(df.head())

                # Display download buttons for both processed and loyal customer files
                with open(processed_file, "rb") as f:
                    st.download_button(
                        label="Download Processed CSV",
                        data=f,
                        file_name=processed_file,
                        mime="text/csv"
                    )

                with open(loyal_customers_file, "rb") as f:
                    st.download_button(
                        label="Download Loyal Customers CSV",
                        data=f,
                        file_name=loyal_customers_file,
                        mime="text/csv"
                    )

            else:
                st.error("❌ Error in processing file!")

    except Exception as e:
        st.error(f"❌ An error occurred while loading the file: {e}")

