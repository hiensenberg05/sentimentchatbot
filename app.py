import streamlit as st
import pandas as pd
from main import (
    download_nltk_resources,
    process_uploaded_file,
    create_sentiment_distribution_plot,
    create_polarity_vs_subjectivity_scatter,
    generate_wordcloud_figure,
    initialize_chatbot # Import the new chatbot function
)
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration and Setup ---
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# --- Main Application Logic ---
def main():
    """The main function that runs the Streamlit application."""
    
    # Step 1: Ensure NLTK resources are available (runs only once)
    with st.spinner('Checking for required NLP resources...'):
        download_nltk_resources()

    st.title("Advanced Sentiment Analysis Dashboard")

    # --- Sidebar for File Upload and Options ---
    with st.sidebar:
        st.header("Dashboard Controls")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        st.header("Analysis Options")
        show_wordclouds = st.checkbox("Generate Word Clouds", value=True)
        show_detailed_plots = st.checkbox("Show Detailed Analysis Plots", value=True)

        st.header("🤖 AI-Powered Analysis")
        
        # Automatically enable AI if key exists, but allow user to disable it.
        api_key_env = os.getenv("OPENAI_API_KEY")
        ai_enabled_by_default = bool(api_key_env)
        
        enable_ai_analysis = st.checkbox("Enable AI Insights", value=ai_enabled_by_default)
        
        api_key_to_use = None
        if enable_ai_analysis:
            if api_key_env:
                st.success("✓ OpenAI API key loaded from .env")
                api_key_to_use = api_key_env
            else:
                api_key_to_use = st.text_input(
                    "Enter your OpenAI API Key", 
                    type="password",
                    help="You can set this permanently in a .env file."
                )
                if not api_key_to_use:
                    st.warning("Please enter your OpenAI API key to use AI features.")


    # --- Main Panel for Displaying Results ---
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        try:
            # Determine if AI analysis should run
            run_ai = enable_ai_analysis and api_key_to_use
            spinner_text = 'Running advanced AI analysis...' if run_ai else 'Analyzing sentiments...'

            # Process the file
            with st.spinner(spinner_text):
                df_processed, ai_summary = process_uploaded_file(uploaded_file, run_ai_analysis=run_ai, api_key=api_key_to_use)
            
            st.success("Analysis complete!")

            # --- Display Key Metrics ---
            st.header("Analysis Overview")
            col1, col2, col3 = st.columns(3)
            positive_count = df_processed[df_processed['sentiment_category'] == 'Positive'].shape[0]
            negative_count = df_processed[df_processed['sentiment_category'] == 'Negative'].shape[0]
            neutral_count = df_processed[df_processed['sentiment_category'] == 'Neutral'].shape[0]
            
            col1.metric("Positive Reviews", f"{positive_count}")
            col2.metric("Negative Reviews", f"{negative_count}")
            col3.metric("Neutral Reviews", f"{neutral_count}")

            # Initialize chatbot if AI is enabled
            if run_ai:
                if 'chatbot_chain' not in st.session_state:
                    st.session_state.chatbot_chain, st.session_state.data_context = initialize_chatbot(api_key_to_use, df_processed)
            
            # --- Display Tabs for Different Views ---
            tabs_list = ["📊 Visualizations", "☁️ Word Clouds", "📄 Raw Data"]
            if run_ai:
                tabs_list.insert(0, "🤖 AI Insights")
                tabs_list.append("💬 Chatbot") # Add Chatbot tab
            
            tabs = st.tabs(tabs_list)
            tab_offset = 0
            
            # AI Insights Tab
            if run_ai:
                with tabs[0]:
                    st.subheader("AI-Generated Summary")
                    st.markdown(ai_summary)
                    st.info("Aspect analysis is added as a new column in the 'Raw Data' tab for the first 10 reviews.")
                tab_offset = 1

            # Standard Tabs
            tab_viz, tab_wc, tab_data = tabs[tab_offset], tabs[tab_offset+1], tabs[tab_offset+2]



            with tab_viz:
                st.subheader("Sentiment Visualizations")
                fig_dist = create_sentiment_distribution_plot(df_processed)
                st.plotly_chart(fig_dist, use_container_width=True)

                if show_detailed_plots:
                    fig_scatter = create_polarity_vs_subjectivity_scatter(df_processed)
                    st.plotly_chart(fig_scatter, use_container_width=True)

            with tab_wc:
                if show_wordclouds:
                    st.subheader("Sentiment Word Clouds")
                    col_wc_1, col_wc_2, col_wc_3 = st.columns(3)
                    with col_wc_1:
                        fig_wc_pos = generate_wordcloud_figure(df_processed, 'Positive')
                        if fig_wc_pos:
                            st.plotly_chart(fig_wc_pos, use_container_width=True)
                    with col_wc_2:
                        fig_wc_neg = generate_wordcloud_figure(df_processed, 'Negative')
                        if fig_wc_neg:
                            st.plotly_chart(fig_wc_neg, use_container_width=True)
                    with col_wc_3:
                        fig_wc_all = generate_wordcloud_figure(df_processed, 'All')
                        if fig_wc_all:
                            st.plotly_chart(fig_wc_all, use_container_width=True)
                else:
                    st.info("Word cloud generation is disabled. Enable it in the sidebar to view.")

            with tab_data:
                st.subheader("Analyzed Data")
                st.dataframe(df_processed)

            # Chatbot Tab
            if run_ai and 'chatbot_chain' in st.session_state and st.session_state.chatbot_chain:
                with tabs[-1]: # Chatbot is always the last tab
                    st.subheader("Chat with Your Data")
                    
                    # Initialize chat history
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Display chat messages from history on app rerun
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # Accept user input
                    if prompt := st.chat_input("Ask a question about your data..."):
                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        # Display user message in chat message container
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # Get assistant response
                        with st.spinner("Thinking..."):
                            response = st.session_state.chatbot_chain.run(
                                question=prompt, 
                                data_context=st.session_state.data_context
                            )
                            with st.chat_message("assistant"):
                                st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

    else:
        st.info("Welcome! Please upload a CSV file to begin the sentiment analysis.")

if __name__ == "__main__":
    main()
