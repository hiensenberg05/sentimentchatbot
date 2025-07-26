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
        
        # Show warning for large files
        if uploaded_file is not None:
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            if file_size > 1:  # Warning for files larger than 1MB
                st.warning(f"⚠️ Large file detected ({file_size:.1f}MB). This may cause rate limit issues with AI features. Consider using a smaller sample for better performance.")
            
            # Add option to limit data size for better performance
            if file_size > 5:  # For very large files
                st.info("💡 **Performance Tip:** Large files may be slow. The app will automatically sample data for AI features.")
        
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
                    
                    # Add a clear chat button
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
                            st.session_state.messages = []
                            st.session_state.chatbot_chain.memory.clear()
                            st.rerun()
                    
                    # Initialize chat history with better structure
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                        # Add initial system message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "👋 Hi! I'm your AI assistant. I can help you analyze your sentiment data. Ask me anything about the reviews, sentiment distribution, or specific patterns you've noticed!"
                        })

                    # Display chat messages from history with better formatting
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            if message["role"] == "assistant" and message["content"].startswith("👋"):
                                st.markdown(message["content"])
                            else:
                                st.markdown(message["content"])

                    # Accept user input with better UX
                    if prompt := st.chat_input("Ask a question about your data...", key="chat_input"):
                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        
                        # Display user message immediately
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # Get assistant response with optimized performance
                        with st.chat_message("assistant"):
                            with st.spinner("🤔 Analyzing..."):
                                try:
                                    # Use a more efficient call with timeout
                                    response = st.session_state.chatbot_chain.run(
                                        question=prompt
                                    )
                                    
                                    # Clean up response if needed
                                    if response and len(response.strip()) > 0:
                                        st.markdown(response)
                                        st.session_state.messages.append({"role": "assistant", "content": response})
                                    else:
                                        error_response = "I couldn't generate a response. Please try asking a different question."
                                        st.markdown(error_response)
                                        st.session_state.messages.append({"role": "assistant", "content": error_response})
                                        
                                except Exception as e:
                                    error_msg = str(e)
                                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                                        response = "⚠️ **Rate limit exceeded!** Please wait 30 seconds and try again, or ask a shorter question."
                                    elif "tokens" in error_msg.lower():
                                        response = "⚠️ **Input too large!** Try asking a more specific question about your data."
                                    elif "timeout" in error_msg.lower():
                                        response = "⏱️ **Request timed out!** The response took too long. Try a simpler question."
                                    else:
                                        response = f"❌ **Error:** {error_msg[:100]}..." if len(error_msg) > 100 else f"❌ **Error:** {error_msg}"
                                    
                                    st.markdown(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Add helpful suggestions
                    if len(st.session_state.messages) <= 2:  # Only show if chat is new
                        st.markdown("---")
                        st.markdown("**💡 Try asking:**")
                        suggestions = [
                            "What's the overall sentiment distribution?",
                            "What are the most common positive themes?",
                            "Show me some negative feedback examples",
                            "What's the average sentiment score?"
                        ]
                        for suggestion in suggestions:
                            if st.button(suggestion, key=f"suggest_{suggestion[:20]}"):
                                st.session_state.messages.append({"role": "user", "content": suggestion})
                                st.rerun()

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

    else:
        st.info("Welcome! Please upload a CSV file to begin the sentiment analysis.")

if __name__ == "__main__":
    main()
