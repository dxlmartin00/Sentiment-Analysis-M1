# import packages
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
import nltk


nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Initialize Gemini client
# The API key is automatically loaded from the GEMINI_API_KEY environment variable.
genai.configure(api_key="AIzaSyCBFLM7LrfLB448sueX631aoTBAzHJsbXQ")

# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "customer_reviews.csv")
    return csv_path


# Function to get sentiment using GenAI
@st.cache_data
def get_sentiment(text):
    if not text or pd.isna(text):
        return "Neutral"
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Classify the sentiment of the following review as exactly one word: Positive, Negative, or Neutral.
        Review: {text}
        Sentiment:"""
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        return response.text.strip()
    except Exception as e:
        st.error(f"API error: {e}")
        return "Neutral"


st.title("🔍 GenAI Sentiment Analysis Dashboard")
st.write("This is your GenAI-powered data processing app.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Load Dataset"):
        try:
            csv_path = get_dataset_path()
            df = pd.read_csv(csv_path)
            st.session_state["df"] = df.head(50)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🔍 Analyze Sentiment"):
        if "df" in st.session_state:
            try:
                with st.spinner("Analyzing sentiment..."):
                    st.session_state["df"].loc[:, "Sentiment"] = st.session_state["df"]["SUMMARY"].apply(get_sentiment)
                    st.success("Sentiment analysis completed!")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please ingest the dataset first.")

if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    
    st.dataframe(filtered_df)

    # Check if sentiment analysis has been performed
    if "Sentiment" in st.session_state["df"].columns:
        st.subheader(f"📊 Sentiment Breakdown for {product}")

        # Debug: Show what sentiments are actually in the data
        st.write("Debug - Unique sentiments found:", filtered_df["Sentiment"].unique())
        st.write("Debug - Sentiment value counts:", filtered_df["Sentiment"].value_counts())

        # Create comprehensive sentiment counts with all categories
        sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        # Define all possible sentiments and their properties
        all_sentiments = ['Negative', 'Neutral', 'Positive']
        sentiment_colors = {
            'Negative': '#ff4444',    # Red
            'Neutral': '#ffa500',     # Orange  
            'Positive': '#00cc66'     # Green
        }
        
        # Create a complete dataframe with all sentiment categories
        complete_sentiment_df = pd.DataFrame({'Sentiment': all_sentiments})
        complete_sentiment_df = complete_sentiment_df.merge(
            sentiment_counts, 
            on='Sentiment', 
            how='left'
        ).fillna(0)
        complete_sentiment_df['Count'] = complete_sentiment_df['Count'].astype(int)
        
        # Debug: Show the complete dataframe
        st.write("Debug - Complete sentiment dataframe:")
        st.write(complete_sentiment_df)
        
        # Ensure correct order
        complete_sentiment_df['Sentiment'] = pd.Categorical(
            complete_sentiment_df['Sentiment'], 
            categories=all_sentiments, 
            ordered=True
        )
        complete_sentiment_df = complete_sentiment_df.sort_values('Sentiment')
        
        # Only create chart if we have data
        if len(filtered_df) > 0:
            # Create the enhanced bar chart
            fig = px.bar(
                complete_sentiment_df,
                x="Sentiment",
                y="Count",
                title=f"Complete Sentiment Distribution - {product}",
                labels={"Sentiment": "Sentiment Category", "Count": "Number of Reviews"},
                color="Sentiment",
                color_discrete_map=sentiment_colors,
                text="Count"  # Show count values on bars
            )
            
            # Customize the chart appearance
            fig.update_traces(
                texttemplate='%{text}', 
                textposition='outside',
                textfont_size=14,
                textfont_color='black'
            )
            
            fig.update_layout(
                xaxis_title="Sentiment Category",
                yaxis_title="Number of Reviews",
                showlegend=False,
                title_font_size=16,
                xaxis_tickfont_size=14,
                yaxis_tickfont_size=12,
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Add grid for better readability
            fig.update_yaxes(gridcolor='lightgray', gridwidth=1)
            fig.update_xaxes(gridcolor='lightgray', gridwidth=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary statistics below the chart
            col1, col2, col3, col4 = st.columns(4)
            
            total_reviews = len(filtered_df)
            
            with col1:
                positive_count = complete_sentiment_df[complete_sentiment_df['Sentiment'] == 'Positive']['Count'].iloc[0]
                positive_pct = (positive_count / total_reviews * 100) if total_reviews > 0 else 0
                st.metric(
                    "Positive Reviews", 
                    f"{positive_count}",
                    f"{positive_pct:.1f}%"
                )
            
            with col2:
                negative_count = complete_sentiment_df[complete_sentiment_df['Sentiment'] == 'Negative']['Count'].iloc[0]
                negative_pct = (negative_count / total_reviews * 100) if total_reviews > 0 else 0
                st.metric(
                    "Negative Reviews", 
                    f"{negative_count}",
                    f"{negative_pct:.1f}%"
                )
            
            with col3:
                neutral_count = complete_sentiment_df[complete_sentiment_df['Sentiment'] == 'Neutral']['Count'].iloc[0]
                neutral_pct = (neutral_count / total_reviews * 100) if total_reviews > 0 else 0
                st.metric(
                    "Neutral Reviews", 
                    f"{neutral_count}",
                    f"{neutral_pct:.1f}%"
                )
            
            with col4:
                st.metric(
                    "Total Reviews", 
                    f"{total_reviews}",
                    "100%"
                )
        else:
            st.warning("No data available for the selected product.")
    else:
        st.info("💡 Click 'Analyze Sentiment' button to generate sentiment analysis and see the visualization.")
        st.warning("Sentiment analysis has not been performed yet. Please run the sentiment analysis first.")