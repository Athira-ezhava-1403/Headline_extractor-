import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required data safely
@st.cache_resource
def load_nlp_data():
    nltk.download('punkt')
    nltk.download('stopwords')

load_nlp_data()

# Initialize tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# --- Page configuration ---
st.set_page_config(
    page_title="Headline Extractor",
    page_icon="📰",
    layout="wide", # This makes it look like a real dashboard rather than squished
)

# --- App Header ---
st.title("📰 News Headline Keyword Extractor")
st.markdown("Transform messy text data into clean, machine-ready root keywords instantly.")
st.write("---")

# --- Layout Columns ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("🛠️ Input Panel")
    option = st.radio("Choose input method:", ("Paste a Headline", "Upload a .txt File"))
    
    user_input = ""
    if option == "Paste a Headline":
        user_input = st.text_input(
            "Enter a news headline:", 
            "Government announces new policy to boost renewable energy investments"
        )
    else:
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        if uploaded_file is not None:
            lines = uploaded_file.read().decode("utf-8").splitlines()
            # Grab the first non-empty line for display purposes
            non_empty = [line for line in lines if line.strip()]
            if non_empty:
                user_input = non_empty[0]
                st.info(f"📁 File loaded. Showing processing steps for the first headline.")

with col2:
    st.subheader("📊 Processing Pipeline Results")
    
    if user_input:
        # --- NLP PROCESSING STEPS ---
        # 1. Tokenization
        tokens = word_tokenize(user_input)
        
        # 2. Cleaning (Lowercasing + removing punctuation/symbols)
        cleaned_tokens = [token.lower() for token in tokens if token.isalnum()]
        
        # 3. Stopword Removal
        filtered_tokens = [token for token in cleaned_tokens if token not in stop_words]
        
        # 4. Stemming / Root Keywords
        root_keywords = [ps.stem(token) for token in filtered_tokens]
        
        # --- VISUAL RENDERING ---
        
        # Highlight original headline beautifully
        st.info(f"**Original Text:** {user_input}")
        
        # Quick metrics showing data reduction
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Original Word Count", len(tokens))
        metric_col2.metric("Final Root Keywords", len(root_keywords), delta=f"{len(root_keywords) - len(tokens)} words", delta_color="inverse")
        
        # Visual breakdown steps using cards
        with st.container():
            st.markdown("#### 🔄 Step-by-Step Breakdown")
            
            st.markdown("**1. Tokens Generated**")
            st.code(f"{tokens}", language="python")
            
            st.markdown("**2. After Cleaning (Lowercase & Punctuation Stripped)**")
            st.code(f"{cleaned_tokens}", language="python")
            
            st.markdown("**3. After Stopwords Removed (Grammar words dropped)**")
            st.code(f"{filtered_tokens}", language="python")
            
            st.markdown("### 🎯 Final Extracted Root Keywords")
            st.success(f"🚀 {root_keywords}")
    else:
        st.warning("Please provide an input headline on the left panel to begin processing.")