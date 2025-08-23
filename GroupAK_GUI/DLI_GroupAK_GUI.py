import streamlit as st
import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import xgboost as xgb
import lightgbm as lgb
import os

# Page configuration
st.set_page_config(
    page_title="Spam Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .spam-result {
        background-color: #fdf2f2;
        border-left: 5px solid #e74c3c;
    }
    .ham-result {
        background-color: #f0f9ff;
        border-left: 5px solid #27ae60;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

@st.cache_resource
def load_models():
    """Load all trained models and preprocessors"""
    try:
        # Load CNN models from .h5 files
        cnn_model = load_model("GroupAK_GUI/cnn_model.h5")
        feature_extractor = load_model("GroupAK_GUI/feature_extractor.h5")
        
        # Load ensemble models from bundled .pkl file
        with open("GroupAK_GUI/ensemble_model_bundle.pkl", "rb") as f:
            model_bundle = pickle.load(f)
        
        return {
            'cnn_model': cnn_model,
            'feature_extractor': feature_extractor,
            'lr': model_bundle["lr"],
            'xgb_model': model_bundle["xgb_model"],
            'lgb_model': model_bundle["lgb_model"],
            'lr_cnn_only': model_bundle["lr_cnn_only"],
            'vectorizer': model_bundle["vectorizer"],
            'tokenizer': model_bundle["tokenizer"],
            'max_words': model_bundle["max_words"],
            'max_len': model_bundle["max_len"]
        }
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None

def preprocess_message(message):
    """Preprocess the input message"""
    # Clean the message
    msg_clean = re.sub(r'[^a-zA-Z0-9\s]', '', message.lower())
    return msg_clean

def predict_spam(message, models):
    """Make prediction using ensemble models"""
    try:
        # Preprocess message
        msg_clean = preprocess_message(message)
        
        # TF-IDF features for traditional models
        X_tfidf = models['vectorizer'].transform([msg_clean])
        
        # CNN features
        X_seq = pad_sequences(
            models['tokenizer'].texts_to_sequences([msg_clean]), 
            maxlen=models['max_len']
        )
        X_cnn_feat = models['feature_extractor'].predict(X_seq)
        
        # Get predictions from all models
        proba_lr = models['lr'].predict_proba(X_tfidf)[:, 1][0]
        proba_xgb = models['xgb_model'].predict_proba(X_tfidf)[:, 1][0]
        proba_lgb = models['lgb_model'].predict_proba(X_tfidf)[:, 1][0]
        proba_cnn = models['lr_cnn_only'].predict_proba(X_cnn_feat)[:, 1][0]
        
        # Ensemble prediction
        ensemble_proba = (proba_xgb + proba_lgb + proba_cnn + proba_lr) / 4
        prediction = "Spam" if ensemble_proba > 0.5 else "Ham"
        
        # Calculate confidence
        confidence = ensemble_proba if prediction == "Spam" else (1 - ensemble_proba)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'individual_probas': {
                'Logistic Regression': proba_lr,
                'XGBoost': proba_xgb,
                'LightGBM': proba_lgb,
                'CNN': proba_cnn,
                'Ensemble': ensemble_proba
            }
        }
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Spam Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Ensemble Model with CNN, XGBoost, LightGBM & Logistic Regression</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Model Information")
    st.sidebar.markdown("""
    **Models Used:**
    - CNN (Convolutional Neural Network)
    - XGBoost
    - LightGBM
    - Logistic Regression
    
    **Features:**
    - TF-IDF Vectorization
    - CNN Feature Extraction
    - Soft Voting
    """)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models..."):
            models = load_models()
            if models:
                st.session_state.models = models
                st.session_state.models_loaded = True
                st.success("✅ All models loaded successfully!")
            else:
                st.error("❌ Failed to load models. Please check if model files exist.")
                return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📧 Email Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a text file with email content",
            type=['txt'],
            help="Upload a .txt file containing the email message to analyze"
        )
        
        # Text input
        message_input = st.text_area(
            "Or type/paste email message here:",
            height=200,
            placeholder="Enter the email message you want to analyze for spam detection..."
        )
        
        # Get the message from either file upload or text input
        message = ""
        if uploaded_file is not None:
            message = uploaded_file.getvalue().decode("utf-8")
        elif message_input:
            message = message_input
        
        # Predict button
        predict_button = st.button(
            "🔍 Analyze Email",
            type="primary",
            disabled=not message.strip(),
            help="Click to analyze the email message for spam detection"
        )
        
        # Make prediction
        if predict_button and message.strip():
            with st.spinner("Analyzing message with AI ensemble..."):
                result = predict_spam(message, st.session_state.models)
                
                if result:
                    st.session_state.prediction_made = True
                    st.session_state.result = result
                    st.session_state.message = message
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        if st.session_state.models_loaded:
            st.metric("Models Loaded", "4/4", "✅")
            st.metric("Status", "Ready", "🟢")
        else:
            st.metric("Models Loaded", "0/4", "❌")
            st.metric("Status", "Loading", "🟡")
        
        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown("""
        1. **Preprocess** email text
        2. **Extract** features using TF-IDF & CNN
        3. **Analyze** with 4 AI models
        4. **Combine** predictions for final result
        """)
    
    # Results section
    if st.session_state.prediction_made:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        result = st.session_state.result
        message = st.session_state.message
        
        # Display message preview
        with st.expander("📧 Message Preview", expanded=False):
            st.text(message[:500] + "..." if len(message) > 500 else message)
        
        # Main result
        if result['prediction'] == "Spam":
            st.markdown(f"""
            <div class="result-box spam-result">
                <h2 style="color: #e74c3c;">🚨 SPAM DETECTED!</h2>
                <p style="font-size: 1.2rem;">This email appears to be spam and should be treated with caution.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box ham-result">
                <h2 style="color: #27ae60;">✅ LEGITIMATE EMAIL</h2>
                <p style="font-size: 1.2rem;">This email appears to be legitimate and safe.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence score
        confidence_percent = result['confidence'] * 100
        st.metric(
            "🎯 Confidence Level",
            f"{confidence_percent:.1f}%",
            delta=f"{confidence_percent:.1f}% confidence"
        )
        
        # Individual model probabilities
        st.subheader("🔍 Individual Model Predictions")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Logistic Regression", f"{result['individual_probas']['Logistic Regression']:.3f}")
        with col2:
            st.metric("XGBoost", f"{result['individual_probas']['XGBoost']:.3f}")
        with col3:
            st.metric("LightGBM", f"{result['individual_probas']['LightGBM']:.3f}")
        with col4:
            st.metric("CNN", f"{result['individual_probas']['CNN']:.3f}")
        with col5:
            st.metric("Ensemble", f"{result['individual_probas']['Ensemble']:.3f}")
        
        # Detailed analysis
        st.subheader("📈 Detailed Analysis")
        
        # Create a bar chart of individual probabilities
        import plotly.express as px
        
        proba_data = {
            'Model': list(result['individual_probas'].keys()),
            'Probability': list(result['individual_probas'].values())
        }
        
        fig = px.bar(
            proba_data,
            x='Model',
            y='Probability',
            title="Individual Model Probabilities",
            color='Probability',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Reset button
        if st.button("🔄 Analyze Another Email"):
            st.session_state.prediction_made = False
            st.rerun()
    
    # Footer with sharing information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;">
        <h3>🌐 Share This App</h3>
        <p>This Streamlit app can be deployed and shared publicly. Use <code>streamlit run SpamEnsembleStreamlit.py --server.port 8501</code> to run locally.</p>
        <p>For public deployment, deploy to Streamlit Cloud or other platforms for public access.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
