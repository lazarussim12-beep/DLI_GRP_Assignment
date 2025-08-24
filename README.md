# 🚀 Spam Detection AI - Streamlit GUI

A powerful web-based interface for detecting spam emails using advanced machine learning models including CNN, XGBoost, LightGBM, and Logistic Regression.

## 📋 Prerequisites

- **Python 3.8 or higher**
- **Git** (for downloading the repository)
- **Internet connection** (for installing packages)

## 🛠️ Installation & Setup

### 1. Download the Project

```bash
# Clone the repository (if using Git)
git clone <your-repository-url>

# OR download and extract the ZIP file manually
# Extract to your desired folder
```

### 2. Navigate to the Project Folder

```bash
cd "Group AK submission/GroupAK_GUI"
```

### 3. Install Required Dependencies

```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt
```

**Alternative installation methods:**

```bash
# If you prefer conda
conda install --file requirements.txt

# Or install packages individually
pip install streamlit tensorflow keras numpy scikit-learn xgboost lightgbm pandas plotly regex
```

## 🚀 Running the Application

### Method 1: Using Streamlit (Recommended)

```bash
# Make sure you're in the GroupAK_GUI folder
cd "Group AK submission/GroupAK_GUI"

# Run the Streamlit app
streamlit run DLI_GroupAK_GUI.py
```

### Method 2: Using Python directly

```bash
# Run with Python
python -m streamlit run DLI_GroupAK_GUI.py
```

## 🌐 Accessing the Application

After running the command, the application will:
1. **Start automatically** in your default web browser
2. **Open at**: `http://localhost:8501`
3. **Show the interface** with spam detection capabilities

## 📁 Required Model Files

Make sure these files are in the same folder as `DLI_GroupAK_GUI.py`:
- `cnn_model.h5` - CNN neural network model
- `feature_extractor.h5` - Feature extraction model
- `ensemble_model_bundle.pkl` - Ensemble models bundle

## 🔧 Troubleshooting

### Common Issues & Solutions

**1. "Module not found" errors:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**2. Port already in use:**
```bash
# Use a different port
streamlit run DLI_GroupAK_GUI.py --server.port 8502
```

**3. Model loading errors:**
- Ensure all `.h5` and `.pkl` files are in the same directory
- Check file permissions
- Verify Python version compatibility

**4. Streamlit not working:**
```bash
# Update Streamlit
pip install --upgrade streamlit

# Check Streamlit version
streamlit --version
```

## 📱 Features

- **Real-time spam detection** using multiple ML models
- **User-friendly interface** with Streamlit
- **Ensemble predictions** for higher accuracy
- **Detailed confidence scores** and explanations
- **Responsive design** for all devices

## 🏗️ Architecture

The application uses an ensemble of models:
- **CNN (Convolutional Neural Network)** for text analysis
- **XGBoost** for gradient boosting
- **LightGBM** for lightweight gradient boosting
- **Logistic Regression** for baseline classification

## 📊 Performance

- **Fast prediction** (< 1 second per email)
- **High accuracy** through ensemble methods
- **Scalable** for batch processing

## 🤝 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure model files are present
4. Check Python version compatibility

## 📝 License

This project is part of the DLI Group AK submission for APU Y3 Sem 1.

---

**Happy Spam Detection! 🎯**
