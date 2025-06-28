# 📰 Fake News Detection using Deep Learning

This project detects whether a news article is real or fake using a GRU-based deep learning model. If an article is predicted as fake, it also suggests a correct real news article on the same topic using keyword extraction and real-time news search.

## 👥 Team

- 2 Fullstack Developers
- 2 AI/ML Developers

## 🚀 Features

- Detect fake vs real news with high accuracy using a GRU model
- Tokenizer-based preprocessing and padded sequences
- Real news suggestion using keyword extraction + NewsAPI
- Modular folder structure for collaboration
- Model & tokenizer saved and ready for deployment

## 🧠 Model Architecture

- Embedding Layer
- GRU Layer
- Dropout Layer
- Dense (sigmoid) Output Layer
- Trained on: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Achieved Test Accuracy: ✅ 99.84%

## 📂 Project Structure
<pre lang="markdown">
fake-news-detector/
├── model/
│ ├── fake_news_model.h5 ← Trained model
│ └── tokenizer.pkl ← Keras tokenizer
├── retriever/
│ └── fetch_real_news.py ← Real news retriever (NewsAPI + spaCy)
├── app/
│ └── app.py ← Flask/FastAPI backend (to be added)
├── notebooks/
│ └── training_notebook.ipynb ← Colab notebook used for training
├── requirements.txt
└── README.md
</pre>

## 🔧 How to Run (Locally)

### 1. Clone the repository:

   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector

### 2. Install dependencies:

    pip install -r requirements.txt

### 3. To load model & make predictions:

    from tensorflow.keras.models import load_model
    import pickle

    model = load_model("model/fake_news_model.h5")
    with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

### 4. To retrieve real news suggestions:

    from retriever.fetch_real_news import fetch_real_news
    result = fetch_real_news("Some fake news text here")

## 📦 Dependencies

    * TensorFlow / Keras
    * Pandas, NumPy
    * scikit-learn
    * spaCy / KeyBERT (for keyword extraction)
    * Requests (for NewsAPI)

## 🌐 API Integration (Planned)

    We will add a Flask or FastAPI backend with a /predict route that:
    * Classifies the news input.
    * If fake, returns the closest matching real news article from trusted sources.

## 📣 Acknowledgments

    * Dataset: Kaggle - Fake and Real News
    * NewsAPI.org for real news suggestions



