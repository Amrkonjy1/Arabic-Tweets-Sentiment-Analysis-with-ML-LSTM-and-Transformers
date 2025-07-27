# Arabic Tweet Sentiment Analysis: A Comparative Study 📊

This project provides a comprehensive analysis and comparison of three distinct methodologies for sentiment classification on a dataset of Arabic tweets. The goal is to benchmark the performance of traditional Machine Learning models, a custom Deep Learning (LSTM) network, and a state-of-the-art, fine-tuned Transformer model.

## 🚀 Project Overview

Sentiment analysis on social media text, especially in a nuanced language like Arabic, presents unique challenges due to slang, dialects, and the use of non-textual cues like emojis. This repository explores and evaluates a range of techniques to tackle this problem effectively.

### Key Features:

* **Advanced Arabic NLP Preprocessing:** A robust pipeline for cleaning and normalizing Arabic text.
* **Smart Feature Engineering:** Emojis are not discarded but converted into textual features using the `emoji` library to capture their sentiment.
* **Three-Pronged Modeling Approach:**
    1.  **Classic ML:** Baseline models using TF-IDF.
    2.  **Deep Learning:** A Bidirectional LSTM network to learn from sequence context.
    3.  **Transformers:** Fine-tuning a pre-trained, language-specific BERT model.
* **In-depth Evaluation:** A thorough comparison of all models using key classification metrics like F1-score, precision, and recall.

## 🛠️ Methodologies & Technologies

### 1. Traditional Machine Learning (Baseline)

* **Vectorization:** `TF-IDF` with n-grams to create feature vectors from text.
* **Models:** A suite of classic classifiers were trained and evaluated, including:
    * `Logistic Regression`
    * `Linear SVC`
    * `Multinomial Naive Bayes`
    * `Random Forest`
    * `XGBoost`
* **Libraries:** `Scikit-learn`, `XGBoost`, `NLTK`

### 2. Deep Learning (Sequence Modeling)

* **Architecture:** A **Bidirectional LSTM** network was built to capture contextual information from both directions in a sentence.
* **Embeddings:** Custom, task-specific word embeddings were learned from scratch using a Keras `Embedding` layer.
* **Regularization:** The model employs a combination of `SpatialDropout1D`, standard `Dropout`, and `Recurrent Dropout` to prevent overfitting.
* **Libraries:** `TensorFlow`, `Keras`

### 3. Transformers (State-of-the-Art)

* **Model:** Fine-tuned the powerful, pre-trained **`CAMeL-BERT`** model (`CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`), which is specialized for Arabic.
* **Ecosystem:** Leveraged the **Hugging Face** ecosystem for a streamlined workflow:
    * `Transformers` for model loading and the `Trainer` API.
    * `Datasets` for efficient data handling and tokenization.
* **Training:** Utilized modern training techniques like `EarlyStopping` and mixed-precision training for efficiency and optimal results.

## 📈 Results & Conclusion

The models were evaluated on a held-out test set. The fine-tuned **CAMeL-BERT Transformer model achieved the highest performance**, demonstrating the power of transfer learning for this NLP task.

| Model                       | F1-Score   | Precision  | Recall     | Accuracy   |
| --------------------------- | :--------: | :--------: | :--------: | :--------: |
| **Transformer (CAMeL-BERT)**| **~0.947** | **~0.955** | **~0.939** | **~0.948** |
| Bidirectional LSTM          | ~0.937     | ~0.937     | ~0.938     | ~0.937     |
| Random Forest (Best ML)     | ~0.941     | ~0.947     | ~0.935     | ~0.942     |

*Note: The Random Forest model slightly edged out the LSTM, highlighting the effectiveness of classic models when well-tuned. However, the Transformer model showed superior overall performance.*
