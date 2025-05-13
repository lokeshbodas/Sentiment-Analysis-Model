# Sentiment-Analysis-Model

This project implements a **Sentiment Analysis Model** using **DistilBERT**, a transformer-based architecture, to classify Twitter comments into three categories: **Negative**, **Neutral**, and **Positive**. The model achieves strong performance metrics, making it suitable for real-world applications.

## Key Features

- **Data Preprocessing**:
  - Removed URLs, special characters, and unnecessary whitespace.
  - Replaced user mentions with placeholders (e.g., `[USER]`).
  - Tokenized text using `HuggingFace Transformers`.

- **Class Imbalance Handling**:
  - Applied class weights to address imbalanced datasets.
  - Improved performance on underrepresented classes.

- **Transformer-based Model**:
  - Fine-tuned `DistilBERT` for sequence classification.
  - Used custom loss functions and optimizers for better performance.

- **Evaluation Metrics**:
  - Generated classification reports, confusion matrices, and class-wise metrics.
  - Visualized training history for accuracy and loss.

## Results

### Enhanced Classification Report
| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.8721    | 0.8912 | 0.8815   | 743     |
| Neutral   | 0.8023    | 0.7854 | 0.7938   | 612     |
| Positive  | 0.8533    | 0.8421 | 0.8477   | 698     |

- **Accuracy**: 84.32%  
- **Macro F1-Score**: 0.8410  
- **Matthews Correlation Coefficient (MCC)**: 0.7634  

### Aggregate Metrics
| Metric              | Value  | Good Range | Excellent Range |
|---------------------|--------|------------|-----------------|
| **Macro F1-Score**  | 0.8410 | 0.75-0.85  | >0.85           |
| **Matthews CC**     | 0.7634 | 0.65-0.75  | >0.75           |
| **Accuracy**        | 84.32% | 75-85%     | >85%            |

### Training History
- **Final Epoch (5/5)**:
  - **Training Loss**: 0.3121  
  - **Training Accuracy**: 89.21%  
  - **Validation Loss**: 0.4012  
  - **Validation Accuracy**: 84.32%  

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python twitter_sa_1.py
   ```

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `transformers`
- `scikit-learn`

Install dependencies using:
```bash
pip install numpy pandas matplotlib seaborn tensorflow transformers scikit-learn
```

## Future Improvements

- Experiment with other transformer models like `BERT` or `RoBERTa`.
- Add hyperparameter tuning for better performance.
- Deploy the model as a web application using Flask or FastAPI.
