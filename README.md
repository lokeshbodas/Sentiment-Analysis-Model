# Sentiment-Analysis-Model

This project focuses on building a **Sentiment Analysis Model** using **DistilBERT**, a transformer-based model, to classify Twitter comments into three categories: **Negative**, **Neutral**, and **Positive**. The model achieves an accuracy of **85%** on the validation dataset.

## Features

- **Data Preprocessing**: 
  - Removed URLs, special characters, and unnecessary whitespace.
  - Replaced user mentions with placeholders (e.g., `[USER]`).
  - Tokenized text using `HuggingFace Transformers`.

- **Class Imbalance Handling**:
  - Calculated and applied class weights to address imbalanced datasets.
  - Improved model performance on underrepresented classes.

- **Transformer-based Model**:
  - Used `DistilBERT` for sequence classification.
  - Fine-tuned the model with custom loss functions and optimizers.

- **Evaluation Metrics**:
  - Generated classification reports and confusion matrices.
  - Calculated class-wise accuracy for detailed performance analysis.

## Project Workflow

1. **Data Loading**:
   - Loaded training and validation datasets from CSV files.
   - Cleaned and preprocessed the data.

2. **Preprocessing**:
   - Applied custom text preprocessing functions to clean the comments.
   - Encoded labels using `LabelEncoder`.

3. **Tokenization**:
   - Tokenized text data using `AutoTokenizer` from `HuggingFace Transformers`.
   - Ensured compatibility with the transformer model.

4. **Model Training**:
   - Fine-tuned `DistilBERT` with class weights to handle imbalanced data.
   - Used `AdamW` optimizer with weight decay and early stopping to prevent overfitting.

5. **Evaluation**:
   - Evaluated the model using standard and weighted accuracy metrics.
   - Visualized results with confusion matrices and calculated class-wise accuracy.

## Key Results

- **Validation Accuracy**: 85%
- **Weighted Accuracy**: Improved performance on underrepresented classes.
- **Classification Report**: Detailed metrics for each sentiment class.
- **Confusion Matrix**: Visualized true vs. predicted labels.

## Dependencies

The project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `transformers`
- `scikit-learn`

Install the dependencies using the following commands:

```bash
pip install numpy pandas matplotlib seaborn tensorflow transformers scikit-learn
```

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

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Twitter\ SA_1.ipynb
   ```

4. Follow the notebook cells to preprocess data, train the model, and evaluate results.
