# Project 4: Medical Analysis Result Prediction

This project aims to predict medical analysis results (Normal, Abnormal, or Inconclusive) based on synthetic patient data. It was developed as part of the Module 5: Machine Learning (ML) course for the Executive Master « Data Engineering » (MSDE).

## Project Structure
- `healthcare_dataset.csv`: The synthetic dataset (10,000 records).
- `notebook.ipynb`: Complete EDA, preprocessing, and modeling pipeline.
- `train.py`: Python script for training and saving the best model.
- `app.py`: Streamlit application for interactive predictions.
- `model.joblib`: Serialized final model pipeline.
- `label_encoder.joblib`: Serialized target label encoder.
- `requirements.txt`: Python dependencies.

## How to Run
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the Model:**
   ```bash
   python train.py
   ```
3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## Model Details
We tested 10 different classification algorithms:
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

The final model is a **Random Forest Classifier** tuned via GridSearchCV.

## Deployment
The application is ready for deployment on Streamlit Cloud or any other platform supporting Python and Streamlit.
