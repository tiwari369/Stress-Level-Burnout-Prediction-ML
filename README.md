# Stress Level and Burnout Prediction Using Machine Learning

A machine learning project that predicts stress level and burnout risk using structured input data. The project includes a trained XGBoost model and a Flask-based web frontend for user-friendly prediction.

## Project Overview

This project focuses on predicting stress and burnout levels using machine learning techniques. It uses input features related to stress indicators and applies a trained model to generate prediction results through a simple web interface.

## Features

* Stress level and burnout prediction
* XGBoost-based trained machine learning model
* Flask web application frontend
* Simple user input form
* Prediction output on web page
* Jupyter Notebook for model development
* Dataset included for academic/project reference

## Tech Stack

* Python
* Flask
* XGBoost
* Pandas
* NumPy
* Scikit-learn
* HTML
* CSS

## Project Structure

```text
Collab ML Code.ipynb                 Model training and experimentation notebook
StressLevelDataset.csv               Dataset used for training/testing
Frontend/
│── app.py                           Flask application
│── requirements.txt                 Required Python packages
│── XGBoost_Final_Burnout_Predictor.pkl  Trained XGBoost model
│── templates/
│   └── index.html                   Frontend HTML page
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/tiwari369/Stress-Level-Burnout-Prediction-ML.git
```

### 2. Open the project folder

```bash
cd Stress-Level-Burnout-Prediction-ML
```

### 3. Go to the frontend folder

```bash
cd Frontend
```

### 4. Install required packages

```bash
pip install -r requirements.txt
```

### 5. Run the Flask app

```bash
python app.py
```

### 6. Open in browser

```text
http://127.0.0.1:5000/
```

## Model Used

The project uses an XGBoost-based machine learning model for stress level and burnout prediction. XGBoost is suitable for structured/tabular data and often performs well on classification and regression tasks.

## Academic Note

This project was created as an academic major project for demonstrating machine learning-based prediction through a usable frontend application.

## Files Not Included Publicly

Academic/private files such as project report, plagiarism report, and presentation files are intentionally excluded from this public repository.

## Author

**Riteek Raj Tiwari**

## Disclaimer

This project is for academic and learning purposes only. It should not be used as a medical or psychological diagnosis tool. For real stress or mental health concerns, consult a qualified professional.
