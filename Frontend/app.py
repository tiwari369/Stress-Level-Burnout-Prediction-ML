from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

MODEL_FILE = 'XGBoost_Final_Burnout_Predictor.pkl'

FEATURE_NAMES = [
    'anxiety_level', 'self_esteem', 'depression', 'headache', 'blood_pressure',
    'sleep_quality', 'breathing_problem', 'noise_level', 'living_conditions',
    'safety', 'basic_needs', 'academic_performance', 'study_load',
    'teacher_student_relationship', 'future_career_concerns',
    'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying'
]


MAPPING_DICT = {
    'mental_health_history': {'Yes': 1, 'No': 0},
    'self_esteem': {'Very Low': 1, 'Low': 2, 'Average': 3, 'High': 4, 'Very High': 5},
    'sleep_quality': {'Very Low': 1, 'Low': 2, 'Average': 3, 'High': 4, 'Very High': 5},
    'peer_pressure': {'Very Low': 1, 'Low': 2, 'Average': 3, 'High': 4, 'Very High': 5},
    'depression': {'None/Minimal': 1, 'Mild': 2, 'Moderate': 3, 'Moderately Severe': 4, 'Severe': 5},
    'blood_pressure': {'Normal': 1, 'Elevated': 2, 'Hypertension Stage 1': 3, 'Hypertension Stage 2': 4, 'Hypertensive Crisis': 5},
    'future_career_concerns': {'Very Confident': 1, 'Confident': 2, 'Neutral/Slightly Concerned': 3, 'Worried': 4, 'Extremely Worried': 5},
    'headache': {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5},
    'breathing_problem': {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5},
    'bullying': {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5},
    'extracurricular_activities': {'Very Rare': 1, 'Rare': 2, 'Average': 3, 'Often': 4, 'Very Often': 5},
    'noise_level': {'Low': 1, 'Medium': 3, 'High': 5},
    'study_load': {'Very Light': 1, 'Light': 2, 'Average': 3, 'Heavy': 4, 'Very Heavy': 5},
    'living_conditions': {'Worst': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Best': 5},
    'safety': {'Very Unsafe': 1, 'Unsafe': 2, 'Average': 3, 'Safe': 4, 'Very Safe': 5},
    'basic_needs': {'Very Poor': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5},
    'academic_performance': {'Very Poor': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5},
    'teacher_student_relationship': {'Very Poor': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5},
    'social_support': {'Very Poor': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5},
}


def get_suggestions(risk_level):
    """Provides tailored suggestions based on the burnout risk level."""
    if risk_level == "High":
        return [
            "⚠️ **Seek Immediate Support:** Contact a university counselor, mental health professional, or trusted family member right away.",
            "⚖️ **Re-evaluate Study Load:** Break down large tasks into smaller steps. Discuss potential course load reductions or extensions with your academic advisor.",
            "💤 **Prioritize Sleep:** Establish a consistent sleep schedule and ensure 7-9 hours of quality sleep. Minimize screen time before bed.",
            "🏃 **Incorporate Movement:** Even a 15-minute walk can help. Physical activity is crucial for managing stress and anxiety.",
            "🗣️ **Strengthen Social Support:** Spend quality time with friends or family. Don't isolate yourself."
        ]
    else: 
        return [
            "🧘 **Maintain Wellness Routine:** Continue to practice stress-reducing activities like meditation, deep breathing, or hobbies.",
            "📅 **Practice Time Management:** Use a planner or calendar to manage your study load efficiently and ensure you schedule breaks.",
            "🍎 **Focus on Nutrition:** Maintain a balanced diet and stay hydrated. Avoid excessive caffeine and sugar.",
            "🔄 **Regular Breaks:** Ensure you take short, frequent breaks during study sessions (e.g., the Pomodoro Technique).",
            "🤝 **Stay Connected:** Keep lines of communication open with teachers and peers to maintain a supportive environment."
        ]

try:
    with open(MODEL_FILE, 'rb') as file:
        model = pickle.load(file)
    print(" Model Loaded Successfully!")
except Exception as e:
    print(" Model Load Error:", e)
    model = None

@app.route('/')
def home():
    return render_template('index.html', feature_names=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction API endpoint."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.form.to_dict()
        input_values = []

        for name in FEATURE_NAMES:
            value = data[name]

            if name in MAPPING_DICT:
                # Text value ko number mein badlo
                numerical_value = MAPPING_DICT[name].get(value, 0)
            else:
                # Ab sirf 'anxiety_level' bacha hai jismein float value leni hai
                numerical_value = float(value)

            input_values.append(numerical_value)


        input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            status = "HIGH Burnout Risk Detected"
            risk_level = "High"
        else:
            status = "Low Burnout Risk"
            risk_level = "Low"
            
        # Get personalized suggestions
        suggestions = get_suggestions(risk_level)

        return jsonify({
            'status': status,
            'probability': f"{proba*100:.2f}%",
            'risk_level': risk_level,
            'suggestions': suggestions # Add suggestions to the response
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f"Prediction input mein error aaya: {str(e)}"}, 400)


if __name__ == '__main__':
    app.run(debug=True)