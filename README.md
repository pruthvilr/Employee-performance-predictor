# 📈 Employee Performance Predictor (ML)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)](https://streamlit.io/)

An industry-oriented machine learning system designed to predict employee performance ratings (High, Medium, Low) and provide actionable insights for HR interventions.

---

## 📖 Project Overview
In many organizations, appraisal cycles are manual, biased, and reactive. This project builds a **data-driven "early warning system"** that helps HR managers and team leads identify performance trends before the review cycle ends.

By analyzing factors like training hours, project completion rates, and historical scores, the model predicts an employee's performance band and highlights the key drivers behind the prediction.

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, Scikit-Learn, NumPy, Matplotlib, Seaborn
- **Algorithm:** Random Forest Classifier (Ensemble Learning)
- **Deployment:** Streamlit (for the interactive dashboard)

## 📊 Dataset Schema
The project uses a synthetic dataset structured to mimic real enterprise HRMS data:
- **Demographics:** Age, Years of Experience, Department.
- **Productivity Metrics:** Projects Completed, Training Hours Attended.
- **Financial Context:** Monthly Salary.
- **Target Variable:** Performance Rating (0: Low, 1: Medium, 2: High).

## 🚀 Key Features
- **Predictive Modeling:** Uses a Random Forest algorithm for high accuracy and robustness.
- **Feature Importance:** Visualizes which factors (e.g., Training vs. Salary) impact performance the most.
- **Automated Preprocessing:** Handles categorical encoding and data scaling within a single pipeline.
- **Interactive Dashboard:** (Optional) Real-time prediction input via Streamlit.

---

## 📁 Folder Structure
```text
Employee-Performance-Predictor/
│
├── data/               # Raw and processed CSV datasets
├── models/             # Saved model files (.pkl)
├── notebooks/          # Exploratory Data Analysis (EDA) Jupyter Notebooks
├── src/                # Core Python scripts (preprocessing, training)
├── app.py              # Main dashboard application
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
