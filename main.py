import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# STEP 1: DATA SIMULATION (Synthetic Data)
# --------------------------------------------------
def generate_data(n=500):
    np.random.seed(42)
    data = {
        'Employee_ID': range(101, 101 + n),
        'Age': np.random.randint(22, 60, n),
        'Tenure_Years': np.random.randint(1, 20, n),
        'Projects_Completed': np.random.randint(1, 15, n),
        'Training_Hours': np.random.randint(10, 100, n),
        'Monthly_Salary': np.random.randint(3000, 10000, n),
        'Department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], n),
        'Performance_Score': np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3]) # 0:Low, 1:Med, 2:High
    }
    return pd.DataFrame(data)

df = generate_data()
print("✅ Dataset Created Successfully!")

# --------------------------------------------------
# STEP 2: PREPROCESSING
# --------------------------------------------------
# Convert Department to numbers
le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])

X = df.drop(['Employee_ID', 'Performance_Score'], axis=1)
y = df['Performance_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------
# STEP 3: MODEL TRAINING (Random Forest)
# --------------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --------------------------------------------------
# STEP 4: EVALUATION
# --------------------------------------------------
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------------------------------
# STEP 5: VISUALIZATION (Feature Importance)
# --------------------------------------------------
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Key Drivers of Employee Performance")
plt.show()