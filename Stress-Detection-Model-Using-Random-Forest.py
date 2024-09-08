import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Create a small synthetic dataset without the stress level parameter
np.random.seed(42)  # For reproducibility

# Generate simple features and labels
X = np.array([
    [70, 8, 1, 7, 40],  # [Heart Rate, Sleep Hours, Physical Activity, Diet Quality, Work Hours]
    [85, 6, 0.5, 5, 50],
    [90, 5, 1.5, 6, 60],
    [60, 9, 2, 8, 35],
    [75, 7, 1, 7, 45],
    [80, 6, 0.5, 4, 55],
    [95, 4, 2, 3, 65],
    [65, 8, 1, 9, 30],
    [88, 5, 1.2, 6, 50],
    [78, 7, 1.1, 7, 40]
])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])  # Labels: 0 (Not Stressed), 1 (Stressed)

# Step 2: Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Step 3: Define a function to predict stress based on user input
def predict_stress(heart_rate, sleep_hours, physical_activity, diet_quality, work_hours):
    # Make prediction
    prediction = model.predict([[heart_rate, sleep_hours, physical_activity, diet_quality, work_hours]])[0]
    # Get the probability (confidence score)
    probability = model.predict_proba([[heart_rate, sleep_hours, physical_activity, diet_quality, work_hours]])[0]
    confidence_score = probability[prediction]
    return prediction, confidence_score

# Step 4: Take user input and make prediction
print("Enter the following details to predict stress level:")
user_heart_rate = float(input("Heart Rate (e.g., 75): "))
user_sleep_hours = float(input("Sleep Hours (e.g., 7): "))
user_physical_activity = float(input("Physical Activity Level (hours per week, e.g., 3): "))
user_diet_quality = float(input("Diet Quality (scale 0 to 10, e.g., 7): "))
user_work_hours = float(input("Work Hours per Week (e.g., 40): "))

# Make prediction
predicted_stress, confidence = predict_stress(
    user_heart_rate,
    user_sleep_hours,
    user_physical_activity,
    user_diet_quality,
    user_work_hours
)
stress_status = "Stressed" if predicted_stress == 1 else "Not Stressed"

# Output results
print(f"Prediction: {stress_status}")
print(f"Confidence Score: {confidence * 100:.2f}%")
