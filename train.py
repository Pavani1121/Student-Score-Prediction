import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("data/student_scores.csv")

X = df.drop("Score", axis=1)
y = df["Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", r2_score(y_test, pred))

joblib.dump(model, "model/model.pkl")

print("Model Saved")