import joblib

model = joblib.load("model/model.pkl")

study = float(input("Study Hours: "))
sleep = float(input("Sleep Hours: "))
attendance = float(input("Attendance: "))
previous = float(input("Previous Marks: "))

prediction = model.predict([[study, sleep, attendance, previous]])

print("Predicted Score:", prediction[0])