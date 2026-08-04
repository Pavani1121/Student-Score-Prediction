import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("data/student_scores.csv")

# Create scatter plot
plt.scatter(df["StudyHours"], df["Score"])

plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score")

plt.show()