import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("titanic.csv")

print("\nDATA LOADED SUCCESSFULLY\n")

# =========================
# 2. EXPLORATORY DATA ANALYSIS
# =========================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATA TYPES")
print(df.dtypes)

print("\nSUMMARY STATISTICS")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nUNIQUE VALUES")
print(df.nunique())

# =========================
# 3. VISUALISATION
# =========================

# --- Categorical Variables ---
sns.countplot(x="Survived", data=df)
plt.title("Survival Distribution")
plt.show()
plt.close()

sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.show()
plt.close()

sns.countplot(x="Embarked", data=df)
plt.title("Embarkation Port Distribution")
plt.show()
plt.close()

sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.show()
plt.close()

# --- Numerical Variables ---
sns.histplot(df["Age"].dropna(), kde=True)
plt.title("Age Distribution")
plt.show()
plt.close()

sns.histplot(df["Fare"], kde=True)
plt.title("Fare Distribution")
plt.show()
plt.close()

# --- Correlation Heatmap ---
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
plt.close()
