import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Data Analysis", layout="wide")

st.title("Titanic Data Analysis")

st.write(
    "This application presents exploratory data analysis of the Titanic dataset "
    "including summary statistics, data types, missing values, and visualisations."
)

# Load dataset
df = pd.read_csv("titanic.csv")

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df)

# -----------------------------
# Data Types (NEW SECTION)
# -----------------------------
st.subheader("Data Types of Each Column")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(dtype_df)

# -----------------------------
# Summary Statistics
# -----------------------------
st.subheader("Summary Statistics")
st.dataframe(df.describe())

# -----------------------------
# Missing Values
# -----------------------------
st.subheader("Missing Values")
st.dataframe(df.isnull().sum())

# -----------------------------
# Survival Distribution
# -----------------------------
st.subheader("Survival Distribution")

fig, ax = plt.subplots()
sns.countplot(data=df, x="Survived", ax=ax)
st.pyplot(fig)

# -----------------------------
# Gender Distribution
# -----------------------------
st.subheader("Gender Distribution")

fig, ax = plt.subplots()
sns.countplot(data=df, x="Sex", ax=ax)
st.pyplot(fig)

# -----------------------------
# Passenger Class Distribution
# -----------------------------
st.subheader("Passenger Class Distribution")

fig, ax = plt.subplots()
sns.countplot(data=df, x="Pclass", ax=ax)
st.pyplot(fig)

# -----------------------------
# Age Distribution
# -----------------------------
st.subheader("Age Distribution")

fig, ax = plt.subplots()
sns.histplot(df["Age"].dropna(), kde=True, ax=ax)
st.pyplot(fig)

# -----------------------------
# Fare Distribution
# -----------------------------
st.subheader("Fare Distribution")

fig, ax = plt.subplots()
sns.histplot(df["Fare"], kde=True, ax=ax)
st.pyplot(fig)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
st.pyplot(fig)