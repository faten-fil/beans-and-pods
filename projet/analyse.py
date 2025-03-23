import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv(r"C:\Users\Fatenne\Desktop\IA1\projet\BeansDataSet.csv", encoding='utf-8')


# Titre de l'application
st.title("Analyse des ventes - Beans & Pods")

# Aperçu rapide
st.subheader("Aperçu des données")
st.dataframe(df.head())

# Total ventes
df["Total Ventes"] = df.iloc[:, 2:].sum(axis=1)

# Ventes par canal
st.subheader("Ventes par canal")
channel_sales = df.groupby("Channel")["Total Ventes"].sum()
st.bar_chart(channel_sales)

# Ventes par région
st.subheader("Ventes par région")
region_sales = df.groupby("Region")["Total Ventes"].sum()
st.bar_chart(region_sales)

# Heatmap Région vs Canal vs Ventes
st.subheader("Analyse croisée région & canal")
pivot_table = df.pivot_table(values="Total Ventes", index="Region", columns="Channel", aggfunc="sum")
st.dataframe(pivot_table)

fig, ax = plt.subplots()
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
st.pyplot(fig)
