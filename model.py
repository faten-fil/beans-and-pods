import pandas as pd

# Chargement des données
def load_data():
    file_path = file_path = "C:/Users/Fatenne/Desktop/IA1/projet/BeansDataSet.csv"
  # Utilisation du fichier local
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        return None

# Fonction pour calculer les ventes totales
def calculate_total_sales(df):
    df["Total Ventes"] = df.iloc[:, 2:].sum(axis=1)
    return df

# Analyser les ventes par canal
def sales_by_channel(df):
    return df.groupby("Channel")["Total Ventes"].sum()

# Analyser les ventes par région
def sales_by_region(df):
    return df.groupby("Region")["Total Ventes"].sum()

# Analyser les ventes globales par produit
def sales_by_product(df):
    return df[["Robusta", "Arabica", "Espresso", "Lungo", "Latte", "Cappuccino"]].sum()

# Fonction pour analyser les données de manière croisée (région, canal)
def cross_analysis(df):
    return df.pivot_table(values="Total Ventes", index="Region", columns="Channel", aggfunc="sum")

# Suggestions pour la campagne marketing
def marketing_recommendations(df):
    region_sales = sales_by_region(df)
    top_region = region_sales.idxmax()
    top_sales = region_sales.max()

    return f"Investir davantage dans la région {top_region} où vous avez généré {top_sales} en ventes."

# Suggestion de nouvelles données à collecter pour la prédiction
def suggest_additional_data():
    return """
    Pour améliorer les prédictions et les analyses, Beans & Pods pourrait envisager de collecter les données suivantes :
    - Détails démographiques des clients (âge, sexe, statut marital, etc.)
    - Historique des achats de chaque client pour identifier les préférences de produit
    - Feedback des clients sur les produits pour identifier les tendances de satisfaction
    - Saison des ventes, pour analyser l'impact des périodes spécifiques (fêtes, vacances, etc.)
    - Données de l'offre concurrentielle pour évaluer l'impact de la concurrence.
    """
