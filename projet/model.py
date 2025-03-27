import pandas as pd

def load_data():
    file_path = file_path = "C:/Users/Fatenne/Desktop/IA1/projet/BeansDataSet.csv"

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        return None

def calculate_total_sales(df):
    df["Total Ventes"] = df.iloc[:, 2:].sum(axis=1)
    return df

def sales_by_channel(df):
    return df.groupby("Channel")["Total Ventes"].sum()

def sales_by_region(df):
    return df.groupby("Region")["Total Ventes"].sum()

def sales_by_product(df):
    return df[["Robusta", "Arabica", "Espresso", "Lungo", "Latte", "Cappuccino"]].sum()

def cross_analysis(df):
    return df.pivot_table(values="Total Ventes", index="Region", columns="Channel", aggfunc="sum")


