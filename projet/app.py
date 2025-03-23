import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette(["#FFB6C1", "#87CEFA", "#98FB98"])  # rose, bleu, vert pâle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model import (
    load_data, calculate_total_sales, sales_by_channel,
    sales_by_region, sales_by_product, cross_analysis,
    marketing_recommendations, suggest_additional_data
)

# Palette personnalisée
custom_palette = ["#FFB6C1", "#87CEFA", "#98FB98"]
sns.set_palette(custom_palette)

def main():
    st.set_page_config(page_title="Analyse des ventes - Beans & Pods", page_icon="☕")

    st.title("Analyse des ventes - Beans & Pods")
    st.write("""
        Bienvenue sur l'application de **Beans & Pods**. 
        Cette application vous permet d'explorer les ventes, 
        d'obtenir des recommandations marketing et de suggérer 
        des données à collecter pour améliorer les prévisions.
    """)

    df = load_data()

    if df is not None:
        df = calculate_total_sales(df)

        option = st.selectbox("Sélectionner une option", 
            ["Analyse des ventes", "Suggestions de données supplémentaires", 
             "Recommandations marketing", "Régression linéaire", "Afficher les ventes"])

        if option == "Analyse des ventes":
            plot_type = st.radio("Choisir le type de graphique", 
                                 ("Bar chart", "Line chart", "Heatmap", "Box plot"))

            # Ventes par canal
            st.subheader("Ventes par canal")
            channel_sales = sales_by_channel(df)

            if plot_type == "Bar chart":
                st.bar_chart(channel_sales)
            elif plot_type == "Line chart":
                st.line_chart(channel_sales)
            elif plot_type == "Heatmap":
                fig, ax = plt.subplots()
                sns.heatmap(channel_sales.values.reshape(1, -1), annot=True, fmt=".0f", 
                            cmap=sns.color_palette(custom_palette, as_cmap=True), ax=ax)
                st.pyplot(fig)
            elif plot_type == "Box plot":
                st.write("Box plot pour les ventes par canal")
                fig, ax = plt.subplots()
                sns.boxplot(data=df[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']], 
                            palette=custom_palette, ax=ax)
                st.pyplot(fig)

            # Ventes par région
            st.subheader("Ventes par région")
            region_sales = sales_by_region(df)

            if plot_type == "Bar chart":
                st.bar_chart(region_sales)
            elif plot_type == "Line chart":
                st.line_chart(region_sales)
            elif plot_type == "Heatmap":
                fig, ax = plt.subplots()
                sns.heatmap(region_sales.values.reshape(1, -1), annot=True, fmt=".0f", 
                            cmap=sns.color_palette(custom_palette, as_cmap=True), ax=ax)
                st.pyplot(fig)
            elif plot_type == "Box plot":
                st.write("Box plot pour les ventes par région")
                fig, ax = plt.subplots()
                sns.boxplot(data=df[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']], 
                            palette=custom_palette, ax=ax)
                st.pyplot(fig)

            # Ventes par produit
            st.subheader("Ventes par produit")
            product_sales = sales_by_product(df)

            if plot_type == "Bar chart":
                st.bar_chart(product_sales)
            elif plot_type == "Line chart":
                st.line_chart(product_sales)
            elif plot_type == "Heatmap":
                fig, ax = plt.subplots()
                sns.heatmap(product_sales.values.reshape(1, -1), annot=True, fmt=".0f", 
                            cmap=sns.color_palette(custom_palette, as_cmap=True), ax=ax)
                st.pyplot(fig)
            elif plot_type == "Box plot":
                st.write("Box plot pour les ventes par produit")
                fig, ax = plt.subplots()
                sns.boxplot(data=df[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']], 
                            palette=custom_palette, ax=ax)
                st.pyplot(fig)

            # Analyse croisée
            st.subheader("Analyse croisée région & canal")
            cross_table = cross_analysis(df)
            st.dataframe(cross_table)

            if plot_type == "Heatmap":
                fig, ax = plt.subplots()
                sns.heatmap(cross_table, annot=True, fmt=".0f", 
                            cmap=sns.color_palette(custom_palette, as_cmap=True), ax=ax)
                st.pyplot(fig)

        elif option == "Suggestions de données supplémentaires":
            st.subheader("Suggestions de données supplémentaires à collecter")
            suggestions = suggest_additional_data()
            st.write(suggestions)

        elif option == "Recommandations marketing":
            st.subheader("Recommandations pour la nouvelle campagne marketing")
            recommendations = marketing_recommendations(df)
            st.write(recommendations)

        elif option == "Régression linéaire":
            st.subheader("Régression linéaire pour prédire les ventes")
            
            X = df[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']]
            y = df['Total Ventes']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.write("### Coefficients de la régression :")
            for feature, coef in zip(X.columns, model.coef_):
                st.write(f"- {feature}: {coef:.2f}")

            mse = mean_squared_error(y_test, y_pred)
            st.write(f"### Erreur quadratique moyenne (MSE) : {mse:.2f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color="#87CEFA")
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='#FFB6C1', linestyle='--')
            ax.set_xlabel("Ventes réelles")
            ax.set_ylabel("Ventes prédites")
            ax.set_title("Régression linéaire : Ventes réelles vs Ventes prédites")
            st.pyplot(fig)

        elif option == "Afficher les ventes":
            st.subheader("📊 Aperçu des ventes")

            col1, col2, col3 = st.columns(3)
            col1.metric("Canaux de vente", df['Channel'].nunique())
            col2.metric("Régions couvertes", df['Region'].nunique())
            col3.metric("Transactions", len(df))

            st.subheader("☕ Pourcentage des ventes par produit")
            ventes_par_produit = df[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']].sum()
            total_ventes = ventes_par_produit.sum()
            ventes_percent = (ventes_par_produit / total_ventes * 100).round(2)

            percent_df = pd.DataFrame({
                "Produit": ventes_percent.index,
                "Pourcentage (%)": ventes_percent.values
            })

            st.dataframe(percent_df)

            st.subheader("🧠 Taille mémoire du dataset")
            mem_size = df.memory_usage(deep=True).sum() / 1024  # en KB
            st.write(f"Taille du dataset : **{mem_size:.2f} KB**")

            st.subheader("🧾 Données brutes")
            st.dataframe(df)

if __name__ == "__main__":
    main()
