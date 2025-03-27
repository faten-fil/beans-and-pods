import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_palette(["#FFB6C1", "#87CEFA", "#98FB98"])  
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model import (
    load_data, calculate_total_sales, sales_by_channel,
    sales_by_region, sales_by_product, cross_analysis,
    
)
from rapport import(
    rapport
)

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
            ["Data visualization", "Model building", "Afficher les ventes","Data cleaning","Rapport danalyse"])

        if option == "Data visualization":
            plot_type = st.radio("Choisir le type de graphique", 
                                 ("Bar chart", "Line chart", "Heatmap", "Box plot","Matrice de corrélation"))

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
            elif plot_type == "Matrice de corrélation":
                st.subheader("📊 Matrice de corrélation")
                st.write("""
                La **matrice de corrélation** permet d'examiner la force et la direction des relations linéaires 
                entre les différentes variables numériques du dataset.
                """)
                corr = df.select_dtypes(include='number').corr()
                fig, ax = plt.subplots(figsize=(10, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, ax=ax)
                ax.set_title("Matrice de corrélation", fontsize=14)
                st.pyplot(fig)

                st.markdown("""
                  **Légende :**
                  - Les valeurs proches de 1 indiquent une forte corrélation positive.
                  - Les valeurs proches de -1 indiquent une forte corrélation négative.
                  - Les valeurs proches de 0 indiquent peu ou pas de corrélation.
                  """)




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

            st.subheader("Analyse croisée région & canal")
            cross_table = cross_analysis(df)
            st.dataframe(cross_table)

            if plot_type == "Heatmap":
                fig, ax = plt.subplots()
                sns.heatmap(cross_table, annot=True, fmt=".0f", 
                            cmap=sns.color_palette(custom_palette, as_cmap=True), ax=ax)
                st.pyplot(fig)

        elif option == "Model building":
          st.subheader("🔧 Construction d'un modèle de machine learning")

          st.markdown("""
         Choisissez un **type de modèle de régression** ainsi qu'une **métrique d'évaluation**.
          Les résultats apparaîtront automatiquement.
          """)

          model_choice = st.radio("Choisissez un modèle :", 
           ["Linear Regression", "Ridge", "Lasso", "K-Nearest Neighbors", "Decision Tree", "Support Vector Regressor"])

          metric_choice = st.radio("Choisissez une métrique :", ["MAE", "MSE", "R2"])

          features = ['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']
          X = df[features].values
          y = df['Total Ventes'].values

          test_size = 0.3
          seed = 41
          x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

          models_dict = {
             "Linear Regression": LinearRegression(),
             "Ridge": Ridge(),
             "Lasso": Lasso(),
             "K-Nearest Neighbors": KNeighborsRegressor(),
              "Decision Tree": DecisionTreeRegressor(),
               "Support Vector Regressor": SVR()
             }

          metrics_dict = {
          "MAE": mean_absolute_error,
          "MSE": mean_squared_error,
          "R2": r2_score
          }

          model = models_dict[model_choice]
          metric_func = metrics_dict[metric_choice]

          model.fit(x_train, y_train)
          y_pred = model.predict(x_test)

          st.markdown(f"### 🧠 Modèle sélectionné : `{model_choice}`")
          result = metric_func(y_test, y_pred)

          if metric_choice == "R2":
           st.write(f"**Score R² :** `{round(result * 100, 2)} %`")
          else:
               st.write(f"**{metric_choice} :** `{round(result, 2)}`")

          fig, ax = plt.subplots()
          ax.scatter(y_test, y_pred, color="#87CEFA")
          ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='#FFB6C1', linestyle='--')
          ax.set_xlabel("Ventes réelles")
          ax.set_ylabel("Ventes prédites")
          ax.set_title(f"{model_choice} - Réel vs Prédit")
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
            mem_size = df.memory_usage(deep=True).sum() / 1024  
            st.write(f"Taille du dataset : **{mem_size:.2f} KB**")

            st.subheader("🧾 Données brutes")
            st.dataframe(df)
            st.subheader("👀 Affichage personnalisé des lignes du dataset")

            option_affichage = st.selectbox(
                "Choisir l'affichage des lignes :",
                ["5 premières lignes", "5 dernières lignes", "Nombre personnalisé"]
            )

            if option_affichage == "5 premières lignes":
                st.dataframe(df.head())
            elif option_affichage == "5 dernières lignes":
                st.dataframe(df.tail())
            else:
                n = st.number_input("Combien de lignes souhaitez-vous afficher ?", min_value=1, max_value=len(df), value=5)
                st.dataframe(df.head(int(n)))
        elif option == "Data cleaning":
            st.subheader("🔍 Analyse des valeurs manquantes")

            st.write("Voici la **visualisation des valeurs manquantes** dans les 10 premières lignes du dataset :")

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(df.head(10).isnull(), annot=True, cmap='plasma', fmt=".1f", cbar=False, ax=ax)
            st.pyplot(fig)

            total_val_manquantes = df.isnull().sum().sum()
            st.write(f"**Total des valeurs manquantes dans le dataset :** `{int(total_val_manquantes)}`")

            total_val_manquantes_par_col = df.isnull().sum()
            st.write("**Valeurs manquantes par attribut :**")
            st.dataframe(total_val_manquantes_par_col[total_val_manquantes_par_col > 0])
            st.markdown("---")
            st.subheader("📈 Statistiques descriptives")
            st.write("""
            Les statistiques suivantes résument les caractéristiques des données numériques du dataset :
             - **count** : Nombre de valeurs non nulles
             - **mean** : Moyenne
             - **std** : Écart-type
              - **min / max** : Valeurs minimum et maximum
              - **25% / 50% / 75%** : Quartiles
               """)

            
            stats_desc = df.describe().T  
            st.dataframe(stats_desc.style.format(precision=2))
        elif option=="Rapport danalyse":
            rapport()    
        
       


if __name__ == "__main__":
    main()
