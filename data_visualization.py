import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc


class DataAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

    def analyze_features(self):
        """Veri setindeki her özellik için detaylı analiz ve görselleştirme"""

        # Sayısal değişkenlerin dağılımı
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for i, col in enumerate(numerical_cols):
            row = i // 2
            col_idx = i % 2
            sns.histplot(data=self.df, x=col, hue='HeartDisease',
                         multiple="dodge", ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'{col} Dağılımı')
        plt.tight_layout()
        plt.savefig('data/numerical_features_dist.png')
        plt.close()

        # Kategorik değişkenlerin analizi
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.df, x=col, hue='HeartDisease')
            plt.xticks(rotation=45)
            plt.title(f'{col} Değişkeni için Kalp Hastalığı Dağılımı')
            plt.tight_layout()
            plt.savefig(f'data/categorical_{col}_dist.png')
            plt.close()

        # Korelasyon matrisi
        # Kategorik değişkenleri sayısala çevirme
        df_encoded = self.df.copy()
        le = LabelEncoder()
        for col in categorical_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col])

        plt.figure(figsize=(12, 10))
        sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Özellikler Arası Korelasyon Matrisi')
        plt.tight_layout()
        plt.savefig('data/correlation_matrix.png')
        plt.close()

        return df_encoded


class ModelAnalyzer:
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negatif', 'Pozitif'],
                    yticklabels=['Negatif', 'Pozitif'])
        plt.title(f'Karmaşıklık Matrisi - {model_name}')
        plt.ylabel('Gerçek Değer')
        plt.xlabel('Tahmin')
        plt.tight_layout()
        plt.savefig(f'data/confusion_matrix_{model_name}.png')
        plt.close()

    def plot_roc_curves(self, models, X_test, y_test):
        plt.figure(figsize=(10, 8))
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Yanlış Pozitif Oranı')
        plt.ylabel('Doğru Pozitif Oranı')
        plt.title('ROC Eğrileri')
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/roc_curves.png')
        plt.close()

    def plot_feature_importance(self, model, feature_names):
        importances = pd.DataFrame({
            'özellik': feature_names,
            'önem': model.feature_importances_
        }).sort_values('önem', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=importances.head(10), x='önem', y='özellik')
        plt.title('En Önemli 10 Özellik')
        plt.tight_layout()
        plt.savefig('data/feature_importance.png')
        plt.close()