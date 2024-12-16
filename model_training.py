from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    def __init__(self):
        self.models = {
            'Lojistik_Regresyon': LogisticRegression(random_state=42, max_iter=1000),
            'DVM': LinearSVC(random_state=42, max_iter=1000),
            'Rastgele_Orman': RandomForestClassifier(random_state=42, n_jobs=-1)
        }

    def train_evaluate_all(self, X_train, X_test, y_train, y_test):
        results = {}
        for name, model in self.models.items():
            print(f"\n{name} eğitiliyor...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            self._plot_confusion_matrix(y_test, y_pred, name)
            print(f"\n{name} Sınıflandırma Raporu:")
            print(classification_report(y_test, y_pred))
            results[name] = y_pred
        return results

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Karmaşıklık Matrisi - {model_name}')
        plt.ylabel('Gerçek Değer')
        plt.xlabel('Tahmin')
        plt.savefig(f'data/karmasiklik_matrisi_{model_name}.png')
        plt.close()