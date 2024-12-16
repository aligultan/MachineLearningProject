from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  # KNN import edildi
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    def __init__(self, n_neighbors=5):  # n_neighbors parametresi eklendi
        self.models = {
            'Lojistik_Regresyon': LogisticRegression(random_state=42, max_iter=1000),
            'DVM': LinearSVC(random_state=42, max_iter=1000),
            'Rastgele_Orman': RandomForestClassifier(random_state=42, n_jobs=-1),
            'KNN': KNeighborsClassifier(n_neighbors=n_neighbors)  # KNN modeli eklendi
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

    def find_best_knn_neighbors(self, X_train, X_test, y_train, y_test):
        """En iyi KNN komşu sayısını bulur"""
        from sklearn.metrics import accuracy_score

        best_accuracy = 0
        best_neighbors = 1

        for n in range(1, 21):  # 1-20 arası komşu sayısı denenecek
            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_neighbors = n

        print(f"En iyi komşu sayısı: {best_neighbors} (Doğruluk: {best_accuracy:.4f})")
        return best_neighbors