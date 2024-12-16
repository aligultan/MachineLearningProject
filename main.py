from data_processing import DataProcessor
from model_training import ModelTrainer
from data_visualization import DataAnalyzer, ModelAnalyzer
from sklearn.metrics import accuracy_score
from cross_validation import perform_cross_validation


def main():
    # Veri analizi ve görselleştirme
    analyzer = DataAnalyzer('data/heart_2020_cleaned.csv')
    df_encoded = analyzer.analyze_features()

    # Veri ön işleme
    processor = DataProcessor('data/heart_2020_cleaned.csv')
    X_train, X_test, y_train, y_test = processor.preprocess_data()

    # Model eğitimi
    trainer = ModelTrainer()
    models = trainer.train_evaluate_all(X_train, X_test, y_train, y_test)

    # Model performans analizi
    model_analyzer = ModelAnalyzer()
    for name, model in trainer.models.items():
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"\n{name} Performans Metrikleri:")
        print(f"Eğitim Doğruluğu: {train_accuracy:.4f}")
        print(f"Test Doğruluğu: {test_accuracy:.4f}")
        model_analyzer.plot_confusion_matrix(y_test, model.predict(X_test), name)

    # Cross-validation
    print("\nCross-validation başlıyor...")
    X = processor.df.drop('HeartDisease', axis=1)
    y = processor.df['HeartDisease']
    cv_results = perform_cross_validation(trainer.models, X, y)


if __name__ == "__main__":
    main()