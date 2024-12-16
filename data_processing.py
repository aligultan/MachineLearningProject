import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def analyze_data(self):
        print("Veri Seti Bilgisi:")
        print(self.df.info())
        print("\nİlk 5 satır:")
        print(self.df.head())
        print("\nEksik değerler:")
        print(self.df.isnull().sum())

        plt.figure(figsize=(10, 6))
        self.df['HeartDisease'].value_counts().plot(kind='bar')
        plt.title('Kalp Hastalığı Dağılımı')
        plt.xlabel('Kalp Hastalığı')
        plt.ylabel('Sayı')
        plt.savefig('data/kalp_hastaligi_dagilimi.png')
        plt.close()

    def preprocess_data(self):
        # HeartDisease sütununu binary'ye çevir
        self.df['HeartDisease'] = (self.df['HeartDisease'] == 'Yes').astype(int)

        # Hedef değişkeni ayır
        y = self.df['HeartDisease']
        X = self.df.drop('HeartDisease', axis=1)

        # Kategorik değişkenleri dönüştür
        kategorik_kolonlar = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=kategorik_kolonlar)

        # Veriyi böl
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Verileri ölçeklendir
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        return self.X_train, self.X_test, self.y_train, self.y_test