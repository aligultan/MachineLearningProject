from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def perform_cross_validation(models, X, y, cv=5):
    # Kategorik değişkenleri dönüştür
    X_encoded = pd.get_dummies(X)

    metrics = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    results = {}
    for name, model in models.items():
        model_scores = {}
        for metric_name, scorer in metrics.items():
            scores = cross_val_score(model, X_encoded, y, cv=cv, scoring=scorer)
            model_scores[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
        results[name] = model_scores

        print(f"\n{name} Cross-Validation Sonuçları:")
        for metric, scores in model_scores.items():
            print(f"{metric}: {scores['mean']:.4f} (+/- {scores['std'] * 2:.4f})")

    plot_cv_results(results)
    return results


def plot_cv_results(results):
    # Grafik kodu aynı kalacak
    metrics = list(next(iter(results.values())).keys())
    models = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        means = [results[model][metric]['mean'] for model in models]
        stds = [results[model][metric]['std'] for model in models]

        ax = axes[idx]
        x = np.arange(len(models))
        ax.bar(x, means, yerr=stds, capsize=5)
        ax.set_title(f'{metric.capitalize()} Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)

    plt.tight_layout()
    plt.savefig('data/cross_validation_results.png')
    plt.close()