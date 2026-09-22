"""Educational binary logistic regression with deterministic training snapshots."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegressionAlgorithm:
    """Batch gradient descent implementation used by the interactive lab.

    The implementation stays deliberately small so every value shown in the UI
    comes from the same sigmoid, cross entropy and gradient update taught on the
    page.  Features are standardized for stable training; coefficients returned
    to the browser are converted back to the original data coordinates.
    """

    @staticmethod
    def _sigmoid(scores):
        scores = np.clip(scores, -500.0, 500.0)
        return np.where(
            scores >= 0,
            1.0 / (1.0 + np.exp(-scores)),
            np.exp(scores) / (1.0 + np.exp(scores)),
        )

    @staticmethod
    def _loss(y, probability, weights, regularization):
        probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
        cross_entropy = -np.mean(
            y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)
        )
        return float(cross_entropy + 0.5 * regularization * np.dot(weights, weights))

    @staticmethod
    def _original_coefficients(weights, bias, scaler):
        coefficients = weights / scaler.scale_
        intercept = bias - np.dot(weights, scaler.mean_ / scaler.scale_)
        return coefficients.astype(float), float(intercept)

    @staticmethod
    def _classification_metrics(y, probability, threshold):
        prediction = (probability >= threshold).astype(int)
        true_positive = int(np.sum((prediction == 1) & (y == 1)))
        true_negative = int(np.sum((prediction == 0) & (y == 0)))
        false_positive = int(np.sum((prediction == 1) & (y == 0)))
        false_negative = int(np.sum((prediction == 0) & (y == 1)))
        total = max(1, len(y))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        return {
            'accuracy': float((true_positive + true_negative) / total),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(2 * precision * recall / max(1e-12, precision + recall)),
            'confusion_matrix': [[true_negative, false_positive], [false_negative, true_positive]],
        }

    def train(self, X, y, epochs=400, learning_rate=0.12, regularization=0.01,
              test_size=0.25, random_state=42, checkpoint_count=40):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError('逻辑回归实验室需要恰好两个特征')
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError('逻辑回归实验室仅支持标签为 0 和 1 的二分类数据')
        if len(X) < 20:
            raise ValueError('至少需要 20 个样本')

        indices = np.arange(len(X))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        weights = np.zeros(2, dtype=float)
        bias = 0.0
        epochs = int(epochs)
        learning_rate = float(learning_rate)
        regularization = float(regularization)
        checkpoint_every = max(1, int(np.ceil(epochs / max(1, checkpoint_count))))
        history = []
        checkpoints = []

        def record(epoch):
            train_probability = self._sigmoid(X_train_scaled @ weights + bias)
            test_probability = self._sigmoid(X_test_scaled @ weights + bias)
            train_loss = self._loss(y_train, train_probability, weights, regularization)
            test_loss = self._loss(y_test, test_probability, weights, regularization)
            coefficients, intercept = self._original_coefficients(weights, bias, scaler)
            metrics = self._classification_metrics(y_test, test_probability, 0.5)
            snapshot = {
                'epoch': int(epoch),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'accuracy': metrics['accuracy'],
                'coefficients': coefficients.tolist(),
                'intercept': intercept,
            }
            checkpoints.append(snapshot)
            return train_loss, test_loss

        initial_train_loss, initial_test_loss = record(0)
        history.append({
            'epoch': 0,
            'train_loss': initial_train_loss,
            'test_loss': initial_test_loss,
        })
        for epoch in range(1, epochs + 1):
            probability = self._sigmoid(X_train_scaled @ weights + bias)
            error = probability - y_train
            gradient_weights = X_train_scaled.T @ error / len(y_train) + regularization * weights
            gradient_bias = float(np.mean(error))
            weights -= learning_rate * gradient_weights
            bias -= learning_rate * gradient_bias

            train_probability = self._sigmoid(X_train_scaled @ weights + bias)
            test_probability = self._sigmoid(X_test_scaled @ weights + bias)
            history.append({
                'epoch': epoch,
                'train_loss': self._loss(y_train, train_probability, weights, regularization),
                'test_loss': self._loss(y_test, test_probability, weights, regularization),
            })
            if epoch % checkpoint_every == 0 or epoch == epochs:
                record(epoch)

        train_probability = self._sigmoid(X_train_scaled @ weights + bias)
        test_probability = self._sigmoid(X_test_scaled @ weights + bias)
        metrics = self._classification_metrics(y_test, test_probability, 0.5)
        metrics['auc'] = float(roc_auc_score(y_test, test_probability))
        coefficients, intercept = self._original_coefficients(weights, bias, scaler)

        x_margin = max(0.5, float(np.ptp(X[:, 0])) * 0.08)
        y_margin = max(0.5, float(np.ptp(X[:, 1])) * 0.08)
        bounds = {
            'x': [float(X[:, 0].min() - x_margin), float(X[:, 0].max() + x_margin)],
            'y': [float(X[:, 1].min() - y_margin), float(X[:, 1].max() + y_margin)],
        }
        return {
            'data': {
                'X': X.tolist(),
                'y': y.tolist(),
                'train_indices': train_indices.tolist(),
                'test_indices': test_indices.tolist(),
                'test_probabilities': test_probability.tolist(),
            },
            'history': history,
            'checkpoints': checkpoints,
            'model': {
                'coefficients': coefficients.tolist(),
                'intercept': intercept,
            },
            'metrics': metrics,
            'bounds': bounds,
            'params': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'regularization': regularization,
                'test_size': float(test_size),
            },
        }
