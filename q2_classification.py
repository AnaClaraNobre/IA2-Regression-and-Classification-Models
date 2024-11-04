import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = './EMGsDataset.csv'
data = pd.read_csv(file_path, header=None)

print("Amostra dos dados carregados:")
print(data.head())

sensor_1 = data.iloc[0, :].values
sensor_2 = data.iloc[1, :].values  
labels = data.iloc[2, :].values.astype(int)  


X_mqo = np.column_stack((sensor_1, sensor_2))

num_classes = 5
Y_mqo = np.zeros((labels.size, num_classes))
Y_mqo[np.arange(labels.size), labels - 1] = 1 

X_gaussian = np.vstack((sensor_1, sensor_2))

Y_gaussian = labels

colors = ['blue', 'green', 'red', 'purple', 'orange']
class_names = ['Neutro', 'Sorriso', 'Sobrancelhas levantadas', 'Surpreso', 'Rabugento']

plt.figure(figsize=(10, 6))

for class_id in range(1, 6):
    plt.scatter(
        sensor_1[labels == class_id],
        sensor_2[labels == class_id],
        c=colors[class_id - 1],
        label=class_names[class_id - 1],
        alpha=0.6,
        edgecolors='k'
    )

plt.xlabel('Sensor 1 (Corrugador do Supercílio)')
plt.ylabel('Sensor 2 (Zigomático Maior)')
plt.title('Distribuição dos Sinais Eletromiográficos por Classe')
plt.legend(title='Expressões Faciais')
plt.show()

def calculate_mean_covariance(X, y, target_class):
    class_data = X[y == target_class]
    mean = np.mean(class_data, axis=0)
    covariance = np.cov(class_data, rowvar=False)
    return mean, covariance

# Traditional Least Squares (MQO) - For reference, not used as a classifier
def traditional_least_squares(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return beta

class TraditionalGaussianClassifier:
    def __init__(self):
        self.means = {}
        self.covariances = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for target_class in self.classes:
            mean, covariance = calculate_mean_covariance(X, y, target_class)
            self.means[target_class] = mean
            self.covariances[target_class] = covariance

    def predict_probability(self, x, mean, covariance):
        covariance += np.eye(covariance.shape[0]) * 1e-6
        n = len(x)
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        constant = 1 / (np.sqrt((2 * np.pi) ** n * det))
        return constant * np.exp(-0.5 * (x - mean).T @ inv @ (x - mean))

    def predict(self, X):
        predictions = []
        for x in X:
            probabilities = {target_class: self.predict_probability(x, self.means[target_class], self.covariances[target_class]) for target_class in self.classes}
            predictions.append(max(probabilities, key=probabilities.get))
        return np.array(predictions)

class EqualCovarianceGaussianClassifier:
    def __init__(self):
        self.means = {}
        self.common_covariance = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        covariances = []
        for target_class in self.classes:
            mean, covariance = calculate_mean_covariance(X, y, target_class)
            self.means[target_class] = mean
            covariances.append(covariance)
        self.common_covariance = np.mean(covariances, axis=0)

    def predict_probability(self, x, mean, covariance):
        covariance += np.eye(covariance.shape[0]) * 1e-6
        n = len(x)
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        constant = 1 / (np.sqrt((2 * np.pi) ** n * det))
        return constant * np.exp(-0.5 * (x - mean).T @ inv @ (x - mean))

    def predict(self, X):
        predictions = []
        for x in X:
            probabilities = {target_class: self.predict_probability(x, self.means[target_class], self.common_covariance) for target_class in self.classes}
            predictions.append(max(probabilities, key=probabilities.get))
        return np.array(predictions)

class RegularizedGaussianClassifier:
    def __init__(self, lambda_):
        self.means = {}
        self.covariances = {}
        self.lambda_ = lambda_
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for target_class in self.classes:
            mean, covariance = calculate_mean_covariance(X, y, target_class)
            regularized_covariance = covariance + self.lambda_ * np.eye(covariance.shape[0])
            self.means[target_class] = mean
            self.covariances[target_class] = regularized_covariance

    def predict_probability(self, x, mean, covariance):
        covariance += np.eye(covariance.shape[0]) * 1e-6
        n = len(x)
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        constant = 1 / (np.sqrt((2 * np.pi) ** n * det))
        return constant * np.exp(-0.5 * (x - mean).T @ inv @ (x - mean))

    def predict(self, X):
        predictions = []
        for x in X:
            probabilities = {target_class: self.predict_probability(x, self.means[target_class], self.covariances[target_class]) for target_class in self.classes}
            predictions.append(max(probabilities, key=probabilities.get))
        return np.array(predictions)

class NaiveBayesClassifier:
    def __init__(self):
        self.means = {}
        self.variances = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for target_class in self.classes:
            class_data = X[y == target_class]
            mean = np.mean(class_data, axis=0)
            variance = np.var(class_data, axis=0) + 1e-6  
            self.means[target_class] = mean
            self.variances[target_class] = variance

    def predict_probability(self, x, mean, variance):
        constant = 1 / np.sqrt(2 * np.pi * variance)
        prob = constant * np.exp(-0.5 * ((x - mean) ** 2 / variance))
        return np.prod(prob)

    def predict(self, X):
        predictions = []
        for x in X:
            probabilities = {target_class: self.predict_probability(x, self.means[target_class], self.variances[target_class]) for target_class in self.classes}
            predictions.append(max(probabilities, key=probabilities.get))
        return np.array(predictions)

lambdas = [0, 0.25, 0.5, 0.75, 1]
print("Accuracy Results for Each Model:")

model_traditional = TraditionalGaussianClassifier()
model_traditional.fit(X_gaussian.T, Y_gaussian)
y_pred_traditional = model_traditional.predict(X_gaussian.T)
accuracy_traditional = np.mean(y_pred_traditional == Y_gaussian)
print(f"Traditional Gaussian Classifier: Accuracy = {accuracy_traditional:.2f}")

model_equal_cov = EqualCovarianceGaussianClassifier()
model_equal_cov.fit(X_gaussian.T, Y_gaussian)
y_pred_equal_cov = model_equal_cov.predict(X_gaussian.T)
accuracy_equal_cov = np.mean(y_pred_equal_cov == Y_gaussian)
print(f"Equal Covariance Gaussian Classifier: Accuracy = {accuracy_equal_cov:.2f}")

model_naive_bayes = NaiveBayesClassifier()
model_naive_bayes.fit(X_gaussian.T, Y_gaussian)
y_pred_naive_bayes = model_naive_bayes.predict(X_gaussian.T)
accuracy_naive_bayes = np.mean(y_pred_naive_bayes == Y_gaussian)
print(f"Naive Bayes Classifier: Accuracy = {accuracy_naive_bayes:.2f}")

for lambda_value in lambdas:
    model_regularized = RegularizedGaussianClassifier(lambda_=lambda_value)
    model_regularized.fit(X_gaussian.T, Y_gaussian)
    y_pred_regularized = model_regularized.predict(X_gaussian.T)
    accuracy_regularized = np.mean(y_pred_regularized == Y_gaussian)
    print(f"Regularized Gaussian Classifier (lambda={lambda_value}): Accuracy = {accuracy_regularized:.2f}")

def monte_carlo_simulation(models, X, y, R=500):
    results = {name: [] for name in models.keys()}
    
    for round_num in range(R):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        train_size = int(len(X) * 0.8)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        print(f"--- Round {round_num + 1} ---")
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            results[name].append(accuracy)
            
            print(f"Modelo: {name}, Acurácia na rodada {round_num + 1}: {accuracy:.4f}")
    
    return results

models = {
    'Traditional Gaussian Classifier': TraditionalGaussianClassifier(),
    'Equal Covariance Gaussian Classifier': EqualCovarianceGaussianClassifier(),
    'Naive Bayes Classifier': NaiveBayesClassifier(),
    'Regularized Gaussian Classifier (lambda=0.25)': RegularizedGaussianClassifier(lambda_=0.25),
    'Regularized Gaussian Classifier (lambda=0.5)': RegularizedGaussianClassifier(lambda_=0.5),
    'Regularized Gaussian Classifier (lambda=0.75)': RegularizedGaussianClassifier(lambda_=0.75)
}

results = monte_carlo_simulation(models, X_gaussian.T, Y_gaussian)

metrics = []
for model_name, accuracies in results.items():
    metrics.append({
        'Model': model_name,
        'Mean': np.mean(accuracies),
        'Std Dev': np.std(accuracies),
        'Max': np.max(accuracies),
        'Min': np.min(accuracies)
    })

metrics_df = pd.DataFrame(metrics)

print("Tabela de Resultados de Acurácia:")
print(metrics_df)

fig, ax = plt.subplots(figsize=(12, 8))  
ax.axis('tight')
ax.axis('off')
ax.table(cellText=metrics_df.values,
         colLabels=metrics_df.columns,
         cellLoc = 'center', 
         loc='center')
plt.title('Tabela de Resultados de Acurácia', fontsize=14)
plt.savefig('tabela_resultados_acuracia.png', bbox_inches='tight') 
plt.show()