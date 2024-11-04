import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = './aerogerador.dat'
data = pd.read_csv(file_path, sep='\t', header=None)
data.columns = ['Velocidade_Vento', 'Potencia_Gerada']

X = data['Velocidade_Vento'].values
y = data['Potencia_Gerada'].values

def regressao_polinomial(X, y):
    X_poly = np.c_[np.ones(X.shape[0]), X, X**2, X**3]
    beta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    return beta

beta_poly = regressao_polinomial(X, y)
print(f"Coeficientes do Modelo Polinomial: Intercepto={beta_poly[0]}, x={beta_poly[1]}, x^2={beta_poly[2]}, x^3={beta_poly[3]}")

def predict_polinomial(X, beta):
    return beta[0] + beta[1] * X + beta[2] * X**2 + beta[3] * X**3

def calculate_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def monte_carlo_polinomial(X, y, R=1000):
    rss_list = []
    for _ in range(R):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        train_size = int(len(X) * 0.8)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        beta_poly = regressao_polinomial(X_train, y_train)

        y_pred = predict_polinomial(X_test, beta_poly)

        rss = calculate_rss(y_test, y_pred)
        rss_list.append(rss)

    metrics = {
        'Média': np.mean(rss_list),
        'Desvio-Padrão': np.std(rss_list),
        'Maior Valor': np.max(rss_list),
        'Menor Valor': np.min(rss_list)
    }
    return metrics

results_polinomial = monte_carlo_polinomial(X, y)
print("Resultados da Regressão Polinomial após 1000 rodadas de Monte Carlo:")
print(f"Média do RSS: {results_polinomial['Média']}")
print(f"Desvio-Padrão do RSS: {results_polinomial['Desvio-Padrão']}")
print(f"Maior Valor do RSS: {results_polinomial['Maior Valor']}")
print(f"Menor Valor do RSS: {results_polinomial['Menor Valor']}")


results_polinomial_df = pd.DataFrame([results_polinomial], columns=['Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor'])
results_polinomial_df.index = ['Regressão Polinomial (Grau 3)']

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=results_polinomial_df.values,
    colLabels=results_polinomial_df.columns,
    rowLabels=results_polinomial_df.index,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(results_polinomial_df.columns))))
plt.title("Resultados da Validação com Monte Carlo - Regressão Polinomial (Grau 3)")
plt.show()

velocidade_vento_plot = np.linspace(X.min(), X.max(), 100)
polynomial_predictions = predict_polinomial(velocidade_vento_plot, beta_poly)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Dados")
plt.plot(velocidade_vento_plot, polynomial_predictions, color="red", label="Regressão Polinomial (Grau 3)")
plt.xlabel("Velocidade do Vento")
plt.ylabel("Potência Gerada")
plt.title("Relação entre Velocidade do Vento e Potência Gerada com Regressão Polinomial")
plt.legend()
plt.show()