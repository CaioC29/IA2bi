# -*- coding: utf-8 -*-
"""Copy of Aula009-RNA.ipynb - VERSÃO MODIFICADA

Versão com parâmetros ajustados para melhores resultados
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1])
y_or  = np.array([0, 1, 1, 1])
y_xor = np.array([0, 1, 1, 0])

def testar_modelo(nome, modelo, X, y):
    modelo.fit(X, y)
    pred = modelo.predict(X)

    tabela = pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "esperado": y,
        "previsto": pred
    })

    print(f"\n=== {nome} ===")
    print(tabela)
    print(f"Acurácia: {accuracy_score(y, pred):.2f}")

    return modelo

# PERCEPTRON - Parâmetros modificados
# Aumentei eta0 (taxa de aprendizado) de 0.1 para 0.2
# Aumentei max_iter de 20 para 100

perceptron_and = Perceptron(
    max_iter=100,          # modificado de 20
    eta0=0.2,             # modificado de 0.1
    tol=None,
    shuffle=False,
    random_state=42
)

testar_modelo("Perceptron - Porta AND", perceptron_and, X, y_and)

perceptron_or = Perceptron(
    max_iter=100,          # modificado de 20
    eta0=0.2,             # modificado de 0.1
    tol=None,
    shuffle=False,
    random_state=42
)

testar_modelo("Perceptron - Porta OR", perceptron_or, X, y_or)

perceptron_xor = Perceptron(
    max_iter=100,          # modificado de 20
    eta0=0.2,             # modificado de 0.1
    tol=None,
    shuffle=False,
    random_state=42
)

testar_modelo("Perceptron - Porta XOR", perceptron_xor, X, y_xor)

# MLP - Parâmetros significativamente modificados
# Aumentei hidden_layer_sizes de (4,) para (8, 6)
# Mudei activation de 'logistic' para 'relu'
# Mudei solver de 'lbfgs' para 'adam'
# Aumentei max_iter de 10000 para 50000

mlp_xor = MLPClassifier(
    hidden_layer_sizes=(8, 6),    # modificado de (4,)
    activation="relu",             # modificado de 'logistic'
    solver="adam",                # modificado de 'lbfgs'
    max_iter=50000,               # modificado de 10000
    learning_rate_init=0.01,      # novo parâmetro para controlar aprendizado
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

testar_modelo("MLP - Porta XOR", mlp_xor, X, y_xor)

modelos = [
    ("Perceptron AND", perceptron_and, y_and),
    ("Perceptron OR", perceptron_or, y_or),
    ("Perceptron XOR", perceptron_xor, y_xor),
    ("MLP XOR", mlp_xor, y_xor)
]

resultados = []

for nome, modelo, y in modelos:
    pred = modelo.predict(X)
    acc = accuracy_score(y, pred)
    resultados.append([nome, acc])

print("\n=== RESUMO DE RESULTADOS ===")
print(pd.DataFrame(resultados, columns=["Modelo", "Acurácia"]))

print("\n=== Arquitetura da rede MLP ===")
print("Número de camadas:", mlp_xor.n_layers_)
print("Número de entradas:", mlp_xor.n_features_in_)
print("Camadas ocultas:", mlp_xor.hidden_layer_sizes)
print("Número de saídas:", mlp_xor.n_outputs_)

print("\n=== Pesos e bias por camada ===")

for i, (W, b) in enumerate(zip(mlp_xor.coefs_, mlp_xor.intercepts_)):
    print(f"\nCamada {i} -> {i+1}")
    print(f"Formato da matriz de pesos: {W.shape}")
    print(W)
    print(f"Formato do vetor de bias: {b.shape}")
    print(b)

print("\n=== Análise detalhada dos pesos ===")

W1 = mlp_xor.coefs_[0]       # entrada -> oculta
b1 = mlp_xor.intercepts_[0]  # bias da camada oculta

df_W1 = pd.DataFrame(
    W1,
    index=["x1", "x2"],
    columns=[f"h{i+1}" for i in range(W1.shape[1])]
)

df_b1 = pd.DataFrame(
    [b1],
    index=["bias"],
    columns=[f"h{i+1}" for i in range(len(b1))]
)

print("\nPesos da entrada para a camada oculta:")
print(df_W1)

print("\nBias da camada oculta:")
print(df_b1)

W2 = mlp_xor.coefs_[1]       # oculta -> saída
b2 = mlp_xor.intercepts_[1]  # bias da saída

df_W2 = pd.DataFrame(
    W2,
    index=[f"h{i+1}" for i in range(W2.shape[0])],
    columns=["y"]
)

df_b2 = pd.DataFrame(
    [b2],
    index=["bias"],
    columns=["y"]
)

print("\nPesos da camada oculta para a saída:")
print(df_W2)

print("\nBias da camada de saída:")
print(df_b2)

# Se houver terceira camada oculta
if len(mlp_xor.coefs_) > 2:
    W3 = mlp_xor.coefs_[2]
    b3 = mlp_xor.intercepts_[2]
    
    df_W3 = pd.DataFrame(
        W3,
        index=[f"h{i+1}" for i in range(W3.shape[0])],
        columns=["y"]
    )
    
    print("\nPesos da segunda camada oculta para a saída:")
    print(df_W3)

print("\n=== Probabilidades preditas ===")

probs = mlp_xor.predict_proba(X)

print("\nx1 x2 | classe esperada | classe prevista | probabilidades")
print("----------------------------------------------------------")

for entrada, esperado, previsto, prob in zip(X, y_xor, mlp_xor.predict(X), probs):
    print(f"{entrada[0]}  {entrada[1]}  |       {esperado}        |        {previsto}       | {prob}")

print("\n=== Número de iterações realizadas ===")
print(f"MLP - Iterações: {mlp_xor.n_iter_}")
print(f"MLP - Função de perda: {mlp_xor.loss_:.6f}")
