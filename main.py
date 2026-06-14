"""
Disciplina de Inteligência Artificial, Professor Munif, Unicesumar 2026
Trabalho Final - Previsão de Diabetes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

os.makedirs("graficos", exist_ok=True)
os.makedirs("modelos", exist_ok=True)

# ──────────────────────────────────────────
# 1. DATASET
# ──────────────────────────────────────────
print("=" * 60)
print("CARREGANDO DATASET...")
print("=" * 60)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
colunas = [
    "Gravidez", "GlicosePlasmatica", "PressaoArterial",
    "EspessuraPeleTricepal", "Insulina", "IMC",
    "FuncaoDiabetesPedigree", "Idade", "Diabetes"
]

try:
    df = pd.read_csv(url, names=colunas)
    print(f"Dataset carregado da internet: {df.shape[0]} registros, {df.shape[1]} colunas")
except Exception:
    # fallback: gera dataset sintético realista
    print("Usando dataset sintético (sem internet)...")
    np.random.seed(42)
    n = 768
    df = pd.DataFrame({
        "Gravidez": np.random.randint(0, 18, n),
        "GlicosePlasmatica": np.random.normal(120, 32, n).clip(0, 199).astype(int),
        "PressaoArterial": np.random.normal(69, 19, n).clip(0, 122).astype(int),
        "EspessuraPeleTricepal": np.random.normal(20, 16, n).clip(0, 99).astype(int),
        "Insulina": np.random.normal(79, 115, n).clip(0, 846).astype(int),
        "IMC": np.round(np.random.normal(32, 8, n).clip(0, 67), 1),
        "FuncaoDiabetesPedigree": np.round(np.random.exponential(0.47, n).clip(0.08, 2.42), 3),
        "Idade": np.random.randint(21, 81, n),
    })
    # variável alvo correlacionada com glicose e IMC
    prob = 1 / (1 + np.exp(-(
        -6
        + 0.04 * df["GlicosePlasmatica"]
        + 0.07 * df["IMC"]
        + 0.01 * df["Idade"]
    )))
    df["Diabetes"] = (np.random.rand(n) < prob).astype(int)

df.to_csv("diabetes.csv", index=False)
print(f"Registros sem diabetes (0): {(df['Diabetes']==0).sum()}")
print(f"Registros com diabetes  (1): {(df['Diabetes']==1).sum()}")

# ──────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────
print("\nPRÉ-PROCESSAMENTO...")

# Substitui zeros inválidos pela mediana
cols_sem_zero = ["GlicosePlasmatica","PressaoArterial","EspessuraPeleTricepal","Insulina","IMC"]
for c in cols_sem_zero:
    mediana = df[c].replace(0, np.nan).median()
    df[c] = df[c].replace(0, mediana)

X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# ──────────────────────────────────────────
# 3. MODELOS
# ──────────────────────────────────────────

# --- PARTE 1: SVM ---
print("\nTREINANDO SVM (Parte 1)...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_sc, y_train)
y_pred_svm = svm.predict(X_test_sc)

with open("modelos/svm_modelo.pkl", "wb") as f:
    pickle.dump(svm, f)
with open("modelos/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# --- PARTE 2: RNA (MLP) ---
print("TREINANDO RNA - Rede Neural (Parte 2)...")
historico_perda_treino = []
historico_perda_val    = []

rna = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1,
    warm_start=True,
    random_state=42,
    learning_rate_init=0.001
)

for epoch in range(150):
    rna.fit(X_train_sc, y_train)
    historico_perda_treino.append(rna.loss_)
    # validação aproximada
    from sklearn.metrics import log_loss
    prob_val = rna.predict_proba(X_test_sc)
    historico_perda_val.append(log_loss(y_test, prob_val))

y_pred_rna = rna.predict(X_test_sc)

with open("modelos/rna_modelo.pkl", "wb") as f:
    pickle.dump(rna, f)

# ──────────────────────────────────────────
# 4. MÉTRICAS
# ──────────────────────────────────────────
def metricas(nome, y_real, y_pred):
    return {
        "Modelo": nome,
        "Acurácia":   round(accuracy_score(y_real, y_pred) * 100, 2),
        "Precisão":   round(precision_score(y_real, y_pred) * 100, 2),
        "Revocação":  round(recall_score(y_real, y_pred) * 100, 2),
        "F1-Score":   round(f1_score(y_real, y_pred) * 100, 2),
    }

m_svm = metricas("SVM (Parte 1)",    y_test, y_pred_svm)
m_rna = metricas("RNA (Parte 2)",    y_test, y_pred_rna)

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)
for m in [m_svm, m_rna]:
    print(f"\n{m['Modelo']}")
    for k, v in m.items():
        if k != "Modelo":
            print(f"  {k}: {v}%")

# ──────────────────────────────────────────
# 5. GRÁFICOS
# ──────────────────────────────────────────
print("\nGERANDO GRÁFICOS...")
COR1, COR2 = "#2196F3", "#FF5722"

# --- Comparação de métricas ---
fig, ax = plt.subplots(figsize=(9, 5))
cats   = ["Acurácia", "Precisão", "Revocação", "F1-Score"]
vals_s = [m_svm[c] for c in cats]
vals_r = [m_rna[c] for c in cats]
x = np.arange(len(cats))
w = 0.35
b1 = ax.bar(x - w/2, vals_s, w, label="SVM (Parte 1)", color=COR1, alpha=0.85)
b2 = ax.bar(x + w/2, vals_r, w, label="RNA (Parte 2)", color=COR2, alpha=0.85)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9)
ax.set_title("Comparação de Métricas – SVM vs RNA", fontsize=13, fontweight='bold')
ax.set_ylabel("Valor (%)")
ax.set_ylim(0, 110)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("graficos/comparacao_metricas.png", dpi=150)
plt.close()

# --- Matrizes de confusão ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, y_pred, titulo, cor in [
    (axes[0], y_pred_svm, "SVM (Parte 1)", COR1),
    (axes[1], y_pred_rna, "RNA (Parte 2)", COR2),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(cor, as_cmap=True),
                xticklabels=["Sem Diabetes","Com Diabetes"],
                yticklabels=["Sem Diabetes","Com Diabetes"])
    ax.set_title(f"Matriz de Confusão – {titulo}", fontweight='bold')
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig("graficos/matrizes_confusao.png", dpi=150)
plt.close()

# --- Curva de perda RNA ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(historico_perda_treino, label="Perda Treino",    color=COR2, linewidth=2)
ax.plot(historico_perda_val,   label="Perda Validação",  color=COR1, linewidth=2, linestyle='--')
ax.set_title("Curva de Perda – RNA (Parte 2)", fontsize=12, fontweight='bold')
ax.set_xlabel("Época"); ax.set_ylabel("Loss (Log Loss)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graficos/curva_perda_rna.png", dpi=150)
plt.close()

# --- Distribuição da variável alvo ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = y.value_counts()
ax.bar(["Sem Diabetes (0)", "Com Diabetes (1)"], counts.values,
       color=[COR1, COR2], alpha=0.85)
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
ax.set_title("Distribuição da Variável Alvo", fontsize=12, fontweight='bold')
ax.set_ylabel("Quantidade"); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("graficos/distribuicao_alvo.png", dpi=150)
plt.close()

print("Gráficos salvos em /graficos/")
print("\nTreinamento concluído com sucesso!")

# Salva métricas para uso no PDF/README
import json
with open("metricas.json", "w") as f:
    json.dump({"svm": m_svm, "rna": m_rna}, f, ensure_ascii=False, indent=2)
