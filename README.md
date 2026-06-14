Disciplina de Inteligência Artificial, Professor Munif, Unicesumar 2026

# Previsão de Diabetes com SVM e Redes Neurais

## Integrantes
- Nome do Aluno 1 - RA: XXXXXXXX
- Nome do Aluno 2 - RA: XXXXXXXX
- Nome do Aluno 3 - RA: XXXXXXXX
- Nome do Aluno 4 - RA: XXXXXXXX

---

## Resumo do Projeto

### Contextualização
O diabetes mellitus é uma das doenças crônicas de maior prevalência no mundo, afetando milhões de pessoas. O diagnóstico precoce é fundamental para evitar complicações graves como problemas cardiovasculares, renais e neurológicos. Técnicas de Inteligência Artificial têm se mostrado eficazes no auxílio ao diagnóstico por meio da análise de dados clínicos.

### Problema
Este trabalho investiga a seguinte questão: **é possível prever, a partir de dados clínicos de uma paciente, se ela possui ou virá a desenvolver diabetes?**

### Hipótese
Acreditamos que modelos de IA treinados com dados clínicos como glicose plasmática, IMC, idade e histórico familiar são capazes de classificar corretamente pacientes diabéticas com acurácia superior a 70%, e que a Rede Neural tende a superar o SVM por sua maior capacidade de aprender padrões não-lineares complexos.

### Dataset
Utilizamos o **Pima Indians Diabetes Dataset**, amplamente usado em pesquisas de Machine Learning.

- **Origem:** UCI Machine Learning Repository / Kaggle
- **Registros:** 768 amostras de pacientes do sexo feminino com idade ≥ 21 anos
- **Atributos:** 8 variáveis de entrada + 1 variável alvo
- **Variável alvo:** `Diabetes` — 0 (não diabética) ou 1 (diabética)
- **Distribuição:** 500 negativos (65,1%) e 268 positivos (34,9%)

#### Atributos principais

| Atributo | Descrição |
|---|---|
| Gravidez | Número de gestações |
| GlicosePlasmatica | Concentração de glicose no plasma (mg/dL) |
| PressaoArterial | Pressão diastólica (mm Hg) |
| EspessuraPeleTricepal | Espessura da dobra cutânea tricipital (mm) |
| Insulina | Nível de insulina sérica (µU/ml) |
| IMC | Índice de Massa Corporal (kg/m²) |
| FuncaoDiabetesPedigree | Função que pondera histórico familiar |
| Idade | Idade em anos |

#### Preparação dos dados
- Valores zerados em colunas clínicas (glicose, pressão, IMC, etc.) foram substituídos pela mediana, pois zero não é biologicamente plausível.
- Divisão: **80% treino** / **20% teste**, com estratificação pela variável alvo.
- Normalização via **StandardScaler** (média 0, desvio padrão 1) antes de alimentar os modelos.

### Métodos utilizados

#### Parte 1 — SVM (Support Vector Machine)
Método de aprendizado supervisionado que busca o hiperplano de máxima margem entre as classes. Utilizado com kernel RBF (Radial Basis Function) para capturar relações não-lineares nos dados.

**Parâmetros:** `kernel='rbf'`, `C=10`, `gamma='scale'`

#### Parte 2 — RNA (Rede Neural Artificial / MLP)
Rede neural do tipo Perceptron Multicamadas com duas camadas ocultas (64 e 32 neurônios), função de ativação ReLU e otimizador Adam. Treinada por 150 épocas com validação cruzada.

**Arquitetura:** `Input(8) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1, Sigmoid)`

---

## Avaliação dos Modelos

### Métricas

| Métrica | SVM (Parte 1) | RNA (Parte 2) |
|---|---|---|
| Acurácia | 71,43% | 74,03% |
| Precisão | 59,62% | 65,22% |
| Revocação | 57,41% | 55,56% |
| F1-Score | 58,49% | 60,00% |

### Gráfico de Comparação

![Comparação de Métricas](graficos/comparacao_metricas.png)

### Matrizes de Confusão

![Matrizes de Confusão](graficos/matrizes_confusao.png)

### Curva de Perda da RNA

![Curva de Perda](graficos/curva_perda_rna.png)

### Distribuição da Variável Alvo

![Distribuição](graficos/distribuicao_alvo.png)

---

## Comparação e Conclusão

### Comparação dos resultados
A RNA obteve desempenho superior ao SVM em todas as métricas avaliadas. A maior acurácia da RNA (74,03% vs 71,43%) e precisão (65,22% vs 59,62%) indicam que a capacidade de aprendizado em múltiplas camadas permitiu capturar padrões mais complexos nos dados clínicos.

Ambos os modelos enfrentaram dificuldade com a classe positiva (diabéticas), o que é esperado dado o desbalanceamento do dataset (65%/35%). A revocação relativamente baixa (~56-57%) para a classe positiva indica que há casos de diabetes não identificados — ponto que poderia ser melhorado com técnicas de balanceamento como SMOTE.

### Conclusão
A hipótese foi parcialmente confirmada: os modelos atingiram acurácia superior a 70%, com a RNA superando o SVM em todas as métricas, conforme esperado. O projeto demonstrou de forma prática o pipeline completo de uma solução de IA: coleta e preparação de dados, treinamento, avaliação com métricas e gráficos, e comparação entre modelos de diferentes paradigmas (método clássico vs. rede neural).

---

## Como executar

### Pré-requisitos
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Treinamento
```bash
python main.py
```
O script baixa o dataset automaticamente, treina os modelos, gera os gráficos em `graficos/` e salva os modelos em `modelos/`.

### Dataset
O dataset é baixado automaticamente do repositório público:
`https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`

Também está disponível no próprio repositório como `diabetes.csv`.

### Modelos treinados
Os modelos são salvos automaticamente em `modelos/`:
- `modelos/svm_modelo.pkl` — modelo SVM treinado
- `modelos/rna_modelo.pkl` — modelo RNA treinado
- `modelos/scaler.pkl` — normalizador StandardScaler

---

## Estrutura do repositório
```
├── main.py                  # Script principal de treinamento e avaliação
├── diabetes.csv             # Dataset utilizado
├── README.md                # Este arquivo
├── relatorio.pdf            # PDF com o conteúdo do README
├── graficos/
│   ├── comparacao_metricas.png
│   ├── matrizes_confusao.png
│   ├── curva_perda_rna.png
│   └── distribuicao_alvo.png
└── modelos/
    ├── svm_modelo.pkl
    ├── rna_modelo.pkl
    └── scaler.pkl
```
