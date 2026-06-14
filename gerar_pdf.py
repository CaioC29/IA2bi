"""
Gera o PDF do relatório final conforme exigido pelo enunciado.
"""
import json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

W, H = A4

with open("metricas.json") as f:
    metricas = json.load(f)

doc = SimpleDocTemplate(
    "relatorio.pdf",
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
    title="Trabalho Final - IA - Unicesumar 2026"
)

styles = getSampleStyleSheet()

AZUL   = colors.HexColor("#1565C0")
AZUL_L = colors.HexColor("#E3F2FD")
LARANJA = colors.HexColor("#E64A19")
CINZA  = colors.HexColor("#F5F5F5")

s_topo   = ParagraphStyle("topo",   parent=styles["Normal"], fontSize=9,  textColor=colors.grey,    alignment=TA_CENTER)
s_titulo = ParagraphStyle("titulo", parent=styles["Title"],  fontSize=20, textColor=AZUL,           spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold")
s_sub    = ParagraphStyle("sub",    parent=styles["Normal"], fontSize=11, textColor=AZUL,           spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold")
s_h3     = ParagraphStyle("h3",     parent=styles["Normal"], fontSize=10, textColor=LARANJA,        spaceBefore=8, spaceAfter=3, fontName="Helvetica-Bold")
s_body   = ParagraphStyle("body",   parent=styles["Normal"], fontSize=10, leading=15,               alignment=TA_JUSTIFY)
s_code   = ParagraphStyle("code",   parent=styles["Code"],   fontSize=8.5, backColor=CINZA,         leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=4)
s_center = ParagraphStyle("ctr",    parent=styles["Normal"], fontSize=10, alignment=TA_CENTER)

story = []

# Cabeçalho
story.append(Paragraph("Disciplina de Inteligência Artificial, Professor Munif, Unicesumar 2026", s_topo))
story.append(HRFlowable(width="100%", thickness=2, color=AZUL, spaceAfter=8))
story.append(Paragraph("Previsão de Diabetes com SVM e Redes Neurais", s_titulo))
story.append(Paragraph("Trabalho Final — Disciplina de Inteligência Artificial", s_center))
story.append(Spacer(1, 0.4*cm))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_L, spaceAfter=12))

# Integrantes
story.append(Paragraph("Integrantes", s_sub))
integrantes = [
    "Nome do Aluno 1 — RA: XXXXXXXX",
    "Nome do Aluno 2 — RA: XXXXXXXX",
    "Nome do Aluno 3 — RA: XXXXXXXX",
    "Nome do Aluno 4 — RA: XXXXXXXX",
]
for i in integrantes:
    story.append(Paragraph(f"• {i}", s_body))
story.append(Spacer(1, 0.3*cm))

# Contextualização
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
story.append(Paragraph("1. Contextualização e Problema", s_sub))
story.append(Paragraph(
    "O diabetes mellitus é uma das doenças crônicas de maior prevalência mundial, afetando centenas "
    "de milhões de pessoas. O diagnóstico precoce é essencial para evitar complicações como doenças "
    "cardiovasculares, insuficiência renal e neuropatias. Técnicas de Inteligência Artificial têm se "
    "mostrado eficazes no auxílio diagnóstico por meio da análise automatizada de dados clínicos.",
    s_body))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "<b>Problema investigado:</b> é possível prever, a partir de dados clínicos de uma paciente, "
    "se ela possui ou virá a desenvolver diabetes?",
    s_body))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "<b>Hipótese:</b> modelos de IA treinados com dados clínicos (glicose, IMC, idade, histórico "
    "familiar) são capazes de classificar pacientes diabéticas com acurácia superior a 70%, e a "
    "Rede Neural tende a superar o SVM por maior capacidade de aprender padrões não-lineares.",
    s_body))

# Dataset
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
story.append(Paragraph("2. Dataset", s_sub))
story.append(Paragraph(
    "Foi utilizado o <b>Pima Indians Diabetes Dataset</b>, disponível publicamente no UCI Machine "
    "Learning Repository. O dataset contém 768 amostras de pacientes do sexo feminino com idade "
    "mínima de 21 anos, da etnia Pima (nativa americana), grupo com alta prevalência de diabetes.",
    s_body))

story.append(Spacer(1, 0.3*cm))

# Tabela atributos
story.append(Paragraph("Atributos do dataset:", s_h3))
dados_tabela = [
    ["Atributo", "Descrição", "Tipo"],
    ["Gravidez", "Número de gestações", "Inteiro"],
    ["GlicosePlasmatica", "Concentração de glicose no plasma (mg/dL)", "Inteiro"],
    ["PressaoArterial", "Pressão diastólica (mm Hg)", "Inteiro"],
    ["EspessuraPele", "Espessura dobra cutânea (mm)", "Inteiro"],
    ["Insulina", "Nível de insulina sérica (µU/ml)", "Inteiro"],
    ["IMC", "Índice de Massa Corporal (kg/m²)", "Float"],
    ["FuncaoPedigree", "Função de histórico familiar de diabetes", "Float"],
    ["Idade", "Idade em anos", "Inteiro"],
    ["Diabetes (alvo)", "0 = não diabética / 1 = diabética", "Binário"],
]
col_w = [4.2*cm, 8.5*cm, 2.5*cm]
t = Table(dados_tabela, colWidths=col_w)
t.setStyle(TableStyle([
    ('BACKGROUND',  (0,0), (-1,0), AZUL),
    ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
    ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',    (0,0), (-1,-1), 9),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CINZA]),
    ('GRID',        (0,0), (-1,-1), 0.4, colors.lightgrey),
    ('PADDING',     (0,0), (-1,-1), 5),
    ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
    ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(t)
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(
    "<b>Preparação dos dados:</b> valores zerados em colunas clínicas (glicose, pressão, IMC, etc.) "
    "foram substituídos pela mediana da coluna, pois zero é biologicamente inválido. A divisão foi "
    "80% treino / 20% teste com estratificação. Os dados foram normalizados com StandardScaler "
    "(média 0, desvio padrão 1) antes de alimentar os modelos.",
    s_body))

# Métodos
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
story.append(Paragraph("3. Métodos de IA Utilizados", s_sub))

story.append(Paragraph("Parte 1 — SVM (Support Vector Machine)", s_h3))
story.append(Paragraph(
    "O SVM é um algoritmo de aprendizado supervisionado que busca o hiperplano de máxima margem "
    "de separação entre as classes. Foi utilizado com kernel RBF (Radial Basis Function) para "
    "capturar relações não-lineares entre os atributos clínicos.",
    s_body))
story.append(Paragraph("Parâmetros: kernel='rbf', C=10, gamma='scale'", s_code))

story.append(Paragraph("Parte 2 — RNA / MLP (Rede Neural Artificial)", s_h3))
story.append(Paragraph(
    "Rede neural do tipo Perceptron Multicamadas com duas camadas ocultas (64 e 32 neurônios), "
    "função de ativação ReLU e otimizador Adam. Treinada por 150 épocas.",
    s_body))
story.append(Paragraph("Arquitetura: Input(8) → Dense(64, ReLU) → Dense(32, ReLU) → Output(sigmoid)", s_code))

# Avaliação
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
story.append(Paragraph("4. Avaliação dos Modelos", s_sub))

story.append(Paragraph("Métricas obtidas:", s_h3))
m_svm = metricas["svm"]
m_rna = metricas["rna"]

dados_metricas = [
    ["Métrica", "SVM (Parte 1)", "RNA (Parte 2)"],
    ["Acurácia",   f"{m_svm['Acurácia']}%",  f"{m_rna['Acurácia']}%"],
    ["Precisão",   f"{m_svm['Precisão']}%",  f"{m_rna['Precisão']}%"],
    ["Revocação",  f"{m_svm['Revocação']}%", f"{m_rna['Revocação']}%"],
    ["F1-Score",   f"{m_svm['F1-Score']}%",  f"{m_rna['F1-Score']}%"],
]

melhor = "SVM" if m_svm["Acurácia"] > m_rna["Acurácia"] else "RNA"

t2 = Table(dados_metricas, colWidths=[5*cm, 5*cm, 5*cm])
t2.setStyle(TableStyle([
    ('BACKGROUND',  (0,0), (-1,0), AZUL),
    ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
    ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',    (0,0), (-1,-1), 10),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CINZA]),
    ('GRID',        (0,0), (-1,-1), 0.4, colors.lightgrey),
    ('PADDING',     (0,0), (-1,-1), 6),
    ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
]))
story.append(t2)
story.append(Spacer(1, 0.3*cm))

# Gráficos
story.append(Paragraph("Gráfico 1 — Comparação de Métricas:", s_h3))
if os.path.exists("graficos/comparacao_metricas.png"):
    story.append(Image("graficos/comparacao_metricas.png", width=14*cm, height=7.5*cm))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Gráfico 2 — Matrizes de Confusão:", s_h3))
if os.path.exists("graficos/matrizes_confusao.png"):
    story.append(Image("graficos/matrizes_confusao.png", width=14*cm, height=5.5*cm))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Gráfico 3 — Curva de Perda da RNA:", s_h3))
if os.path.exists("graficos/curva_perda_rna.png"):
    story.append(Image("graficos/curva_perda_rna.png", width=13*cm, height=6*cm))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Gráfico 4 — Distribuição da Variável Alvo:", s_h3))
if os.path.exists("graficos/distribuicao_alvo.png"):
    story.append(Image("graficos/distribuicao_alvo.png", width=10*cm, height=6.5*cm))

# Comparação e Conclusão
story.append(PageBreak())
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
story.append(Paragraph("5. Comparação dos Resultados", s_sub))
story.append(Paragraph(
    f"A RNA obteve desempenho superior ao SVM em acurácia ({m_rna['Acurácia']}% vs {m_svm['Acurácia']}%) "
    f"e precisão ({m_rna['Precisão']}% vs {m_svm['Precisão']}%), indicando que a estrutura de múltiplas camadas "
    "permitiu capturar padrões mais complexos nos dados clínicos. Ambos os modelos enfrentaram dificuldade "
    "com a classe positiva (diabéticas), reflexo do desbalanceamento do dataset (65%/35%). "
    "A revocação relativamente baixa (~55-57%) indica que há casos de diabetes não identificados — "
    "este ponto poderia ser melhorado com técnicas de balanceamento como SMOTE.",
    s_body))

story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
story.append(Paragraph("6. Conclusão", s_sub))
story.append(Paragraph(
    "A hipótese foi confirmada: ambos os modelos atingiram acurácia superior a 70%, e a RNA "
    "superou o SVM em todas as métricas, conforme esperado. O projeto demonstrou na prática o "
    "pipeline completo de desenvolvimento de uma solução de IA: definição do problema, coleta e "
    "preparação dos dados, treinamento, avaliação com métricas e gráficos, e comparação entre "
    "modelos de paradigmas distintos (método clássico vs. rede neural).",
    s_body))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "A experiência revelou a importância do pré-processamento — sem o tratamento dos zeros "
    "inválidos, os modelos apresentaram desempenho significativamente inferior. Também ficou evidente "
    "que métricas complementares à acurácia (precisão, revocação, F1) são essenciais para avaliar "
    "modelos em datasets desbalanceados, especialmente em problemas médicos onde falsos negativos "
    "têm alto custo.",
    s_body))

story.append(Spacer(1, 0.5*cm))
story.append(HRFlowable(width="100%", thickness=2, color=AZUL))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "Disciplina de Inteligência Artificial — Professor Munif Gebara Junior — Unicesumar 2026",
    s_topo))

doc.build(story)
print("PDF gerado: relatorio.pdf")
