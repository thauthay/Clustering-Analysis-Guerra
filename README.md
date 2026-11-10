# Guerra Clustering Analysis

Sistema de análise e clusterização para identificação de perfis de atletas em escola de futebol.

## Descrição

Aplicação de Machine Learning não supervisionado para segmentar atletas com base em características físicas e técnicas. Utiliza três algoritmos (K-Means, Hierarchical, GMM) para formar grupos homogêneos de treinamento.

## Tecnologias

**Core**
- Python 3.8+
- Streamlit 1.50+
- scikit-learn

**Análise e Visualização**
- pandas, numpy
- plotly, matplotlib, seaborn

## Formato de Dados

CSV com colunas: Nome, Idade, Altura, Velocidade (1-5), Força (1-10), Reflexo (1-10), Precisão (1-10)

## Funcionalidades

- Exploração de dados e correlações
- Feature engineering automático
- Três algoritmos de clustering
- Análise de métricas (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Visualização 3D com PCA
- Identificação de perfis (Atletas, Técnicos, Completos, Desenvolvimento)
- Gráficos radar e análise por idade
- Exportação de relatórios

## Perfis

**Atletas**: Físico predominante | Zagueiro, Volante | Treino: Técnica

**Técnicos**: Habilidades predominantes | Meia, Ponta | Treino: Físico

**Completos**: Equilibrado | Meio-campo, Lateral | Treino: Balanceado

**Desenvolvimento**: Em evolução | Variadas | Treino: Fundamentos
