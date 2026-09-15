# NYC Airbnb EDA Pipeline

Análise exploratória e preparação de dados para o dataset **New York City Airbnb Open Data**, com foco em entender os fatores associados ao preço anunciado por noite e deixar uma base consistente para modelagem de regressão.

## Visão Geral

- **Dataset:** New York City Airbnb Open Data.
- **Problema:** preparação para previsão de `price`.
- **Autores:** Maria Eduarda Oliveira Galdino e Tomaz Santos Almeida.
- **Site da análise:** <https://meog38.github.io/nyc-airbnb-eda-pipeline/>.
- **Relatório web:** `index.html`.
- **Notebook de análise:** `Meu_projeto_EDA_ML_revisado (2).ipynb`.

O projeto não treina modelos nesta etapa. O foco é auditoria dos dados, análise exploratória, justificativa das decisões de pré-processamento e construção de um pipeline reproduzível.

## Estrutura

```text
.
├── data/raw/                  # coloque AB_NYC_2019.csv aqui, se quiser usar cópia local
├── reports/
│   ├── figures/               # gráficos usados no relatório web
│   └── *.csv                  # tabelas resumidas geradas pelo pipeline
├── src/nyc_airbnb_eda/
│   ├── data.py                # carregamento do dataset
│   ├── eda.py                 # tabelas e visualizações
│   ├── preprocessing.py       # split e transformações
│   └── cli.py                 # ponto de entrada
├── tests/                     # testes unitários
├── index.html                 # página estática do relatório
├── styles.css                 # estilos da página
├── Meu_projeto_EDA_ML_revisado (2).ipynb
├── pyproject.toml
└── requirements.txt
```

## Dataset

O arquivo `AB_NYC_2019.csv` pode ser baixado no Kaggle:

<https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>

Salve o CSV em `data/raw/`. O arquivo bruto não é versionado pelo Git. Quando nenhuma cópia local é encontrada, o pipeline tenta carregar uma cópia pública definida em `src/nyc_airbnb_eda/data.py`.

## Execução Local

No PowerShell:

```powershell
.\setup.ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.nyc_airbnb_eda.cli
```

Para indicar outro caminho de CSV:

```powershell
python -m src.nyc_airbnb_eda.cli --input caminho\AB_NYC_2019.csv
```

O comando gera tabelas em `reports/` e figuras em `reports/figures/`.

## Relatório Web

A página estática em `index.html` apresenta:

- resumo do dataset;
- dicionário das 16 colunas;
- auditoria de valores ausentes, inconsistências e extremos;
- estatísticas descritivas das variáveis numéricas relevantes;
- visualizações univariadas, bivariadas e multivariadas;
- relações entre variáveis categóricas e `price`;
- análise de PCA das variáveis numéricas;
- arquitetura do pipeline de pré-processamento.

O CSS principal está em `styles.css`, e os estilos essenciais também estão embutidos no HTML para evitar falhas de renderização em ambientes estáticos.

## Principais Achados

- A base original tem 48.895 anúncios e 16 colunas.
- O alvo de modelagem é `price`, preço anunciado por noite.
- Existem 11 registros com `price = 0`, removidos como inconsistência do alvo.
- As ausências em `reviews_per_month` e `last_review` representam anúncios sem avaliações; por isso `reviews_per_month` recebe imputação zero.
- `price`, `minimum_nights`, reviews e contagens apresentam caudas longas.
- Manhattan e Brooklyn concentram a maior parte dos anúncios.
- `Entire home/apt` e `Private room` dominam os tipos de acomodação.
- As correlações numéricas isoladas com preço são fracas.
- Tipo de acomodação, borough e bairro mostram diferenças relevantes de preço mediano.
- O PCA ajuda a visualizar a estrutura numérica, mas não separa claramente faixas de preço.

## Pré-Processamento

As decisões de preparação foram reunidas com `Pipeline` e `ColumnTransformer`:

- remover registros com `price <= 0`;
- separar treino e teste antes de aprender imputação, escala ou encoding;
- imputar `reviews_per_month` com zero;
- imputar demais variáveis numéricas com mediana;
- imputar variáveis categóricas com moda;
- aplicar `log1p` em variáveis assimétricas de contagem;
- padronizar numéricas com `StandardScaler`;
- aplicar one-hot encoding em `neighbourhood_group`, `neighbourhood` e `room_type`;
- usar `min_frequency=20` e `handle_unknown="ignore"` para controlar categorias raras e novas.

## Cobertura Técnica

| Etapa | Evidência no projeto |
|---|---|
| Inspeção inicial | Dicionário de dados, tipos conceituais, dimensões, ausências e inconsistências |
| Análise univariada | Estatísticas descritivas e gráficos para variáveis numéricas e categóricas |
| Análise bivariada/multivariada | Correlações, scatter plots, boxplots conjuntos e preço por categoria |
| Pré-processamento | Missing values, outliers, encoding, escala e split sem vazamento |
| Redução de dimensionalidade | PCA numérico exploratório com interpretação dos componentes |
| Reprodutibilidade | Código modular em `src/`, testes em `tests/` e pipeline executável por CLI |

## GitHub Pages

O workflow em `.github/workflows/pages.yml` publica a página a cada push na branch `main`. Se o repositório ainda não estiver configurado, abra **Settings > Pages**, selecione **GitHub Actions** em *Build and deployment* e execute o workflow novamente.

Arquivos necessários para a página:

- `index.html`
- `styles.css`
- `.nojekyll`
- `reports/figures/*.png`
- `Meu_projeto_EDA_ML_revisado (2).ipynb`

## Referências

- Kaggle. New York City Airbnb Open Data: <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
- Documentação do scikit-learn para `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler` e `PCA`.
