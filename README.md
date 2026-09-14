# NYC Airbnb EDA Pipeline

Projeto de análise exploratória e preparação de dados para o dataset **New York City Airbnb Open Data**. A entrega foi organizada para a APS1 de Machine Learning e publicada como página estática via GitHub Pages.

## Entrega

- **Tema:** regressão para previsão futura de `price`.
- **Grupo:** Maria Eduarda Oliveira Galdino e Tomaz Santos Almeida.
- **Data:** 14/09.
- **Relatório web:** abra `index.html`.
- **Notebook principal:** `Meu_projeto_EDA_ML_revisado (2).ipynb`.
- **Rubrica:** `Projeto_de_Machine_Learning (1).ipynb`.

## Objetivos

- entender estrutura, tipos e qualidade dos dados;
- identificar ausências, inconsistências, outliers e desbalanceamentos de grupos;
- analisar distribuições numéricas e categóricas;
- estudar relações entre variáveis e o alvo `price`;
- separar treino e teste antes de qualquer transformação;
- construir um pré-processador reproduzível com `Pipeline` e `ColumnTransformer`.

Esta etapa não treina modelos. As relações discutidas são associativas, não causais.

## Estrutura

```text
.
├── data/raw/                  # coloque AB_NYC_2019.csv aqui, se quiser usar cópia local
├── reports/
│   ├── figures/               # gráficos usados no GitHub Pages
│   └── *.csv                  # tabelas resumidas da EDA
├── src/nyc_airbnb_eda/
│   ├── data.py                # carregamento do dataset
│   ├── eda.py                 # tabelas e visualizações
│   ├── preprocessing.py       # split e transformações
│   └── cli.py                 # ponto de entrada
├── tests/                     # testes unitários
├── index.html                 # página do GitHub Pages
├── styles.css                 # estilos da página
├── Meu_projeto_EDA_ML_revisado (2).ipynb
├── pyproject.toml
└── requirements.txt
```

## Dataset

Baixe `AB_NYC_2019.csv` no Kaggle:

<https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>

Salve o arquivo em `data/raw/`. O CSV não é versionado pelo Git. Quando nenhuma cópia local é encontrada, o pipeline tenta usar uma cópia pública definida em `src/nyc_airbnb_eda/data.py`.

## Instalação e execução

No PowerShell:

```powershell
.\setup.ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.nyc_airbnb_eda.cli
```

Para indicar outro CSV:

```powershell
python -m src.nyc_airbnb_eda.cli --input caminho\AB_NYC_2019.csv
```

O comando gera tabelas em `reports/` e figuras em `reports/figures/`.

## GitHub Pages

O workflow em `.github/workflows/pages.yml` publica automaticamente a página a cada push na branch `main`. Se o repositório ainda não estiver configurado, abra **Settings > Pages**, escolha **GitHub Actions** em *Build and deployment* e execute o workflow novamente.

Arquivos importantes para a publicação:

- `index.html`
- `styles.css`
- `.nojekyll`
- `reports/figures/*.png`
- notebooks referenciados pelos links da página

## Principais achados

- A base tem 48.895 anúncios e 16 colunas.
- O alvo futuro é `price`, preço anunciado por noite.
- Existem 11 registros com `price = 0`, removidos como inconsistência do alvo.
- `reviews_per_month` e `last_review` possuem ausências coerentes com anúncios sem reviews.
- `price`, `minimum_nights`, reviews e contagens apresentam caudas longas.
- Manhattan, Brooklyn, `Entire home/apt` e `Private room` dominam a amostra.
- As correlações numéricas isoladas com preço são fracas.
- Tipo de acomodação, borough e bairro mostram diferenças relevantes de preço mediano.
- O PCA é útil para exploração, mas não mostrou separabilidade clara de faixas de preço.

## Pré-processamento proposto

- remover registros com `price <= 0`;
- separar treino e teste antes de aprender imputação, escala ou encoding;
- imputar `reviews_per_month` com zero;
- imputar demais numéricas com mediana, se necessário;
- imputar categóricas com moda, se necessário;
- aplicar `log1p` em variáveis de contagem assimétricas;
- padronizar numéricas com `StandardScaler`;
- aplicar one-hot encoding em `neighbourhood_group`, `neighbourhood` e `room_type`;
- usar `min_frequency=20` e `handle_unknown="ignore"` para controlar categorias raras e novas.

## Aderência à rubrica da APS1

| Critério | Situação |
|---|---|
| Carregamento, descrição de features, tipos, ausências e inconsistências | Cumpre |
| Separação treino/teste sem vazamento | Cumpre |
| Estatísticas descritivas e visualizações univariadas | Cumpre |
| Relações bivariadas e multivariadas | Cumpre |
| Missing values, outliers, encoding e padronização justificados | Cumpre |
| PCA aplicado e interpretado | Cumpre |
| Pipeline com `Pipeline` e `ColumnTransformer` | Cumpre |
| Relatório final e documentação aberta | Cumpre bem |

## Referências

- Kaggle. New York City Airbnb Open Data: <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
- Documentação do scikit-learn para `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler` e `PCA`.
