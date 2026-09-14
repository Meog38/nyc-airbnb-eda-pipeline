# NYC Airbnb EDA Pipeline

Projeto de análise exploratória e preparação de dados para o dataset **New York City Airbnb Open Data**. O notebook original foi organizado em um pipeline Python reproduzível, mantendo as decisões da APS1 e deixando os dados prontos para a etapa futura de regressão.

## Objetivos

- entender a estrutura e a qualidade dos anúncios;
- investigar distribuições, valores ausentes, outliers e relações com `price`;
- separar treino e teste antes de aprender qualquer transformação;
- gerar uma matriz pronta para receber um regressor na APS2.

Esta etapa não treina modelos. As relações encontradas são descritivas e não representam causalidade.

## Estrutura

```text
.
├── data/raw/                  # coloque AB_NYC_2019.csv aqui
├── reports/figures/           # gráficos gerados pelo pipeline
├── src/nyc_airbnb_eda/
│   ├── data.py                # carregamento do dataset
│   ├── eda.py                # tabelas e visualizações
│   ├── preprocessing.py      # split e transformações
│   └── cli.py                # ponto de entrada
├── tests/                     # testes unitários
├── Meu_projeto_EDA_ML_revisado (2).ipynb
├── pyproject.toml
└── requirements.txt
```

## Dataset

Baixe `AB_NYC_2019.csv` no [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) e salve-o em `data/raw/`. O arquivo não é versionado pelo Git. Sem um caminho explícito, o pipeline também tenta usar uma cópia pública via URL.

## Instalação

Para recriar os diretórios do projeto no PowerShell:

```powershell
.\setup.ps1
```

O script equivale a criar `data/raw`, `reports/figures`, `src/nyc_airbnb_eda` e `tests` com `mkdir`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Execução

Abra [index.html](index.html) no navegador para visualizar a página de apresentação do projeto.

### Publicação no GitHub Pages

O workflow em `.github/workflows/pages.yml` publica automaticamente a página a cada push na branch `main`.

Na primeira publicação, no GitHub, abra **Settings > Pages**, selecione **GitHub Actions** em *Build and deployment* e aguarde o workflow terminar. A página ficará disponível em:

<https://meog38.github.io/nyc-airbnb-eda-pipeline/>

Na raiz do projeto:

```powershell
python -m pytest -q
python -m src.nyc_airbnb_eda.cli
```

Para indicar outro arquivo CSV:

```powershell
python -m src.nyc_airbnb_eda.cli --input caminho\AB_NYC_2019.csv
```

O comando salva tabelas CSV em `reports/` e imagens em `reports/figures/`, além de mostrar as dimensões do dataset, o número de preços inválidos removidos e as dimensões após o pré-processamento.

## Tratamentos aplicados

- registros com `price <= 0` são removidos;
- `reviews_per_month` recebe imputação zero, pois os nulos correspondem a anúncios sem avaliações;
- contagens assimétricas recebem `log1p`, imputação pela mediana e padronização;
- latitude, longitude e disponibilidade recebem imputação pela mediana e padronização;
- variáveis categóricas usam imputação pela moda e one-hot encoding;
- categorias desconhecidas são ignoradas e bairros raros são agrupados com `min_frequency=20`;
- o pré-processador é ajustado apenas no treino, evitando data leakage.

## Principais achados da EDA

- a base original possui 48.895 anúncios e 16 colunas;
- há 11 registros com `price = 0`;
- Manhattan e Brooklyn concentram a maior parte dos anúncios;
- `Entire home/apt` e `Private room` dominam `room_type`;
- preço e contagens apresentam caudas longas;
- correlações numéricas isoladas com o preço são fracas;
- o PCA exploratório não é incluído no pipeline final.

## Referências

- [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
- [Scikit-learn](https://scikit-learn.org/stable/)
