# G.SubtForestMAEnsemble — Guia de Execução

Este repositório consolida os pipelines de ensemble para classificação (tile, imagem e paciente) a partir de resultados de múltiplos folds e arquiteturas de rede.

## Visão Geral

Há três notebooks principais:
- `G.SubtVision etapa 1 - ensemble single_arquitetura.ipynb` (SA): gera ensembles por arquitetura (MobileNetV2, GoogleNet, ShuffleNetV2, EFF-NET 2º treinamento) consolidando seus 10 folds.
- `G.SubtVision etapa 2 - ensemble multiarquitetura.ipynb` (MA): unifica os resultados de três arquiteturas (30 folds) e calcula métricas/curvas.
- `G.SubtVision etapa 3 - ensemble consolidado.ipynb` (Consolidado): executa as etapas 1 e 2 de forma organizada, reutilizando os módulos da pasta `gsubtvision`.

Os módulos reutilizados (clean code, boas práticas):
- `gsubtvision.constants`: classes e caminhos padrão.
- `gsubtvision.parsing`: conversões robustas de vetores de probabilidade.
- `gsubtvision.io_utils`: utilitários de IO (criação de diretórios, salvar CSV).
- `gsubtvision.ensemble_sa`: lógica da Etapa 1 (Single Arquitetura).
- `gsubtvision.ensemble_ma`: lógica da Etapa 2 (Multi-Arquitetura), incluindo agregações por imagem e paciente.

## Estrutura de Pastas (resumo)

- `inputs/summary_results/`: CSVs de entrada (folds por arquitetura).
- `outputs/`: diretório base de saídas (tabelas mescladas e gráficos).
  - `outputs/merged_table_*.csv`: tabelas mescladas por arquitetura (SA) e o arquivo unificado (MA).
  - `outputs/auc_roc_sa_plots/`: gráficos ROC gerados pelo notebook (padrão dos notebooks originais).
  - `outputs/auc_roc_ma_plots/`: opcional, caso configure o módulo MA para salvar as figuras separadamente.

## Pré-requisitos

- Python 3.x
- Jupyter Notebook/Lab
- Dependências: instale com

```bash
pip install -r requeriments.txt
```

Observação: se o `pandas` emitir avisos sobre `numexpr` ou `bottleneck`, atualize-os:

```bash
pip install --upgrade numexpr bottleneck
```

## Execução Rápida (Etapa Consolidada)

Para executar todo o pipeline de forma automática (Etapas 1 e 2):

```bash
python -m jupyter nbconvert --to notebook --execute "G.SubtVision etapa 3 - ensemble consolidado.ipynb" --output "G.SubtVision etapa 3 - ensemble consolidado.ipynb"
```

Isso irá:
- Ler os CSVs em `inputs/summary_results/`.
- Gerar as tabelas mescladas em `outputs/`.
- Imprimir métricas de classificação (tile, imagem, paciente para SA; tile, imagem e paciente para MA quando habilitado).
- Salvar figuras de curvas ROC em `outputs/auc_roc_sa_plots/` (por padrão).

## Execução Detalhada por Etapa

### Etapa 1 — Single Arquitetura (SA)
1. Mescla 10 folds da arquitetura em uma tabela consolidada.
2. Calcula métricas em nível de TILE:
   - Hard Voting (votação majoritária)
   - Soft Voting (média dos vetores de probabilidade)
3. Agrega e avalia em nível de IMAGEM (média de probabilidades por imagem, classificação por voto/soft).
4. Agrega e avalia em nível de PACIENTE (média de probabilidades por paciente, classificação por voto/soft).
5. Gera curva ROC (Soft Voting) em nível de TILE.

Saídas principais (exemplos):
- `outputs/merged_table_mob.csv`, `outputs/merged_table_gg.csv`, `outputs/merged_table_sh.csv`, `outputs/merged_table_eff2.csv`
- `outputs/auc_roc_sa_plots/sa_tile_roc_*.png`

### Etapa 2 — Multi-Arquitetura (MA)
1. Unifica as tabelas de 30 folds (MobileNetV2, ShuffleNetV2 e GoogleNet).
2. Calcula métricas em nível de TILE:
   - Hard Voting
   - Soft Voting
3. (Opcional no notebook consolidado, disponível no módulo) Agrega e avalia em nível de IMAGEM.
4. (Opcional no notebook consolidado, disponível no módulo) Agrega e avalia em nível de PACIENTE.
5. Gera curva ROC (Soft Voting) em nível de TILE.

Saídas principais:
- `outputs/merged_table_mob_shu_gg.csv`
- Figuras ROC por padrão em `outputs/auc_roc_sa_plots/` (podem ser direcionadas para `outputs/auc_roc_ma_plots/` se preferir).

## Fluxo de Execução (Diagrama)

```mermaid
graph TD
  A[Inputs: CSVs de folds por arquitetura<br/>inputs/summary_results/] --> B[Etapa 1 (SA): Mesclar 10 folds por arquitetura]
  B --> C1[SA: Hard Voting (TILE)]
  B --> C2[SA: Soft Voting (TILE)]
  C2 --> D1[SA: Curvas ROC (TILE)]
  B --> E1[SA: Agregação por IMAGEM]
  E1 --> F1[SA: Métricas por IMAGEM]
  B --> E2[SA: Agregação por PACIENTE]
  E2 --> F2[SA: Métricas por PACIENTE]
  B --> G[Salvar CSVs: merged_table_mob.csv, gg.csv, sh.csv, eff2.csv]

  A --> H[Etapa 2 (MA): Mesclar 30 folds de 3 arquiteturas]
  H --> I1[MA: Hard Voting (TILE)]
  H --> I2[MA: Soft Voting (TILE)]
  I2 --> J[MA: Curvas ROC (TILE)]
  H --> K1[MA: Agregação por IMAGEM]
  K1 --> L1[MA: Métricas por IMAGEM]
  H --> K2[MA: Agregação por PACIENTE]
  K2 --> L2[MA: Métricas por PACIENTE]
  H --> M[Salvar CSV: merged_table_mob_shu_gg.csv]
```


## Onde os Resultados São Salvos

- Tabelas:
  - `outputs/merged_table_mob.csv`
  - `outputs/merged_table_gg.csv`
  - `outputs/merged_table_sh.csv`
  - `outputs/merged_table_eff2.csv`
  - `outputs/merged_table_mob_shu_gg.csv`

- Figuras (ROC):
  - Por padrão: `outputs/auc_roc_sa_plots/`
  - Opcional (MA): `outputs/auc_roc_ma_plots/` (se configurado)

## Personalizações

- Altere `gsubtvision/constants.py` para modificar classes ou caminhos de saída.
- No notebook, ajuste as listas de arquiteturas se quiser incluir/remover redes.
- No módulo `ensemble_ma.py`, use `plot_sodt_roc_curve(..., save_output_dir='outputs/auc_roc_ma_plots')` para separar claramente as figuras de MA das de SA.
- Use `aggregate_to_image_level` e `aggregate_to_patient_level` (em SA e MA) para avaliar os níveis desejados.

## Solução de Problemas

- Erro ao abrir notebook no editor: se a ferramenta acusar JSON inválido, revise strings com `\n` e aspas dentro das células (ex.: prefira f-strings com aspas simples dentro do JSON). Já corrigimos isso no notebook consolidado.
- Pacotes faltando/obsoletos: rode `pip install -r requeriments.txt` e atualize `numexpr`/`bottleneck` se avisado pelo pandas.
- Caminhos incorretos: confirme `inputs/summary_results/` e o conteúdo dos arquivos `_foldX_results.csv` para cada arquitetura.

---

Em caso de dúvidas, abra o notebook consolidado e execute célula por célula — cada bloco está documentado com mensagens de início/fim e nomes claros, evitando ambiguidades.

## Download do Dataset

Para baixar o conjunto de dados utilizado, acesse:

https://drive.google.com/drive/folders/1fF8F0cEJx-bdbiAf87v8unXYd6xFVDHc?usp=drive_link