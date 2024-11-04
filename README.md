# Predição de acionamento de sinistro em seguro automobilístico com base em algoritmo de aprendizagem de máquina supervisionado.

## Introdução

Apólices de seguros são contratos que oferecem proteção financeira contra perdas ou danos específicos, como acidentes de carro, roubo ou desastres naturais. No mercado automobilístico, essas apólices são fundamentais para mitigar os riscos associados à posse e operação de veículos, proporcionando segurança financeira tanto para os proprietários quanto para as seguradoras.

Desenvolver um modelo preditivo para determinar a probabilidade de uma apólice de seguro ser acionada é de extrema importância. Para as seguradoras, isso significa a capacidade de precificar os seguros de forma mais precisa, gerenciar riscos de maneira mais eficaz e, consequentemente, melhorar a lucratividade. Para os clientes, modelos preditivos podem resultar em prêmios de seguro mais justos e personalizados. No entanto, o desafio reside na complexidade dos dados e na necessidade de criar modelos que sejam tanto precisos quanto interpretáveis, garantindo que as previsões sejam confiáveis e úteis na prática.

Diante desses desafios, grandes companhias como a Porto Seguro disponibilizaram suas bases de dados para que a comunidade pudesse contribuir com essa modelagem. Essa base pública foi divulgada na plataforma de dados abertos Kaggle.

Este trabalho tem como objetivo abordar esse desafio, implementando técnicas de Machine Learning (ML) para prever se um sinistro será acionado ou não, dada uma base de referência disponibilizada por empresa de seguros brasileira.

## Materiais e Métodos

### Hardware e Software

Este trabalho é desenvolvido em software escrito na linguagem R, a infraestrutura utilizada é baseada em containers Docker. A linguagem R é uma linguagem interpretada com vasta coleção que apoiam a comunidade cientifica nas mais diversas frentes, inclusive em ciência de dados [R language](). Os containers Docker formam uma solução que viabilizam que projetos de software sejam desenvolvidos e publicados com todas as dependências de ambiente descritas como um código, a execução deste projetos é portanto garantida desde o processo inicial de desenvolvimento. [Docker](),[Dev Containers]().

| Software | Versão |
|----------|--------|
| R        | 4.0.2  |
| Docker   | 19.03  |

| Hardware | Valores          |
|----------|------------------|
| CPU      | Intel i7-9700K   |
| RAM      | 32GB             |
| Storage  | 1TB SSD          |

### Base de Dados

A base de dados trabalhada é disponibilizada por empresa de seguros automobilísticos na internet em plataforma pública Kaggle. Os dados disponibilizados consistem em dois arquivos, "test.csv" e "train.csv", ambos arquivos em formato CSV (Comma Separated Values), onde o conteúdo é disposto em arquivo de texto plano separado por vírgulas.

As colunas de uma base de dados são referenciadas pela palavra feature, ou variável, que indica uma característica ou atributo do registro estudado.

| Features | Descrição |
|---|---|
| id       | Código identificador |
| target   | A feature target, é o valor que indica o acionamento do seguro. É o alvo que desejamos conseguir replicar com o modelo preditivo. |
| ps_ind   | São 18 features (ps_ind_01 - ps_ind_18_bin) associadas com dados de individuais ou de motoristas. |
| ps_reg   | São 03 features (ps_reg_01 - ps_reg_03) associadas com dados regionais. |
| ps_car   | São 03 features (ps_car_01_cat - ps_car_15) com dados associados aos veículos assegurados. |
| ps_calc  | São 20 features (ps_calc_01 - ps_calc_20_bin) com dados calculados.|

São 49 features categorizadas, contudo existem também features sem categoria implicando em dados contínuos ou ordinais. São 10 as features não categorizadas, implicando em um total de 59 features disponíveis para analise.

A base de dados "train.csv" é destinada ao treinamento do modelo, e possui variável target. Quando treinado, o modelo poderá ser aplicado em nova base de dados, "test.csv" e neste caso a variável target será construída com base no modelo treinado.

### Métodos

Será realizada análise exploratória de cada feature dos dados, seguida por analises quantitativas e qualitativas de suas medidas resumo e suas respectivas distribuições.

Será identificada relações de interdependência entre as features por meio de analise com componentes principais (PCA), avaliando a possibilidade de redução de dimensionalidade do problema posto.

Para cada agrupamento de features identificadas no passo anterior, será aplicado modelo de aprendizagem de máquina supervisionado de classificação, para confrontar com variável target. 

A qualidade desta previsão será obtida a partir de analise de métricas padrão de desempenho como a area sob a curva ROC [AUC](), que agrega métricas de falso-positivo e outros desvios que possam existir.

## Resultados e Discussão

### Analise de Dados Iniciais

Estrutura da base de dados: 59 Features

```{r}
> table(sapply(train, class))
factor integer logical numeric 
    16      16      17      10 
```

## Considerações Finais


## Agradecimento