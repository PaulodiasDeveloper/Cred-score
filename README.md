# Projeto de Concessão de Cartões de Crédito

## Etapa 1 CRISP-DM: Entendimento do Negócio

### Objetivos do Negócio

Este projeto tem como objetivo desenvolver um modelo preditivo para identificar o risco de inadimplência de proponentes de cartões de crédito. O problema foi publicado no Kaggle, uma plataforma que promove desafios de ciência de dados. Nosso objetivo é construir um modelo que auxilie o mutuário (o cliente) a tomar decisões informadas sobre crédito.

### Descrição do Problema

- **Objetivo do modelo**: Identificar o risco de inadimplência (definido pela ocorrência de um atraso maior ou igual a 90 dias em um horizonte de 12 meses) no momento da avaliação do crédito.
- **Público-alvo**: Proponentes de cartões de crédito.

### Situação do Negócio

Nesta etapa, avaliamos a situação do segmento de concessão de crédito para entender o tamanho do público, relevância, problemas presentes e todos os detalhes do processo gerador do fenômeno em questão.

## Etapa 2 CRISP-DM: Entendimento dos Dados

### Dicionário de Dados

A tabela de dados contém uma linha para cada cliente e uma coluna para cada variável armazenando as características dos clientes. Abaixo está o dicionário de dados:

| Variable Name             | Description                                    | Tipo    |
|---------------------------|------------------------------------------------|---------|
| sexo                      | M = 'Masculino'; F = 'Feminino'                | M/F     |
| posse_de_veiculo          | Y = 'possui'; N = 'não possui'                 | Y/N     |
| posse_de_imovel           | Y = 'possui'; N = 'não possui'                 | Y/N     |
| qtd_filhos                | Quantidade de filhos                           | inteiro |
| tipo_renda                | Tipo de renda (ex: assaliariado, autônomo etc) | texto   |
| educacao                  | Nível de educação (ex: secundário, superior)   | texto   |
| estado_civil              | Estado civil (ex: solteiro, casado etc)        | texto   |
| tipo_residencia           | Tipo de residência (ex: casa/apartamento)      | texto   |
| idade                     | Idade em anos                                  | inteiro |
| tempo_emprego             | Tempo de emprego em anos                       | inteiro |
| possui_celular            | Indica se possui celular (1 = sim, 0 = não)    | binária |
| possui_fone_comercial     | Indica se possui telefone comercial (1 = sim)  | binária |
| possui_fone               | Indica se possui telefone (1 = sim, 0 = não)   | binária |
| possui_email              | Indica se possui e-mail (1 = sim, 0 = não)     | binária |
| qt_pessoas_residencia     | Quantidade de pessoas na residência            | inteiro |
| mau                       | Indicadora de mau pagador (True = mau)         | binária |

### Carregando os Pacotes

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
```


### Carregando os Dados

```df = pd.read_csv('demo01.csv')
print("Número de linhas e colunas da tabela: {}".format(df.shape))
df.head()
```

### Etapa 3 CRISP-DM: Preparação dos Dados
Limpeza e Transformação dos Dados

Nesta etapa, identificamos e tratamos dados faltantes, além de realizar conversões necessárias.

```metadata = pd.DataFrame(df.dtypes, columns=['tipo'])
metadata['n_categorias'] = 0

for var in metadata.index:
    metadata.loc[var, 'n_categorias'] = len(df.groupby([var]).size())

def convert_dummy(df, feature, rank=0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest], axis=1, inplace=True)
    df.drop([feature], axis=1, inplace=True)
    df = df.join(pos)
    return df

for var in metadata[metadata['tipo'] == 'object'].index:
    df = convert_dummy(df, var)
```

### Etapa 4 CRISP-DM: Modelagem
Construção do Modelo

Selecionamos a técnica de Floresta Aleatória (Random Forest) para a construção do modelo.

```x = df.drop("mau", axis=1)
y = df["mau"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

clf = RandomForestClassifier(n_estimators=3)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print('Acurácia: {0:.2f}%'.format(acc * 100))
```

### Avaliação do Modelo
Avaliação da acurácia do modelo e da matriz de confusão.

```
tab = pd.crosstab(index=y_pred, columns=y_test)
print(tab)
```

### Etapa 5 CRISP-DM: Avaliação dos Resultados
Nesta etapa, avaliamos o impacto do uso do modelo no negócio, considerando um cenário simples de lucro e prejuízo.

### Etapa 6 CRISP-DM: Implantação

A implantação do modelo envolve a implementação do mesmo em um motor de crédito que toma decisões com algum nível de automação, aprovando automaticamente clientes muito bons, negando automaticamente clientes muito ruins, e enviando os intermediários para análise manual.



