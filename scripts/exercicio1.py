import numpy as np
import pandas as pd
from src.si.io.csv import read_csv

#1.1 - Neste exercício, vamos usar o iris dataset. Carrega o iris.csv usando o método read apropriado para o tipo de ficheiro.
iris = read_csv(r"C:\Users\Tiago\GitHub\Repositorio de Sistemas\Repositorio-Sistemas\datasets\iris.csv", features= True, label= True)

iris = iris.to_dataframe()
print(f"1.1: \n {iris} \n ----------------------------------------------------------------")

#1.2 - Seleciona a primeira variável independente e verifica a dimensão do array resultante.
print(f"1.2: \n {iris.iloc[0]} \n Dimensão: {iris.iloc[0].shape} \n ----------------------------------------------------------------")

#1.3 - Seleciona as últimas 5 amostras do iris dataset. Qual a média das últimas 5 amostras para cada variável independente/feature?
ultimas5 = iris.loc[-5:]
medias = np.mean(ultimas5, axis=0)
print(f"1.3: \n {medias} \n ----------------------------------------------------------------")

#1.4 - Seleciona todas as amostras do dataset com valor superior ou igual a 1. Nota que o array resultante deve ter apenas
#amostras com valores iguais ou superiores a 1 para todas as features.
iris_data = iris.loc[:, iris.columns != 'class']
new_df = iris_data[iris_data.apply(lambda row: all(float(column) >= 1 for column in row), axis=1)]
print(f"1.4: \n {new_df} \n ----------------------------------------------------------------")

#1.5 -  Seleciona todas as amostras com a classe/label igual a ‘Irissetosa’. Quantas amostras obténs?
iris_setosa = iris["class"] == "Iris-setosa"
print(f"1.5: \n {iris.loc[iris_setosa]}")


