import numpy as np
import pandas as pd

#1.1 - Neste exercício, vamos usar o iris dataset. Carrega o iris.csv usando o método read apropriado para o tipo de ficheiro.
iris = pd.read_csv(r"C:\Users\Tiago\GitHub\Repositorio de Sistemas\Repositorio-Sistemas\datasets\iris.csv")

#1.2 - Seleciona a primeira variável independente e verifica a dimensão do array resultante.
print(iris.loc[0])
print(iris.loc[0].ndim)

#1.3 - Seleciona as últimas 5 amostras do iris dataset. Qual a média das últimas 5 amostras para cada variável independente/feature?
ultimas5 = iris[-5:]
print(np.mean(ultimas5, axis=0))

#1.4 - Seleciona todas as amostras do dataset com valor superior ou igual a 1. Nota que o array resultante deve ter apenas
#amostras com valores iguais ou superiores a 1 para todas as features.
options = ["sepal_length","sepal_width","petal_length","petal_width"]
maior_ou_igual_1=(iris["sepal_length"] >= 1) & (iris["sepal_width"] >= 1) & (iris["petal_length"] >= 1) & (iris["petal_width"] >= 1)
print(iris.loc[maior_ou_igual_1])

#1.5 -  Seleciona todas as amostras com a classe/label igual a ‘Irissetosa’. Quantas amostras obténs?
iris_setosa = iris["class"] == "Iris-setosa"
print(iris.loc[iris_setosa])