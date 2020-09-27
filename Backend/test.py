import numpy as np
import regression
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Salary_Data.csv')
x = data.iloc[:, [0]].values
y = data.iloc[:, -1].values
lr = regression.LinearRegression()
lr.fit(x, y)
