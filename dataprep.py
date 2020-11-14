import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('mushrooms.csv')

print(data.describe())
print(data.info())
print(data.head())

classMushroom = data['class']
Y = data['population']
capShape = data['cap-shape']
capColor = data['cap-color']
plt.hist(capShape)
plt.show()