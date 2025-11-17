import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Aclaracion:
# composición de ingresos de los recursos = índice de desarrollo humano que combina el Producto Nacional Bruto (PNB) per cápita,
# los años de escolaridad y la esperanza de vida al nacer, y sirve como un fuerte indicador del nivel de desarrollo socioeconómico de un país
# (el índice varía de 0 a 1) 0 significando muy bajos ingresos y 1 ingresos muy altos.

df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip()

X_col = 'Income composition of resources'
Y_col = 'Life expectancy'

# elimina las filas con datos faltantes en las columnas X_col y Y_col
df_clean = df[[X_col, Y_col]].dropna()

X = np.array(df_clean[X_col]).reshape(-1, 1)
Y = np.array(df_clean[Y_col]).reshape(-1, 1)

Y_log = np.log(Y)

# entrenar modelo
model = LinearRegression(fit_intercept=True)
model.fit(X, Y_log)

r_squared = model.score(X, Y_log)

# obtener coeficientes
intercept = model.intercept_[0]
coef = model.coef_[0][0]

# rango de valores X para la curva de regresión
X_new = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# predecir Y
Y_new = np.exp(model.predict(X_new))

# crear grafico
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, Y, label='Datos Reales')
ax.plot(X_new, Y_new, color='red', label=r'Regresión Exponencial')

print("El score es: ", r_squared)

ax.set_xlabel(f'Índice Composición de ingresos de los recursos')
ax.set_ylabel('Expectativa de vida (Años)')
ax.set_title(f'Prediccion de Expectativa de Vida basada en Indice Composición de ingresos de los recursos')
ax.legend()
ax.grid(True)
plt.show()