# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 21:34:45 2025

@author: jalis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

try:
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    use_lstm = True
except ImportError:
    use_lstm = False

# Configuración de página
st.set_page_config(page_title="Proyección Asegurados", layout="centered")
st.title("📈 Proyección de Asegurados - Guadalajara")

# Selección de modelos
modelos_seleccionados = st.multiselect(
    "Selecciona modelos de pronóstico a mostrar:",
    ["Lineal", "Ridge", "ARIMA", "Prophet", "LSTM" if use_lstm else None],
    default=["Lineal", "Ridge", "ARIMA"]
)
ruta = 'D:/Coordinacion/Codigos_python/empleo/'
df = pd.read_csv(ruta + "guadalajara_asegurados.csv ")
df['fecha'] = pd.to_datetime(df['fecha'])
df['año'] = df['fecha'].dt.year
df_anual = df.groupby('año')['asegurados'].sum().reset_index()

X = df_anual[['año']]
y = df_anual['asegurados']
años_futuro = np.arange(df_anual['año'].max() + 1, df_anual['año'].max() + 7)

resultados = pd.DataFrame({'Año': años_futuro})

# Modelos y predicciones
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_anual['año'], y, marker='o', label='Histórico', color='black')

if "Lineal" in modelos_seleccionados:
    model = LinearRegression().fit(X, y)
    pred = model.predict(años_futuro.reshape(-1, 1))
    ax.plot(años_futuro, pred, '--o', label='Lineal', color='blue')
    resultados["Lineal"] = pred

if "Ridge" in modelos_seleccionados:
    model = Ridge().fit(X, y)
    pred = model.predict(años_futuro.reshape(-1, 1))
    ax.plot(años_futuro, pred, '--o', label='Ridge', color='orange')
    resultados["Ridge"] = pred

if "ARIMA" in modelos_seleccionados:
    serie = pd.Series(y.values, index=df_anual['año'])
    model = ARIMA(serie, order=(1, 1, 1)).fit()
    pred = model.forecast(steps=6)
    ax.plot(años_futuro, pred, '--o', label='ARIMA', color='green')
    resultados["ARIMA"] = pred.values

if "Prophet" in modelos_seleccionados:
    df_p = df_anual.rename(columns={'año': 'ds', 'asegurados': 'y'})
    df_p['ds'] = pd.to_datetime(df_p['ds'], format='%Y')
    model = Prophet(yearly_seasonality=True)
    model.fit(df_p)
    future = model.make_future_dataframe(periods=6, freq='Y')
    forecast = model.predict(future).tail(6)
    pred = forecast['yhat'].values
    ax.plot(años_futuro, pred, '--o', label='Prophet', color='purple')
    resultados["Prophet"] = pred

if "LSTM" in modelos_seleccionados and use_lstm:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    X_lstm, y_lstm = [], []
    for i in range(len(scaled) - 1):
        X_lstm.append(scaled[i:i+1])
        y_lstm.append(scaled[i+1])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_lstm, y_lstm, epochs=300, verbose=0)
    preds = []
    input_val = scaled[-1].reshape(1, 1, 1)
    for _ in range(6):
        next_pred = model.predict(input_val, verbose=0)
        preds.append(scaler.inverse_transform(next_pred)[0][0])
        input_val = next_pred.reshape(1, 1, 1)
    ax.plot(años_futuro, preds, '--o', label='LSTM', color='red')
    resultados["LSTM"] = preds

# Etiquetas y leyenda
ax.set_title("Proyección de asegurados (Guadalajara)")
ax.set_xlabel("Año")
ax.set_ylabel("Asegurados")
ax.legend()
ax.grid(True)

# Mostrar gráfica
st.pyplot(fig)

# Mostrar tabla
st.markdown("### 📋 Tabla de predicciones")
st.dataframe(resultados.style.format(precision=0, thousands=","))

# Descargar CSV
csv = resultados.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Descargar CSV con predicciones",
    data=csv,
    file_name='proyecciones_asegurados.csv',
    mime='text/csv'
)
