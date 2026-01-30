CODE-5 PART 3

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# 1. Definición de Parámetros y Portafolios (1 unidad de cada acción)
portafolios_def = {
    "P_Peru": ["BVN", "SCCO"],
    "P_Sudamerica": ["FM.TO", "HBM", "SCCO", "GMEXICOB.MX"],
    "P_Norteamerica": ["FCX", "HL", "HBM", "FM.TO"],
    "P_Grupal": ["BHP", "GLEN.L", "FCX", "FM.TO", "RIO"]
}
fechas_objetivo = ["2020-03-09", "2022-12-01", "2023-09-15", "2023-10-07", "2024-03-14", "2024-09-15"]
all_tickers = list(set([t for p in portafolios_def.values() for t in p]))

# 2. Descarga de Datos y Cálculo de Retornos
print("Descargando datos de Yahoo Finance...")
data = yf.download(all_tickers, start="2019-11-01", end="2024-12-01", progress=False)['Close']
returns_all = data.pct_change().dropna()

# Funciones de Cálculo (Aseguran Convergencia)
def calc_var(s): return np.percentile(s, 5)
def calc_cvar(s):
    cutoff = np.percentile(s, 5)
    tail = s[s < cutoff]
    return tail.mean() if not tail.empty else np.nan

# Contenedores para Tablas 7x5
resultados_var = {p: [] for p in portafolios_def}
resultados_cvar = {p: [] for p in portafolios_def}
log_backtesting = []

# 3. Procesamiento de los 6 Eventos
for fecha in fechas_objetivo:
    f_dt = pd.to_datetime(fecha)
    f_inicio, f_fin = f_dt - pd.DateOffset(months=2), f_dt + pd.DateOffset(months=2)
    f_margen = f_inicio - timedelta(days=35) # Margen para rolling inicial

    # Gráfico Sobrepuesto de CVaR por Evento (Análisis comparativo)
    fig_ov, ax_ov = plt.subplots(figsize=(10, 5))

    for p_nombre, tickers in portafolios_def.items():
        # Ponderación Price-Weighted al inicio de la ventana
        precios_ini = data.asof(f_inicio)[tickers]
        weights = precios_ini / precios_ini.sum()

        # Retornos del Portafolio
        ret_p_full = (returns_all[tickers] * weights).sum(axis=1)
        ret_p_window = ret_p_full.loc[f_inicio:f_fin]

        # Riesgo Rolling 21d (VaR y CVaR)
        v_roll = ret_p_full.loc[f_margen:f_fin].rolling(21).apply(calc_var).loc[f_inicio:f_fin]
        cv_roll = ret_p_full.loc[f_margen:f_fin].rolling(21).apply(calc_cvar).loc[f_inicio:f_fin]

        # Llenado de Tablas
        resultados_var[p_nombre].append(v_roll.min())
        resultados_cvar[p_nombre].append(cv_roll.min())

        # Backtesting: Conteo de Violaciones y TEF
        ret_s, var_s = ret_p_window.align(v_roll, join='inner')
        viol = ret_s[ret_s < var_s]
        n_viol = len(viol)
        tef = n_viol / len(ret_s) if len(ret_s) > 0 else 0
        log_backtesting.append({'Fecha': fecha, 'Portafolio': p_nombre, 'Violaciones': n_viol, 'TEF': f"{tef:.2%}"})

        # --- GENERACIÓN DE LOS 48 GRÁFICOS INDIVIDUALES ---
        # A) Gráfico de CVaR Individual
        plt.figure(figsize=(10, 5))
        plt.plot(cv_roll.index, cv_roll, label='CVaR 95%', color='teal', linewidth=1.5)
        plt.title(f"Graph 8: CVaR Rolling {p_nombre} - Evento {fecha}")
        plt.grid(False); plt.gca().spines[:].set_edgecolor('black')
        plt.savefig(f"CVAR_INDIV_{p_nombre}_{fecha}.png")
        plt.close()

        # B) Gráfico de Backtesting Individual
        plt.figure(figsize=(10, 5))
        plt.plot(ret_p_window.index, ret_p_window, color='white', alpha=0.3, label='Retornos (Lineas Transp.)')
        plt.plot(v_roll.index, v_roll, color='blue', linestyle='--', label='VaR 95% Rolling')
        plt.scatter(viol.index, viol, color='red', s=25, label='Violación', zorder=5)
        plt.title(f"Graph 10: Backtesting VaR 95% - {p_nombre} - {fecha}")
        plt.grid(False); plt.gca().set_facecolor('black') # Fondo oscuro para resaltar lineas blancas
        plt.gca().spines[:].set_edgecolor('black')
        plt.legend(facecolor='gray')
        plt.savefig(f"BT_INDIV_{p_nombre}_{fecha}.png")
        plt.close()

        # Añadir al gráfico sobrepuesto
        ax_ov.plot(cv_roll.index, cv_roll, label=p_nombre)

    # Finalizar gráfico sobrepuesto por evento
    ax_ov.set_title(f"Evolución CVaR Sobrepuesto - Evento {fecha}")
    ax_ov.axvline(f_dt, color='red', linestyle=':')
    ax_ov.grid(False); ax_ov.legend()
    fig_ov.savefig(f"CVAR_SOBREPUESTO_{fecha}.png")
    plt.close(fig_ov)

# 4. Consolidación de Tablas 7x5
df_var_final = pd.DataFrame(resultados_var, index=fechas_objetivo)
df_var_final.loc['PROMEDIO'] = df_var_final.mean()

df_cvar_final = pd.DataFrame(resultados_cvar, index=fechas_objetivo)
df_cvar_final.loc['PROMEDIO'] = df_cvar_final.mean()

# 5. Reporte de Resultados
print("\n" + "="*50)
print("ANÁLISIS DE RIESGO: TABLA VaR 95% (7x5)")
print(df_var_final)
print("\nANÁLISIS DE RIESGO: TABLA CVaR 95% (7x5)")
print(df_cvar_final)
print("\n" + "="*50)
print("RESUMEN DE BACKTESTING Y TASA EMPÍRICA DE FALLO (TEF)")
print(pd.DataFrame(log_backtesting).to_string(index=False))
