# CODE-5 PART 1

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# 1. Configuración de Portafolios
portafolios_def = {
    "P_Peru": ["BVN", "SCCO"],
    "P_Sudamerica": ["FM.TO", "HBM", "SCCO", "GMEXICOB.MX"],
    "P_Norteamerica": ["FCX", "HL", "HBM", "FM.TO"],
    "P_Grupal": ["BHP", "GLEN.L", "FCX", "FM.TO", "RIO"]
}
fechas_objetivo = ["2020-03-09", "2022-12-01", "2023-09-15", "2023-10-07", "2024-03-14", "2024-09-15"]
all_tickers = list(set([t for p in portafolios_def.values() for t in p]))

# Descarga de datos con margen suficiente
data = yf.download(all_tickers, start="2019-12-01", end="2024-12-01", progress=False)['Close']
returns_all = data.pct_change().dropna()

def calc_var(s): return s.quantile(0.05)
def calc_cvar(s): return s[s <= s.quantile(0.05)].mean()

resumen_cvar_max = []
backtesting_results = []

# --- PROCESAMIENTO CORE ---
for fecha in fechas_objetivo:
    f_dt = pd.to_datetime(fecha)
    f_inicio, f_fin = f_dt - pd.DateOffset(months=2), f_dt + pd.DateOffset(months=2)
    # Margen técnico de 21 días bursátiles (aprox 30 días calendario) para el primer cálculo rolling
    f_margen = f_inicio - timedelta(days=32)

    cvar_event = {}

    for p_nombre, tickers in portafolios_def.items():
        # Ponderación 1 acción al inicio del periodo
        p_inicio_precios = data.asof(f_inicio)[tickers]
        weights = p_inicio_precios / p_inicio_precios.sum()

        # Retornos desde el margen para que el rolling esté listo en f_inicio
        ret_full = (returns_all[tickers] * weights).sum(axis=1).loc[f_margen:f_fin]

        # Cálculos Rolling
        v_roll = ret_full.rolling(21).apply(calc_var).loc[f_inicio:f_fin].dropna()
        cv_roll = ret_full.rolling(21).apply(calc_cvar).loc[f_inicio:f_fin].dropna()

        # 2) Análisis CVaR Máximo (Mínimo valor matemático = máxima pérdida)
        cvar_event[p_nombre] = cv_roll.min()

        # 4) Backtesting: Violaciones y Tasa Empírica
        ret_eval = ret_full.loc[v_roll.index]
        n_viol = (ret_eval < v_roll).sum()
        tasa_fallo = n_viol / len(ret_eval)

        backtesting_results.append({
            'Fecha_Evento': fecha,
            'Portafolio': p_nombre,
            'Violaciones': n_viol,
            'Muestra_N': len(ret_eval),
            'Tasa_Empirica_Fallo': f"{tasa_fallo:.2%}"
        })

    resumen_cvar_max.append(pd.Series(cvar_event, name=fecha))

# --- TABLA 2) CVAR MÁXIMO 7x5 ---
df_cvar_max = pd.DataFrame(resumen_cvar_max)
df_cvar_max.loc['PROMEDIO'] = df_cvar_max.mean()
print("2) ANÁLISIS CVaR 95% MÁXIMO (MAYOR PÉRDIDA ESPERADA EN EL HORIZONTE)\n", df_cvar_max)

# --- GRÁFICO 3) EVOLUCIÓN CVAR (3x2) SOBREPUESTOS ---
fig3, axes3 = plt.subplots(3, 2, figsize=(16, 15))
axes3 = axes3.flatten()

for i, fecha in enumerate(fechas_objetivo):
    ax = axes3[i]
    f_dt = pd.to_datetime(fecha)
    f_inicio, f_fin = f_dt - pd.DateOffset(months=2), f_dt + pd.DateOffset(months=2)
    f_margen = f_inicio - timedelta(days=32)

    for p_nombre, tickers in portafolios_def.items():
        p_ini = data.asof(f_inicio)[tickers]
        w = p_ini / p_ini.sum()
        cv = (returns_all[tickers] * w).sum(axis=1).loc[f_margen:f_fin].rolling(21).apply(calc_cvar).loc[f_inicio:f_fin]
        ax.plot(cv.index, cv, label=p_nombre, alpha=0.8)

    ax.set_title(f"Evolución CVaR 95% - Horizonte {fecha}", fontweight='bold')
    ax.grid(False)
    for s in ax.spines.values(): s.set_edgecolor('black')
    ax.legend(loc='best', fontsize='x-small')

plt.tight_layout()
plt.show()

# --- GRÁFICO 5) BACKTESTING 24 GRÁFICOS (6x4) ---
fig5, axes5 = plt.subplots(6, 4, figsize=(20, 26))

for r, fecha in enumerate(fechas_objetivo):
    f_dt = pd.to_datetime(fecha)
    f_inicio, f_fin = f_dt - pd.DateOffset(months=2), f_dt + pd.DateOffset(months=2)
    f_margen = f_inicio - timedelta(days=32)

    for c, (p_nombre, tickers) in enumerate(portafolios_def.items()):
        ax = axes5[r, c]
        p_ini = data.asof(f_inicio)[tickers]
        w = p_ini / p_ini.sum()
        ret_p = (returns_all[tickers] * w).sum(axis=1).loc[f_inicio:f_fin]
        # Cálculo de VaR usando el margen de 21 días previo
        v_roll = (returns_all[tickers] * w).sum(axis=1).loc[f_margen:f_fin].rolling(21).apply(calc_var).loc[f_inicio:f_fin]

        # Sincronización para violaciones
        ret_s, var_s = ret_p.align(v_roll, join='inner')
        viol = ret_s[ret_s < var_s]

        ax.plot(ret_p.index, ret_p, color='silver', alpha=0.3, label='Retornos')
        ax.plot(v_roll.index, v_roll, color='blue', alpha=0.7, label='VaR 95%')
        ax.scatter(viol.index, viol, color='red', s=20, label='Violación')

        ax.set_title(f"{fecha} | {p_nombre}", fontsize=9, fontweight='bold')
        ax.grid(False)
        for s in ax.spines.values(): s.set_edgecolor('black')
        if r == 0 and c == 0: ax.legend(fontsize='xx-small')

plt.tight_layout()
plt.show()

# 4) Resumen textual detallado
print("\n4) ANÁLISIS DE BACKTESTING (VIOLACIONES Y TASA EMPÍRICA):")
print(pd.DataFrame(backtesting_results).to_string(index=False))
