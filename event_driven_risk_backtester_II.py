#CODE-5 PART 2

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

# Descarga de datos
data = yf.download(all_tickers, start="2019-12-01", end="2024-12-01", progress=False)['Close']
returns_all = data.pct_change().dropna()

def calc_var(s): return s.quantile(0.05)
def calc_cvar(s): return s[s <= s.quantile(0.05)].mean()

resumen_cvar_detallado = []
backtesting_final = []

# --- PROCESAMIENTO ---
for fecha in fechas_objetivo:
    f_dt = pd.to_datetime(fecha)
    f_inicio, f_fin = f_dt - pd.DateOffset(months=2), f_dt + pd.DateOffset(months=2)
    f_margen = f_inicio - timedelta(days=32)

    dict_cvar_max = {}

    for p_nombre, tickers in portafolios_def.items():
        # Ponderación 1 acción
        p_ini_precios = data.asof(f_inicio)[tickers]
        weights = p_ini_precios / p_ini_precios.sum()

        # Retornos con margen para el Rolling
        serie_ret = (returns_all[tickers] * weights).sum(axis=1).loc[f_margen:f_fin]

        # Cálculos de Riesgo
        v_roll = serie_ret.rolling(21).apply(calc_var).loc[f_inicio:f_fin].dropna()
        cv_roll = serie_ret.rolling(21).apply(calc_cvar).loc[f_inicio:f_fin].dropna()

        # IDENTIFICAR MÁXIMO RIESGO (Valor más bajo del CVaR)
        cvar_max_val = cv_roll.min()
        cvar_max_fecha = cv_roll.idxmin() # Fecha del pico de riesgo
        dict_cvar_max[p_nombre] = cvar_max_val

        # Backtesting
        ret_eval, var_eval = serie_ret.align(v_roll, join='inner')
        n_viol = (ret_eval < var_eval).sum()
        tasa_fallo = n_viol / len(ret_eval)

        backtesting_final.append({
            'Evento': fecha, 'Portafolio': p_nombre, 'Violaciones': n_viol,
            'TEF': f"{tasa_fallo:.2%}", 'Fecha_Pico_CVaR': cvar_max_fecha.date()
        })

    resumen_cvar_detallado.append(pd.Series(dict_cvar_max, name=fecha))

# --- SALIDA DE TABLAS ---
df_cvar_max = pd.DataFrame(resumen_cvar_detallado)
df_cvar_max.loc['PROMEDIO'] = df_cvar_max.mean()

print("\n2) ANÁLISIS CVaR 95% MÁXIMO (Punto más bajo de pérdida esperada)")
print(df_cvar_max)

print("\n4) RESUMEN DE BACKTESTING Y LOCALIZACIÓN DEL PICO DE RIESGO:")
print(pd.DataFrame(backtesting_final).to_string(index=False))

# --- GRÁFICOS (Se omiten aquí por brevedad, pero siguen la lógica anterior de 3x2 y 6x4) ---
