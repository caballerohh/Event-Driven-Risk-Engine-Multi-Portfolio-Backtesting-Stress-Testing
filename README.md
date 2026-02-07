# 📉 Event-Driven-Risk-Engine-Multi-Portfolio-Backtesting-Stress-Testing

This repository features a specialized quantitative framework designed to evaluate portfolio resilience during historical stress events (e.g., pandemic shocks, geopolitical conflicts). The engine automates the validation of risk models through systematic backtesting and visualizes tail risk evolution across regional and sectoral portfolios.

🎯 **Objective:** To quantify the impact of tail risk events on diversified portfolios and validate the accuracy of Value at Risk (VaR) and Conditional VaR (CVaR) models.

---

## 📖 Extended Overview
The system analyzes multiple thematic portfolios (Peru, South America, North America, and Global) during specific high-volatility windows. It utilizes a **"look-back" window of 252 trading days** to calibrate risk thresholds and a **40-day "stress-window"** to monitor model breaches (violations), providing a rigorous statistical audit of downside risk protection.

### 🎯 Key Objectives of the Analysis
* **Event-Specific Stress Testing:** Isolation of key historical dates to analyze asset behavior during market dislocations (e.g., March 2020, October 2023).
* **Dynamic Risk Backtesting:** Implementation of Rolling VaR (95%) to track the frequency and magnitude of violations, identifying potential model underestimation.
* **Tail Risk Quantification (CVaR):** Evaluation of Expected Shortfall (Conditional VaR) to measure the average loss in the worst 5% of cases, offering a deeper view than standard VaR.
* **Regional Portfolio Comparison:** Comparative analysis of risk metrics between specialized portfolios, such as the P_Peru (BVN, SCCO) vs. P_Norteamerica (FCX, HL, etc.).

---

## 🔍 Portfolios & Assets Analyzed
The engine processes a diverse universe of mining and industrial equities across different jurisdictions:

* **🇵🇪 Peru:** BVN (Buenaventura), SCCO (Southern Copper).
* **🌎 Sudamerica:** FM.TO, HBM, SCCO, GMEXICOB.MX.
* **🇺🇸 Norteamerica:** FCX (Freeport-McMoRan), HL (Hecla Mining), HBM, FM.TO.
* **🌐 Grupal:** BHP, GLEN.L, FCX, FM.TO, RIO.

---

## 📊 Key Portfolio & Risk Results
* **Tail Risk Sensitivity:** Identification of heightened CVaR levels during specific "Event Windows," allowing for the ranking of portfolios by their defensive capabilities.
* **Model Integrity:** Through the Backtesting Graph, the system identifies clusters of VaR violations, signaling periods where market volatility exceeded statistical expectations (e.g., early 2020 shocks).
* **Diversification Benefit:** Analysis of the "P_Grupal" vs. regional portfolios to quantify the reduction in idiosyncratic risk through global asset allocation.



---

## 🛠️ Code Structure & Pipeline

### 1. Data Processing Layer
* Automated retrieval of global tickers via **yfinance** with adaptive date handling for different international exchanges.

### 2. Risk Calculation Engine
* **Rolling Quantile:** `rolling().quantile(0.05)` for dynamic VaR estimation.
* **Conditional VaR:** Calculated as the mean of returns exceeding the VaR threshold.

### 3. Backtesting Module
* Identification of **"Violations"** where actual returns fall below the predicted VaR, marked with specialized markers in the visual reports.

### 4. Visualization Suite
* **CVaR Overlap:** Synchronized plotting of multiple portfolios during the same event window for direct comparison.
* **Backtesting Scatter:** High-contrast visualization of outliers and model failures.

---

## 🚀 Technologies & Concepts Used
* **Quantitative Risk Management:** VaR Backtesting, Conditional VaR (Expected Shortfall).
* **Stress Testing:** Event-Driven Analysis, Historical Simulation.
* **Python Stack:** Pandas (Time-series manipulation), NumPy (Statistical vectorization), Matplotlib (Advanced risk plotting).
* **Asset Classes:** Global Mining Equities, Base & Precious Metals (Copper, Gold, Silver).

---

## 📧 Contact & Networking
📩 **Open to connecting and exploring new opportunities.** 📧 **carloscaballeromc@gmail.com**
