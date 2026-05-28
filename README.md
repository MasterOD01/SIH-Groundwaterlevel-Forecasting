# Groundwater Level Forecasting: Production-Grade ML Pipeline

[![SIH 2025](https://img.shields.io/badge/SIH-2025-blue.svg)](https://www.sih.gov.in/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Model](https://img.shields.io/badge/Models-STL--ARIMA%20%7C%20LightGBM-success.svg)]()
[![Explainability](https://img.shields.io/badge/Explainability-SHAP-orange.svg)]()

> **Problem Statement ID:** SIH25068  
> **Theme:** Miscellaneous | **Category:** Software  
> **Team ID:** 86211 | **Team Name:** Placeholder  

This repository contains the core Machine Learning research and modeling pipeline for real-time groundwater resource evaluation. Validated on the **Barnala DWLR station (Punjab)**, this pipeline supersedes standard 6-hourly SARIMAX prototypes by implementing daily spatial resampling, hydrologically-grounded feature engineering (crop calendars, antecedent rainfall), and a two-tier model hierarchy validated via walk-forward cross-validation.

---

## 📂 Repository Structure

```text
sih25068-ml-pipeline/
├── Barnala_dataset.csv             # Raw 6-hourly DWLR data for validation
├── README.md                       # Project documentation
├── SIH_DWLR_Improved.ipynb         # Main execution pipeline and research notebook
└── requirements.txt                # Python environment dependencies
