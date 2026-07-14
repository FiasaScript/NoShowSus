## SUS Vitória - Absenteísmo & Triagem de Risco

<p align="center">
  <img width="25%" src="https://img.shields.io/badge/FiasaScript-057019.svg?style=for-the-badge&logo=adobe&logoColor=white" />
</p>

<p align="center">
  <a href="#about-the-project">About</a> •
  <a href="#features">Features</a> •
  <a href="#how-to-run">How to Run</a> 
</p>

---

##  About the Project

This project uses a historical dataset of public health appointments in **Vitória, ES (Brazil)** to analyze and predict patient non-attendance (absenteeism) in the SUS (Unified Health System). Using a **Random Forest Classifier**, the system identifies key behavioral and clinical features contributing to missed appointments and provides an interactive triage tool for healthcare administrators.

> ⚠️ **Disclaimer:** This software is for educational and demonstration purposes. It does not replace professional medical or administrative triage protocols.

---

## Features

* **Feature Importance Analysis (`AnaliseDosDados.py`):** Explores how attributes like Age, Waiting Time (`Dias_Espera`), SMS reminders, and socioeconomic programs affect attendance rates.
* **Predictive Web Interface (`previsor.py`):** A Streamlit app where clinical staff can register a new patient, assess their probability of missing an appointment, and receive preventative, customized actions.
* **Actionable Countermeasures:** Automatically suggests tailored preventative steps (e.g., active phone calls, community health agent visits) based on patient risk stratification.

---

##  Key Variables Analyzed
* `Age` (Idade)
* `Gender_Num` (Gênero mapeado numericamente)
* `Scholarship` (Beneficiário do Bolsa Família)
* `Hipertension` (Paciente hipertenso)
* `Diabetes` (Paciente diabético)
* `Alcoholism` (Histórico de alcoolismo)
* `SMS_received` (Se recebeu SMS de confirmação)
* `Dias_Espera` (Tempo entre agendamento e consulta)

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/FiasaScript/vitoria-sus-absenteismo.git](https://github.com/FiasaScript/vitoria-sus-absenteismo.git)
   cd vitoria-sus-absenteismo

## Run through Stlit Community Cloud
```bash
https://noshowsus-ukudqtlz5rg67hjvkdasnu.streamlit.app
