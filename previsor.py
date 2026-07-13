import pandas as pd
import streamlit as st
from  sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@st.cache_resource
def treinar_modelo():
    df = pd.read_csv(r'Data_and_analysis\KaggleV2-May-2016.csv')

    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.normalize()
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.normalize()

    df['Dias_Espera'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

    df['No-show'] = df['No-show'].apply(lambda x:1 if x=='Yes' else 0)

    df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

    features = [
    'Age', 'Gender', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'Dias_Espera'
    ]

    X = df[features]
    y = df['No-show']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia Geral do Modelo: {acuracia * 100:.2f}%")

    print("\n==== RELATÓRIO DE CLASSIFICAÇÃO ====")
    print(classification_report(y_test, y_pred, target_names=['Compareceu (0)', 'Faltou (1)']))


    print("==== MATRIZ DE CONFUSÃO ====")
    print(confusion_matrix(y_test, y_pred))

    return modelo

modelo = treinar_modelo()

st.title(':red MODELO DE PREVISÃO DE FALTA EM CONSULTAS', text_alignment='center')
st.write('Este modelo foi treinado com uma base de dados do Kaggle que leva em conta multifatores de pacientes que faltaram em consultas no município de Vitória - ES')

st.divider()

idade = st.number_input('IDADE DO PACIENTE', min_value= 0, max_value= 110, step= 1)
sexo = st.radio(
    'SEXO DO PACIENTE',
    options=[0, 1],
    format_func=lambda x: 'Masculino' if x == 0 else 'Feminino'
)
ss = st.radio(
    'O PACIENTE RECEBE BENEFÍCIO BOLSA FAMÍLIA?',
    options=[1, 0],
    format_func=lambda x: 'Sim' if x == 1 else 'Não'
)
ht = st.radio(
    'O PACIENTE É HIPERTENSO?',
    options=[1, 0],
    format_func=lambda x: 'Sim' if x == 1 else 'Não'
)
dbt= st.radio(
    'O PACIENTE É DIABÉTICO?',
    options=[1, 0],
    format_func=lambda x: 'Sim' if x == 1 else 'Não'
)
ach = st.radio(
    'O PACIENTE É ÁLCOOLATRA?',
    options=[1, 0],
    format_func=lambda x: 'Sim' if x == 1 else 'Não'
)
sms = st.radio(
    'O PACIENTE RECEBEU SMS AVISANDO SOBRE A CONSULTA?',
    options=[1, 0],
    format_func=lambda x: 'Sim' if x == 1 else 'Não'
)
tesp = st.number_input('QUANTOS DIAS O PACIENTE ESPERARÁ ATÉ REALIZAR A CONSULTA?', min_value= 0, max_value= 365, step= 1)

paciente=[idade, sexo, ss, ht, dbt, ach, sms, tesp]

risco_falta = modelo.predict_proba([paciente])[0][1]
tr = risco_falta * 100

if st.button('Calcular probabilidade de falta do paciente'):
    st.divider()
    st.write(f"Alerta de Triagem: O risco deste paciente faltar é de {tr:.2f}%")
    
    if tr < 15:
        st.success('O risco do paciente faltar é baixo, recomenda-se a confirmação da consulta via SMS!')

    elif tr <=30:
        st.warning('O risco do paciente faltar é médio, recomenda-se a confirmação da consulta via SMS e via ligação telefônica alguns dias antes da consulta!')
    
    else:
        st.error('O risco do paciente faltar é alto, recomenda-se a confirmação via SMS, ligação telefônica e visita do Agente de Saúde!')