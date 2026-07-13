import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'Data_and_analysis\KaggleV2-May-2016.csv')

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.normalize()
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.normalize()
df['Dias_Espera'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

df = df[df['Dias_Espera'] >= 0]

df['No-show_Num'] = df['No-show'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Gender_Num'] = df['Gender'].map({'M': 0, 'F': 1})

print("==== ANÁLISE COMPLETA DOS FATORES DE ABSENTEÍSMO ====\n")

taxa_geral = df['No-show_Num'].mean() * 100
print(f"Taxa Geral de Faltas no SUS (Vitória-ES): {taxa_geral:.2f}%\n")

def faixas_espera(dias):
    if dias == 0: return 'Mesmo Dia'
    elif dias <= 3: return '1 a 3 dias'
    elif dias <= 7: return '4 a 7 dias'
    elif dias <= 15: return '8 a 15 dias'
    else: return 'Mais de 15 dias'

df['Faixa_Espera'] = df['Dias_Espera'].apply(faixas_espera)
taxa_por_espera = df.groupby('Faixa_Espera')['No-show_Num'].mean() * 100
print("Taxa de faltas por tempo de espera:")
print(taxa_por_espera.round(2), "\n")

def faixas_idade(idade):
    if idade <= 12: return '00 a 12 anos (Criança)'
    elif idade <= 24: return '13 a 24 anos (Jovem)'
    elif idade <= 45: return '25 a 45 anos (Adulto Jovem)'
    elif idade <= 60: return '46 a 60 anos (Adulto)'
    else: return 'Mais de 60 anos (Idoso)'

df['Faixa_Idade'] = df['Age'].apply(faixas_idade)
taxa_por_idade = df.groupby('Faixa_Idade')['No-show_Num'].mean() * 100

print("Taxa de faltas por Faixa Etária:")
print(taxa_por_idade.round(2), "\n")

fatores = ['Scholarship','Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'Gender_Num']
print("Taxa de falta cruzada por condições (1 = Sim, 0 = Não):")
for fator in fatores:
    cruzamento = df.groupby(fator)['No-show_Num'].mean() * 100
    print(f"\n--- Análise do fator: {fator} ---")
    print(cruzamento.round(2))

features = ['Age', 'Gender_Num', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'Dias_Espera']
X = df[features]
y = df['No-show_Num']

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

importancias = pd.DataFrame({
    'Fator': features,
    'Importância (%)': modelo.feature_importances_ * 100
}).sort_values(by='Importância (%)', ascending=False)

print("\n==== RANKING DE INFLUÊNCIA (FEATURE IMPORTANCE) ====")
print(importancias.round(2))

plt.figure(figsize=(10, 5))
sns.barplot(x='Importância (%)', y='Fator', data=importancias, palette='viridis')
plt.title('Quais fatores mais pesam na decisão do Paciente faltar?')
plt.xlabel('Impacto Percentual no Algoritmo')
plt.ylabel('Fator Analisado')
plt.tight_layout()
plt.savefig('importancia_fatores.png') 
plt.show()