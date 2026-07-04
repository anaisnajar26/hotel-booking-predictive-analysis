#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOTEL RESERVATIONS — Pipeline Completo de Machine Learning
===========================================================
Preparación de datos, Modelado Ensamble, Redes Neuronales y Simulación de Negocio
Autora: anaisnajar26
"""

# ==============================================================================
# 1. LIBRERÍAS Y CONFIGURACIÓN GLOBAL
# ==============================================================================
import warnings
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn: Preprocesamiento y Splits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sklearn: Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Modelos Avanzados de Gradiente
import xgboost as xgb
from lightgbm import LGBMClassifier

# Sklearn: Métricas de Evaluación
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Explicabilidad del Modelo
import shap

# Configuración
warnings.filterwarnings("ignore")
N_JOBS = multiprocessing.cpu_count()

# ==============================================================================
# HELPER: Función de evaluación unificada
# ==============================================================================
def evaluar_modelo(nombre, modelo, X_tr, y_tr, X_v, y_v, mostrar_importancias=False):
    """Entrena, predice y muestra métricas completas para un modelo."""
    print("\n" + "=" * 55)
    print(f"   {nombre}")
    print("=" * 55)

    modelo.fit(X_tr, y_tr)

    y_hat_tr = modelo.predict(X_tr)
    y_hat_v  = modelo.predict(X_v)

    metricas = {
        'Accuracy' : (accuracy_score,  {}),
        'Precision': (precision_score, {'zero_division': 0}),
        'Recall'   : (recall_score,    {'zero_division': 0}),
        'F1'       : (f1_score,        {'zero_division': 0}),
    }

    resultados = {}
    print(f"   {'Métrica':<12} {'Train':>8} {'Val':>8}  {'Brecha':>8}")
    print(f"   {'-'*40}")
    for nombre_m, (fn, kwargs) in metricas.items():
        tr = fn(y_tr, y_hat_tr, **kwargs)
        v  = fn(y_v,  y_hat_v,  **kwargs)
        print(f"   {nombre_m:<12} {tr:>8.4f} {v:>8.4f}  {tr-v:>+8.4f}")
        resultados[nombre_m] = {'train': tr, 'val': v}

    print(f"\n   Informe completo (Val):")
    print(classification_report(y_v, y_hat_v,
                                target_names=['No cancela', 'Cancela'],
                                zero_division=0))

    if mostrar_importancias and hasattr(modelo, 'feature_importances_'):
        imp = (
            pd.DataFrame({'feature': X_tr.columns,
                          'importance': modelo.feature_importances_})
            .sort_values('importance', ascending=False)
            .head(5)
        )
        print("   Top 5 variables más importantes:")
        print(imp.to_string(index=False))

    return resultados, modelo

# ==============================================================================
# 2. CARGA DE DATOS
# ==============================================================================
df = pd.read_csv("Hotel reservations c.csv", sep=";", decimal=",")

print("=" * 60)
print("2. DATOS CARGADOS")
print("=" * 60)
print(f" Filas: {df.shape[0]:,}  |  Columnas: {df.shape[1]}")
print(f" Nulos: {df.isnull().sum().sum()}")
print(f" Duplicados: {df.duplicated().sum()}")

# ==============================================================================
# 3. CONSTRUCCIÓN Y VALIDACIÓN DE FECHAS
# ==============================================================================
df['date'] = pd.to_datetime(
    df[['arrival_year', 'arrival_month', 'arrival_date']].rename(
        columns={'arrival_year': 'year', 'arrival_month': 'month', 'arrival_date': 'day'}
    ),
    errors='coerce'
)

fechas_invalidas = df['date'].isna().sum()
print("\n" + "=" * 60)
print("3. VALIDACIÓN DE FECHAS")
print("=" * 60)
print(f" Fechas inválidas (NaT) eliminadas: {fechas_invalidas}")
df = df.dropna(subset=['date']).copy()
print(f" Filas restantes: {len(df):,}")

# ==============================================================================
# 4. LIMPIEZA DE FILAS SOSPECHOSAS
# ==============================================================================
print("\n" + "=" * 60)
print("4. LIMPIEZA DE FILAS SOSPECHOSAS")
print("=" * 60)

mask_sin_noches = (df['no_of_weekend_nights'] + df['no_of_week_nights']) == 0
print(f" Reservas sin noches eliminadas: {mask_sin_noches.sum()}")
df = df[~mask_sin_noches].copy()

mask_sin_adultos = df['no_of_adults'] == 0
print(f" Reservas sin adultos eliminadas: {mask_sin_adultos.sum()}")
df = df[~mask_sin_adultos].copy()

mask_precio_cero_no_comp = ((df['avg_price_per_room'] == 0) & (df['market_segment_type'] != 'Complementary'))
print(f" Precios = 0 en seg. no-complementario eliminados: {mask_precio_cero_no_comp.sum()}")
df = df[~mask_precio_cero_no_comp].copy()
print(f" Filas tras limpieza: {len(df):,}")

# ==============================================================================
# 5. VARIABLE OBJETIVO Y TRATAMIENTO DEL DESBALANCE
# ==============================================================================
df['booking_status_num'] = df['booking_status'].replace({'Not_Canceled': 0, 'Canceled': 1})

print("\n" + "=" * 60)
print("5. VARIABLE OBJETIVO")
print("=" * 60)
conteo = df['booking_status_num'].value_counts()
total  = len(df)
print(f" No cancela (0): {conteo[0]:,}  ({conteo[0]/total:.1%})")
print(f" Cancela    (1): {conteo[1]:,}  ({conteo[1]/total:.1%})")

# ==============================================================================
# 6. INGENIERÍA DE CARACTERÍSTICAS Y ENCODING
# ==============================================================================
df_trees = pd.concat([
    df,
    pd.get_dummies(df['market_segment_type'], drop_first=True, prefix='market'),
    pd.get_dummies(df['room_type_reserved'],   drop_first=True, prefix='room'),
    pd.get_dummies(df['type_of_meal_plan'],    drop_first=True, prefix='meal'),
], axis=1)

df_trees.sort_values('date', ascending=True, inplace=True)
df_trees.reset_index(drop=True, inplace=True)

# Features temporales y de huéspedes
df_trees['arrival_month_num']   = df_trees['date'].dt.month
df_trees['arrival_day_of_week'] = df_trees['date'].dt.dayofweek
df_trees['arrival_is_weekend']  = df_trees['date'].dt.dayofweek.isin([5, 6]).astype(int)
df_trees['total_nights'] = df_trees['no_of_weekend_nights'] + df_trees['no_of_week_nights']
df_trees['total_guests'] = df_trees['no_of_adults'] + df_trees['no_of_children']
df_trees['tiene_ninos']  = (df_trees['no_of_children'] > 0).astype(int)

# Lag de precio
df_trees['precio_lag1'] = df_trees.groupby('market_segment_type')['avg_price_per_room'].shift(1)
df_trees['tiene_lag_precio'] = df_trees['precio_lag1'].notna().astype(int)
media_por_grupo = df_trees.groupby('market_segment_type')['avg_price_per_room'].transform('mean')
df_trees['precio_lag1'] = df_trees['precio_lag1'].fillna(media_por_grupo)

# ==============================================================================
# 7. SPLIT TEMPORAL (60% Train | 25% Val | 15% Test)
# ==============================================================================
pct_val  = 0.25
pct_test = 0.15

columnas_a_borrar = [
    'Booking_ID', 'arrival_year', 'arrival_month', 'arrival_date', 'date',
    'booking_status', 'market_segment_type', 'room_type_reserved', 'type_of_meal_plan'
]
df_final = df_trees.drop(columns=columnas_a_borrar)

X = df_final.drop(columns=['booking_status_num'])
y = df_final['booking_status_num']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=pct_test, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=pct_val / (1 - pct_test), shuffle=False)

# Evitar Data Leakage mapeando medias exclusivamente de Train
seg_col   = 'market_segment_type'
seg_train = df_trees.loc[X_train.index, seg_col]
seg_val   = df_trees.loc[X_val.index,   seg_col]
seg_test  = df_trees.loc[X_test.index,  seg_col]

media_train = X_train['avg_price_per_room'].groupby(seg_train).mean()

X_train = X_train.copy()
X_val   = X_val.copy()
X_test  = X_test.copy()

X_train['precio_market_avg'] = seg_train.map(media_train).values
X_val['precio_market_avg']   = seg_val.map(media_train).values
X_test['precio_market_avg']  = seg_test.map(media_train).values

for split in [X_train, X_val, X_test]:
    split['precio_over_market_avg'] = split['avg_price_per_room'] / split['precio_market_avg']

# Escalado de datos específico para Redes Neuronales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 8. INSTANCIACIÓN DE MODELOS
# ==============================================================================
modelo_cart = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight='balanced', n_jobs=N_JOBS, random_state=42)

ratio_clases = (y_train == 0).sum() / (y_train == 1).sum()
modelo_xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, 
                               colsample_bytree=0.8, scale_pos_weight=ratio_clases, eval_metric="logloss", 
                               early_stopping_rounds=15, n_jobs=N_JOBS, random_state=42)

modelo_lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=20, max_depth=5, 
                             class_weight='balanced', subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS, 
                             random_state=42, verbose=-1)

modelo_nn = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', 
                          alpha=0.01, max_iter=500, early_stopping=True, random_state=42)

print("\n✅ Todos los modelos (Árboles y Redes) instanciados correctamente.\n")

# ==============================================================================
# 9. ENTRENAMIENTO Y EVALUACIÓN EN VALIDACIÓN
# ==============================================================================
tabla_resultados = {}

# CART, RF y LightGBM mediante función helper
for nom, mod in [("CART", modelo_cart), ("Random Forest", modelo_rf), ("LightGBM", modelo_lgbm)]:
    res, _ = evaluar_modelo(f"🌲 {nom.upper()}", mod, X_train, y_train, X_val, y_val, mostrar_importancias=(nom != "LightGBM"))
    tabla_resultados[nom] = res

# XGBoost (requiere gestión de early stopping)
print("\n=======================================================\n 🚀 MODELO: XGBOOST\n=======================================================")
modelo_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_hat_tr_xgb, y_hat_v_xgb = modelo_xgb.predict(X_train), modelo_xgb.predict(X_val)

tabla_resultados['XGBoost'] = {
    'Accuracy': {'train': accuracy_score(y_train, y_hat_tr_xgb), 'val': accuracy_score(y_val, y_hat_v_xgb)},
    'Precision': {'train': precision_score(y_train, y_hat_tr_xgb, zero_division=0), 'val': precision_score(y_val, y_hat_v_xgb, zero_division=0)},
    'Recall': {'train': recall_score(y_train, y_hat_tr_xgb, zero_division=0), 'val': recall_score(y_val, y_hat_v_xgb, zero_division=0)},
    'F1': {'train': f1_score(y_train, y_hat_tr_xgb, zero_division=0), 'val': f1_score(y_val, y_hat_v_xgb, zero_division=0)}
}

# Red Neuronal Profunda (usa datos escalados)
res_nn, _ = evaluar_modelo("🧠 NEURAL NETWORK PROFUNDA", modelo_nn, X_train_scaled, y_train, X_val_scaled, y_val)
tabla_resultados['Neural Network'] = res_nn

# ==============================================================================
# 10. TABLA COMPARATIVA GENERAL EN VALIDACIÓN
# ==============================================================================
print("\n" + "=" * 60 + "\n  📊 TABLA COMPARATIVA — VALIDACIÓN\n" + "=" * 60)
filas = [{
    'Modelo': k, 'Accuracy': v['Accuracy']['val'], 'Precision': v['Precision']['val'],
    'Recall': v['Recall']['val'], 'F1': v['F1']['val'], 'Brecha(Acc)': v['Accuracy']['train'] - v['Accuracy']['val']
} for k, v in tabla_resultados.items()]

df_resultados = pd.DataFrame(filas).sort_values('F1', ascending=False).reset_index(drop=True)
df_resultados.index += 1
print(df_resultados.to_string())

ganador_nombre = df_resultados.iloc[0]['Modelo']
print(f"\n🏆 Mejor modelo en Validación (por F1): {ganador_nombre}")

# ==============================================================================
# 11. EXAMEN DEFINITIVO EN EL CONJUNTO DE TEST (MÉTRICAS FUTURAS)
# ==============================================================================
print("\n" + "=" * 60 + "\n 🏁 EXAMEN FINAL EN TEST PARA TODOS LOS MODELOS\n" + "=" * 60)
modelos_dict = {'CART': modelo_cart, 'Random Forest': modelo_rf, 'XGBoost': modelo_xgb, 'LightGBM': modelo_lgbm}

resultados_test = []
for nom, mod in modelos_dict.items():
    y_t_hat = mod.predict(X_test)
    resultados_test.append({
        'Modelo': nom, 'Accuracy': accuracy_score(y_test, y_t_hat), 'Precision': precision_score(y_test, y_t_hat, zero_division=0),
        'Recall': recall_score(y_test, y_t_hat, zero_division=0), 'F1': f1_score(y_test, y_t_hat, zero_division=0)
    })

# Añadir Red Neuronal con sus datos escalados correspondientes
y_t_hat_nn = modelo_nn.predict(X_test_scaled)
resultados_test.append({
    'Modelo': 'Neural Network', 'Accuracy': accuracy_score(y_test, y_t_hat_nn), 'Precision': precision_score(y_test, y_t_hat_nn, zero_division=0),
    'Recall': recall_score(y_test, y_t_hat_nn, zero_division=0), 'F1': f1_score(y_test, y_t_hat_nn, zero_division=0)
})

df_test_final = pd.DataFrame(resultados_test).sort_values('Precision', ascending=False).reset_index(drop=True)
df_test_final.index += 1
print(df_test_final.to_string())

# ==============================================================================
# 12. EXPLICABILIDAD GLOBAL Y LOCAL (SHAP VALUES CON LIGHTGBM)
# ==============================================================================
print("\n" + "=" * 60 + "\n 🧠 12. EXPLICABILIDAD DEL MODELO (SHAP)\n" + "=" * 60)
explainer = shap.TreeExplainer(modelo_lgbm)
shap_values = explainer(X_test)
shap_values_clase1 = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

# Gráfico 1: Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_clase1, X_test, show=False)
plt.title("Impacto Global de las Variables en las Cancelaciones (LightGBM)")
plt.tight_layout(); plt.show()

# Gráfico 2: Dependence Plot (lead_time)
if 'lead_time' in X_test.columns:
    shap.dependence_plot('lead_time', shap_values_clase1.values, X_test, show=False)
    plt.tight_layout(); plt.show()

# ==============================================================================
# 13. ESTRATEGIA DE NEGOCIO: THRESHOLD TUNING Y OVERBOOKING
# ==============================================================================
print("\n" + "=" * 60 + "\n 🎯 13. ESTRATEGIA DE NEGOCIO: AJUSTE DE THRESHOLD UMBRAL\n" + "=" * 60)
probabilidades_lgbm = modelo_lgbm.predict_proba(X_test)[:, 1]

print(f" {'Umbral':<8} | {'Precisión':<12} | {'Recall':<10} | {'Especificidad':<14}")
print("-" * 65)
for umbral in [0.50, 0.60, 0.70, 0.80, 0.90]:
    y_pred_temp = (probabilidades_lgbm >= umbral).astype(int)
    prec = precision_score(y_test, y_pred_temp, zero_division=0)
    rec  = recall_score(y_test, y_pred_temp, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
    especificidad = tn / (tn + fp)
    
    marcador = "<- NUESTRA ELECCIÓN" if umbral == 0.70 else ""
    print(f" {umbral:<8.2f} | {prec:>10.2%}   | {rec:>8.2%} | {especificidad:>12.2%} {marcador}")

# Repositorio de Habitaciones vacías con umbral 0.70
y_pred_70 = (probabilidades_lgbm >= 0.70).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_70).ravel()
print(f"\n 🛏️ Falsos Negativos (Habitaciones vacías imprevistas): {fn} de un total de {len(y_test)} reservas.")

# ==============================================================================
# 14. SIMULACIÓN FINANCIERA ORIENTADA A RIESGO REPUTACIONAL
# ==============================================================================
def simulador_beneficio_hotel(y_true, probabilidades, precio_hab=105, coste_reubicacion=700):
    print("\n" + "=" * 70 + "\n 💶 14. SIMULADOR ECONÓMICO: IMPACTO MONETARIO REAL\n" + "=" * 70)
    resultados_economicos = []
    factor_escala = 100 / len(y_true)
    
    for umbral in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        y_pred = (probabilidades >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        beneficio = ((tn * factor_escala) * precio_hab) + ((tp * factor_escala) * precio_hab) + ((fp * factor_escala) * (precio_hab - coste_reubicacion))
        resultados_economicos.append({
            'Umbral': umbral, 'Hab. Llenas': f"{(tn+tp)*factor_escala:.1f}", 
            'Overbookings (FP)': f"{fp*factor_escala:.1f}", 'BENEFICIO TOTAL': beneficio
        })

    df_eco = pd.DataFrame(resultados_economicos)
    df_eco['BENEFICIO TOTAL'] = df_eco['BENEFICIO TOTAL'].apply(lambda x: f"{x:,.0f} €")
    print(df_eco.to_string(index=False))
    print("=" * 70)

simulador_beneficio_hotel(y_test, probabilidades_lgbm)