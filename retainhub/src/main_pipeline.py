"""RetainHub: Pipeline de alerta temprana para la fidelización de clientes."""

import os
import joblib
import kagglehub
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean():
    """Descarga el dataset de Kaggle y aplica limpieza básica."""
    path = kagglehub.dataset_download(
        "salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data"
    )
    dataset = None
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            dataset = os.path.join(dirname, filename)

    datos = pd.read_csv(dataset)
    print(f"Dataset cargado: {datos.shape[0]} registros, {datos.shape[1]} columnas.")

    datos = datos.drop_duplicates()
    datos.columns = [col.lower().replace(' ', '_').strip() for col in datos.columns]

    if datos['purchase_amount'].dtype == 'object':
        datos['purchase_amount'] = (
            datos['purchase_amount']
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
    datos['purchase_amount'] = pd.to_numeric(datos['purchase_amount'], errors='coerce')
    datos = datos.dropna(subset=['purchase_amount'])
    datos = datos[datos['purchase_amount'] > 0]
    datos = datos[(datos['age'] >= 18) & (datos['age'] <= 100)]

    for col in datos.select_dtypes(include=['object']).columns:
        datos[col] = datos[col].str.strip()
    datos.replace(['?', 'n/a', 'Unknown', 'unknown', 'None'], np.nan, inplace=True)

    datos['time_of_purchase'] = pd.to_datetime(datos['time_of_purchase'])
    for col in ['gender', 'location', 'purchase_category']:
        if col in datos.columns:
            datos[col] = datos[col].astype(str).str.title()

    print(f"Limpieza completada. Registros finales: {datos.shape[0]}")
    return datos


def rfm_segmentation(datos):
    """Calcula métricas RFM y segmenta clientes. Entregable principal del proyecto."""
    fecha_ref = datos['time_of_purchase'].max() + pd.Timedelta(days=1)

    rfm = datos.groupby('customer_id').agg(
        recencia=('time_of_purchase', lambda x: (fecha_ref - x.max()).days),
        frecuencia=('purchase_amount', 'count'),
        monetario=('purchase_amount', 'sum')
    ).reset_index()

    r_pct = rfm['recencia'].rank(pct=True, method='first')
    f_pct = rfm['frecuencia'].rank(pct=True, method='first')
    m_pct = rfm['monetario'].rank(pct=True, method='first')

    rfm['R'] = r_pct.apply(
        lambda x: 1 if x <= 0.2 else 2 if x <= 0.4 else 3 if x <= 0.6 else 4 if x <= 0.8 else 5
    )
    rfm['R'] = 6 - rfm['R']
    rfm['F'] = f_pct.apply(
        lambda x: 1 if x <= 0.2 else 2 if x <= 0.4 else 3 if x <= 0.6 else 4 if x <= 0.8 else 5
    )
    rfm['M'] = m_pct.apply(
        lambda x: 1 if x <= 0.2 else 2 if x <= 0.4 else 3 if x <= 0.6 else 4 if x <= 0.8 else 5
    )
    rfm['rfm_score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

    def segmentar(score):
        if score[0] >= '4' and score[1] >= '4':
            return 'Campeones'
        if score[0] >= '3' and score[2] >= '3':
            return 'Leales / Potenciales'
        if score[0] <= '2':
            return 'En Riesgo / Perdidos'
        return 'Otros'

    rfm['segmento_cliente'] = rfm['rfm_score'].apply(segmentar)

    print("\nSegmentación RFM completada:")
    print(rfm['segmento_cliente'].value_counts().to_string())

    # Top 100: clientes "En Riesgo" ordenados por mayor recencia (más días sin comprar)
    top_100 = (
        rfm[rfm['segmento_cliente'] == 'En Riesgo / Perdidos']
        .sort_values('recencia', ascending=False)
        .head(100)[['customer_id', 'recencia', 'frecuencia', 'monetario', 'rfm_score']]
    )
    top_100.to_csv('top_100_riesgo_clientes.csv', index=False)
    rfm.to_csv('rfm_segmentos.csv', index=False)

    print(f"Top 100 clientes en riesgo exportados a 'top_100_riesgo_clientes.csv'.")
    print(f"Segmentación completa exportada a 'rfm_segmentos.csv'.")
    return rfm


def train_predictive_model(datos, rfm):
    """Entrena un modelo ML experimental usando variables de comportamiento."""
    print("\n[NOTA] El modelo predictivo tiene un carácter experimental.")
    print("El dataset es sintético con variables independientes entre sí,")
    print("por lo que el rendimiento esperado es cercano al azar (AUC ~0.5).")
    print("El entregable principal del proyecto es la segmentación RFM.")

    datos_merged = datos.merge(
        rfm[['customer_id', 'segmento_cliente']], on='customer_id', how='left'
    )
    datos_merged['churn'] = (
        datos_merged['segmento_cliente'] == 'En Riesgo / Perdidos'
    ).astype(int)

    # Features: variables de comportamiento, excluimos recencia/frecuencia
    # (que definen el segmento RFM) y columnas no informativas
    cols_excluir = {
        'customer_id', 'time_of_purchase', 'location',
        'segmento_cliente', 'churn'
    }
    cols_cat_encode = [c for c in [
        'gender', 'income_level', 'marital_status', 'education_level',
        'occupation', 'purchase_category', 'purchase_channel', 'payment_method',
        'shipping_preference', 'discount_used', 'customer_loyalty_program_member',
        'social_media_influence', 'discount_sensitivity', 'engagement_with_ads',
        'device_used_for_shopping', 'purchase_intent'
    ] if c in datos_merged.columns and c not in cols_excluir]

    datos_features = datos_merged.drop(columns=list(cols_excluir), errors='ignore')
    datos_features = pd.get_dummies(datos_features, columns=cols_cat_encode, drop_first=True)
    datos_features = datos_features.select_dtypes(include=[np.number, 'bool'])
    datos_features = datos_features.fillna(datos_features.median())

    scaler = StandardScaler()
    x_scaled = pd.DataFrame(
        scaler.fit_transform(datos_features), columns=datos_features.columns
    )
    y_target = datos_merged['churn'].values

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_target, test_size=0.2, random_state=42, stratify=y_target
    )
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(x_train_res, y_train_res)
    auc_rf = roc_auc_score(y_test, rf.predict_proba(x_test)[:, 1])
    print(f"\n[Random Forest] AUC-ROC: {auc_rf:.4f}")
    print(classification_report(y_test, rf.predict(x_test), zero_division=0))

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(x_train_res, y_train_res)
    auc_lr = roc_auc_score(y_test, lr.predict_proba(x_test)[:, 1])
    print(f"[Regresión Logística] AUC-ROC: {auc_lr:.4f}")
    print(classification_report(y_test, lr.predict(x_test), zero_division=0))

    mejor_modelo = rf if auc_rf >= auc_lr else lr
    joblib.dump(mejor_modelo, 'retainhub_model_v1.pkl')
    print("Modelo guardado como 'retainhub_model_v1.pkl'.")


def run_pipeline():
    """Ejecuta el pipeline completo de RetainHub."""
    print("=" * 60)
    print("  RetainHub: Sistema de Alerta Temprana para Churn")
    print("=" * 60)

    print("\n--- [1/3] Ingesta y Limpieza de Datos ---")
    datos = load_and_clean()

    print("\n--- [2/3] Segmentación RFM (Entregable Principal) ---")
    rfm = rfm_segmentation(datos)

    print("\n--- [3/3] Modelo Predictivo (Experimental) ---")
    train_predictive_model(datos, rfm)

    print("\n" + "=" * 60)
    print("  Pipeline finalizado con éxito.")
    print("  Archivos generados:")
    print("    - top_100_riesgo_clientes.csv  (entregable principal)")
    print("    - rfm_segmentos.csv            (segmentación completa)")
    print("    - retainhub_model_v1.pkl       (modelo experimental)")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
