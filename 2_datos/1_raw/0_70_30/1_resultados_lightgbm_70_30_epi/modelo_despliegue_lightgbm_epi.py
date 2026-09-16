"""
Script de despliegue del modelo de predicción de casos de dengue
usando SOLO predictores epidemiológicos (rezagos de dengue) con LightGBM.
Generado automáticamente - 2026-09-16 17:26:11
"""
import pandas as pd
import numpy as np
import pickle
import json
import os

class DengueEpiDeploymentLGBM:
    EPIDEMIOLOGICAL_PREDICTORS = ['fecha', 'semana_epi', 'casos_dengue_lag_1', 'casos_dengue_lag_2', 'casos_dengue_lag_3', 'casos_dengue_lag_4', 'casos_dengue_lag_5', 'casos_dengue_lag_6', 'casos_dengue_lag_7', 'casos_dengue_lag_8', 'casos_dengue_lag_9', 'casos_dengue_lag_10', 'casos_dengue_lag_11', 'casos_dengue_lag_12']

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.model_dir, 'lgbm_model.pkl'), 'rb') as f:
            self.lgbm_model = pickle.load(f)
        with open(os.path.join(self.model_dir, 'model_config.json'), 'r') as f:
            cfg = json.load(f)
            self.selected_features = cfg['selected_features']
            self.X_augmented_columns = cfg['X_augmented_columns']
            self.best_params = cfg.get('best_params', {{}})
        print("✅ Modelo LightGBM (epidemiológico) cargado")

    def _filter_epi(self, X):
        allowed = [c for c in X.columns if c in self.EPIDEMIOLOGICAL_PREDICTORS]
        return X[allowed].copy()

    def _lag_aggs(self, X):
        d = pd.DataFrame(index=X.index)
        lag_cols = [c for c in X.columns if c.startswith('casos_dengue_lag_')]
        if len(lag_cols) >= 3:
            l = X[lag_cols]
            d['dengue_lag_mean'] = l.mean(axis=1)
            d['dengue_lag_std'] = l.std(axis=1)
            d['dengue_lag_sum'] = l.sum(axis=1)
            d['dengue_lag_max'] = l.max(axis=1)
            d['dengue_lag_min'] = l.min(axis=1)
            d['dengue_lag_median'] = l.median(axis=1)
        return d

    def _trends(self, X):
        d = pd.DataFrame(index=X.index)
        if 'casos_dengue_lag_1' in X.columns and 'casos_dengue_lag_2' in X.columns:
            d['dengue_diff_1_2'] = X['casos_dengue_lag_1'] - X['casos_dengue_lag_2']
        if 'casos_dengue_lag_1' in X.columns and 'casos_dengue_lag_4' in X.columns:
            d['dengue_trend_4w'] = X['casos_dengue_lag_1'] - X['casos_dengue_lag_4']
        if 'casos_dengue_lag_1' in X.columns and 'casos_dengue_lag_8' in X.columns:
            d['dengue_trend_8w'] = X['casos_dengue_lag_1'] - X['casos_dengue_lag_8']
        if 'casos_dengue_lag_1' in X.columns and 'casos_dengue_lag_12' in X.columns:
            d['dengue_trend_12w'] = X['casos_dengue_lag_1'] - X['casos_dengue_lag_12']
            d['dengue_ratio_1_12'] = X['casos_dengue_lag_1'] / (X['casos_dengue_lag_12'] + 1)
        cols_1_4 = [f'casos_dengue_lag_{{i}}' for i in range(1, 5) if f'casos_dengue_lag_{{i}}' in X.columns]
        cols_9_12 = [f'casos_dengue_lag_{{i}}' for i in range(9, 13) if f'casos_dengue_lag_{{i}}' in X.columns]
        if len(cols_1_4) >= 2 and len(cols_9_12) >= 2:
            d['dengue_ma_1_4'] = X[cols_1_4].mean(axis=1)
            d['dengue_ma_9_12'] = X[cols_9_12].mean(axis=1)
            d['dengue_ma_ratio'] = d['dengue_ma_1_4'] / (d['dengue_ma_9_12'] + 1)
        return d

    def _rolling(self, X):
        d = pd.DataFrame(index=X.index)
        if 'semana_epi' in X.columns:
            d['week_sin'] = np.sin(2*np.pi*X['semana_epi']/52)
            d['week_cos'] = np.cos(2*np.pi*X['semana_epi']/52)
            d['week_sin2'] = np.sin(4*np.pi*X['semana_epi']/52)
            d['week_cos2'] = np.cos(4*np.pi*X['semana_epi']/52)
        return d

    def _polys(self, X):
        d = pd.DataFrame(index=X.index)
        for v in ['casos_dengue_lag_1','casos_dengue_lag_2','casos_dengue_lag_4','casos_dengue_lag_8']:
            if v in X.columns:
                d[v+'_squared'] = X[v]**2
        return d

    def _augment(self, X):
        Xe = self._filter_epi(X)
        num = [c for c in Xe.columns
               if pd.api.types.is_numeric_dtype(Xe[c]) and c.lower() not in
               ['fecha','date','datetime','timestamp']]
        Xn = Xe[num]
        parts = [Xn]
        for fn in [self._lag_aggs, self._trends, self._rolling, self._polys]:
            f = fn(Xn)
            if not f.empty: parts.append(f)
        Xa = pd.concat(parts, axis=1)
        Xa = Xa.replace([np.inf, -np.inf], np.nan).fillna(Xa.mean())
        return Xa

    def predict(self, X):
        Xa = self._augment(X)
        for c in self.X_augmented_columns:
            if c not in Xa.columns: Xa[c] = 0
        Xa = Xa[self.X_augmented_columns]
        Xs = pd.DataFrame(self.scaler.transform(Xa),
                          columns=Xa.columns, index=Xa.index)
        return self.lgbm_model.predict(Xs[self.selected_features])

if __name__ == "__main__":
    model_dir = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\4_LightGBM\2_datos\1_raw\0_70_30\1_resultados_lightgbm_70_30_epi\modelo_guardado_lightgbm_epi"
    model = DengueEpiDeploymentLGBM(model_dir)
    test_path = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\4_LightGBM\2_datos\1_raw\0_70_30\2_meteo_epi_2021-2026_1_rezagos_meteo_epi_test_70_30.xlsx"
    df = pd.read_excel(test_path)
    X_test = df.drop('casos_dengue', axis=1)
    preds = model.predict(X_test)
    print(f"Predicciones: {{len(preds)}}")
    print(f"  Media: {{np.mean(preds):.2f}} | Mediana: {{np.median(preds):.2f}}")
    out = os.path.join(model_dir, "predicciones_ejemplo.xlsx")
    res = X_test.copy(); res['prediccion_casos_dengue'] = preds
    res.to_excel(out, index=False)
    print(f"✅ Guardado en: {{out}}")
