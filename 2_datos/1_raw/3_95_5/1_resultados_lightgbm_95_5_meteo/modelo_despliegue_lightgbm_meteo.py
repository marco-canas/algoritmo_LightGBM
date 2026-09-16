"""
Script de despliegue del modelo de predicción de casos de dengue
usando SOLO predictores meteorológicos con LightGBM.
Generado automáticamente - 2026-09-16 17:13:08
"""
import pandas as pd
import numpy as np
import pickle
import json
import os

class DengueMeteoDeploymentLGBM:
    METEOROLOGICAL_PREDICTORS = ['fecha', 'semana_epi', 'temp', 'temp_max', 'temp_min', 'hum_esp', 'hum_rel', 'prec', 'dias_lluvia', 'temp_lag_1', 'temp_lag_2', 'temp_lag_3', 'temp_lag_4', 'temp_lag_5', 'temp_lag_6', 'temp_lag_7', 'temp_lag_8', 'temp_lag_9', 'temp_lag_10', 'temp_lag_11', 'temp_lag_12', 'temp_max_lag_1', 'temp_max_lag_2', 'temp_max_lag_3', 'temp_max_lag_4', 'temp_max_lag_5', 'temp_max_lag_6', 'temp_max_lag_7', 'temp_max_lag_8', 'temp_max_lag_9', 'temp_max_lag_10', 'temp_max_lag_11', 'temp_max_lag_12', 'temp_min_lag_1', 'temp_min_lag_2', 'temp_min_lag_3', 'temp_min_lag_4', 'temp_min_lag_5', 'temp_min_lag_6', 'temp_min_lag_7', 'temp_min_lag_8', 'temp_min_lag_9', 'temp_min_lag_10', 'temp_min_lag_11', 'temp_min_lag_12', 'hum_esp_lag_1', 'hum_esp_lag_2', 'hum_esp_lag_3', 'hum_esp_lag_4', 'hum_esp_lag_5', 'hum_esp_lag_6', 'hum_esp_lag_7', 'hum_esp_lag_8', 'hum_esp_lag_9', 'hum_esp_lag_10', 'hum_esp_lag_11', 'hum_esp_lag_12', 'hum_rel_lag_1', 'hum_rel_lag_2', 'hum_rel_lag_3', 'hum_rel_lag_4', 'hum_rel_lag_5', 'hum_rel_lag_6', 'hum_rel_lag_7', 'hum_rel_lag_8', 'hum_rel_lag_9', 'hum_rel_lag_10', 'hum_rel_lag_11', 'hum_rel_lag_12', 'prec_lag_1', 'prec_lag_2', 'prec_lag_3', 'prec_lag_4', 'prec_lag_5', 'prec_lag_6', 'prec_lag_7', 'prec_lag_8', 'prec_lag_9', 'prec_lag_10', 'prec_lag_11', 'prec_lag_12', 'dias_lluvia_lag_1', 'dias_lluvia_lag_2', 'dias_lluvia_lag_3', 'dias_lluvia_lag_4', 'dias_lluvia_lag_5', 'dias_lluvia_lag_6', 'dias_lluvia_lag_7', 'dias_lluvia_lag_8', 'dias_lluvia_lag_9', 'dias_lluvia_lag_10', 'dias_lluvia_lag_11', 'dias_lluvia_lag_12', 'soi_lag_8', 'soi_lag_9', 'soi_lag_10', 'soi_lag_11', 'soi_lag_12', 'sst_lag_8', 'sst_lag_9', 'sst_lag_10', 'sst_lag_11', 'sst_lag_12']

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
        print("✅ Modelo LightGBM (meteorológico) cargado")

    def _filter_meteo(self, X):
        allowed = [c for c in X.columns if c in self.METEOROLOGICAL_PREDICTORS]
        return X[allowed].copy()

    def _interactions(self, X):
        d = pd.DataFrame(index=X.index)
        for a, b, n in [('temp','hum_rel','temp_hum_rel'),('temp_max','hum_rel','temp_max_hum_rel'),
                        ('prec','temp','prec_temp'),('dias_lluvia','hum_rel','dias_lluvia_hum_rel'),
                        ('soi_lag_12','sst_lag_12','soi_sst_lag12'),('temp','soi_lag_12','temp_soi_lag12'),
                        ('prec','hum_rel','prec_hum_rel')]:
            if a in X.columns and b in X.columns:
                d[n] = X[a]*X[b]
        return d

    def _polys(self, X):
        d = pd.DataFrame(index=X.index)
        for v in ['temp','hum_rel','prec','temp_max','temp_min']:
            if v in X.columns:
                d[v+'_squared'] = X[v]**2
        return d

    def _lag_aggs(self, X):
        d = pd.DataFrame(index=X.index)
        for b in ['temp','temp_max','temp_min','hum_esp','hum_rel','prec','dias_lluvia']:
            cols = [c for c in X.columns if c.startswith(b+'_lag_')]
            if len(cols) >= 3:
                l = X[cols]
                d[b+'_lag_mean'] = l.mean(axis=1); d[b+'_lag_std'] = l.std(axis=1)
                d[b+'_lag_max'] = l.max(axis=1); d[b+'_lag_min'] = l.min(axis=1)
        return d

    def _rolling(self, X):
        d = pd.DataFrame(index=X.index)
        if 'semana_epi' in X.columns:
            d['week_sin'] = np.sin(2*np.pi*X['semana_epi']/52)
            d['week_cos'] = np.cos(2*np.pi*X['semana_epi']/52)
        if 'temp' in X.columns and 'temp_lag_4' in X.columns:
            d['temp_trend_4w'] = X['temp'] - X['temp_lag_4']
        if 'temp' in X.columns and 'temp_lag_12' in X.columns:
            d['temp_trend_12w'] = X['temp'] - X['temp_lag_12']
        if 'prec' in X.columns and 'prec_lag_4' in X.columns:
            d['prec_trend_4w'] = X['prec'] - X['prec_lag_4']
        if 'hum_rel' in X.columns and 'hum_rel_lag_4' in X.columns:
            d['hum_trend_4w'] = X['hum_rel'] - X['hum_rel_lag_4']
        return d

    def _augment(self, X):
        Xm = self._filter_meteo(X)
        num = [c for c in Xm.columns
               if pd.api.types.is_numeric_dtype(Xm[c]) and c.lower() not in
               ['fecha','date','datetime','timestamp']]
        Xn = Xm[num]
        parts = [Xn]
        for fn in [self._interactions, self._polys, self._lag_aggs, self._rolling]:
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
    model_dir = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\4_LightGBM\2_datos\1_raw\3_95_5\1_resultados_lightgbm_95_5_meteo\modelo_guardado_lightgbm_meteo"
    model = DengueMeteoDeploymentLGBM(model_dir)
    test_path = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\4_LightGBM\2_datos\1_raw\3_95_5\2_meteo_epi_2021-2026_1_rezagos_meteo_epi_test_95_5.xlsx"
    df = pd.read_excel(test_path)
    X_test = df.drop('casos_dengue', axis=1)
    preds = model.predict(X_test)
    print(f"Predicciones: {len(preds)}")
    print(f"  Media: {np.mean(preds):.2f} | Mediana: {np.median(preds):.2f}")
    out = os.path.join(model_dir, "predicciones_ejemplo.xlsx")
    res = X_test.copy(); res['prediccion_casos_dengue'] = preds
    res.to_excel(out, index=False)
    print(f"✅ Guardado en: {out}")
