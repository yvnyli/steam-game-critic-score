import pandas as pd
import xgboost as xgb
import pickle

df_X_z = pd.read_parquet('/app/data/df_X_z_v1.parquet')
df_y_z = pd.read_parquet('/app/data/df_y_z_v1.parquet')
X = df_X_z.loc[df_y_z["score"].notna(),:].to_numpy(dtype="float32")
y = df_y_z.loc[df_y_z["score"].notna(),"score"].to_numpy(dtype="float32")
best_XGBR = xgb.XGBRegressor(objective="reg:squarederror",eval_metric="rmse",tree_method="hist",
    random_state=0,n_jobs=-1,colsample_bytree=0.7,learning_rate=0.03,max_depth=7,min_child_weight=40,
    n_estimators=200,reg_alpha=0.1,reg_lambda=0.1,subsample=0.7)
best_XGBR.fit(X, y)
data = {"model": best_XGBR}
with open("/app/models/final_model2.pkl", "wb") as f:
    pickle.dump(data,f)