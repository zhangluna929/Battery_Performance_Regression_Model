import logging
import xgboost as xgb

# XGBoost回归
def build_xgboost_model(X_train, y_train, n_estimators=100, learning_rate=0.01, max_depth=6):
    logging.info("Building XGBoost model...")
    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    xgb_model.fit(X_train, y_train)
    logging.info("XGBoost model trained successfully.")
    return xgb_model 