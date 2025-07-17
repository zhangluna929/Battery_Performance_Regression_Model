import logging
from sklearn.ensemble import RandomForestRegressor

# 随机森林回归
def build_random_forest_model(X_train, y_train, n_estimators=100, random_state=42):
    logging.info("Building Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    logging.info("Random Forest model trained successfully.")
    return rf_model 