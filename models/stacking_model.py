from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import logging

def build_stacking_model(X_train, y_train, base_models, model_names):
    """
    构建并训练一个 Stacking 集成模型。

    Stacking 通过一个元模型来学习如何最好地组合基模型的预测。

    参数:
    - X_train: 训练集特征.
    - y_train: 训练集目标.
    - base_models: 一个包含基模型的列表.
    - model_names: 基模型的名称列表.

    返回:
    - stacking_model: 已训练的 Stacking 模型.
    """
    logging.info("正在构建 Stacking 集成模型...")

    estimators = list(zip(model_names, base_models))
    
    # 我们使用一个简单的线性回归作为元模型
    meta_model = LinearRegression()
    
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,  # 使用交叉验证来训练元模型
        n_jobs=-1
    )
    
    stacking_model.fit(X_train, y_train)
    
    logging.info("Stacking 集成模型训练完成。")
    return stacking_model 