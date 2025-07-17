import logging
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score

def tune_model(model, X_train, y_train, param_space, n_iter=32, cv_folds=3):
    """
    使用贝叶斯优化来调整模型的超参数。

    贝叶斯优化能够更高效地在参数空间中找到最优解。

    参数:
    - model: 需要调优的模型.
    - X_train: 训练集特征.
    - y_train: 训练集目标.
    - param_space: 超参数的搜索空间 (一个字典).
    - n_iter: 贝叶斯优化的迭代次数.
    - cv_folds: 交叉验证的折数.

    返回:
    - best_model: 调整后的最佳模型.
    """
    logging.info(f"正在使用贝叶斯优化调整模型: {type(model).__name__}...")
    
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    
    bayes_search.fit(X_train, y_train)
    
    best_model = bayes_search.best_estimator_
    logging.info(f"找到的最佳参数: {bayes_search.best_params_}")
    logging.info(f"最佳交叉验证分数 (Negative MSE): {bayes_search.best_score_:.4f}")
    
    return best_model

# 交叉验证
def cross_validation(model, X_train, y_train, cv_folds=5):
    logging.info("Performing cross-validation...")
    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
    logging.info(f"Cross-validation scores: {scores}")
    return scores 