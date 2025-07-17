from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import logging

def build_gpr_model(X_train, y_train):
    """
    构建并训练高斯过程回归 (GPR) 模型。

    GPR 能够提供预测的不确定性，这在许多科学和工程应用中至关重要。
    我们使用 RBF 内核来捕捉非线性关系，并加入 WhiteKernel 来解释噪声。

    参数:
    - X_train: 训练集特征.
    - y_train: 训练集目标.

    返回:
    - gpr_model: 已训练的 GPR 模型.
    """
    logging.info("正在构建高斯过程回归模型...")
    
    # 定义 GPR 内核
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
             + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
             
    gpr_model = GaussianProcessRegressor(kernel=kernel, 
                                       alpha=0.0,
                                       n_restarts_optimizer=10, 
                                       random_state=42)
    
    gpr_model.fit(X_train, y_train)
    
    logging.info("高斯过程回归模型训练完成。")
    return gpr_model 