import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_battery_data(num_samples=1000, num_cycles=100):
    """
    生成更真实的合成电池数据集。
    
    参数:
    - num_samples: 电池样本数量。
    - num_cycles: 每个电池的循环次数。
    
    返回:
    - 一个 pandas DataFrame，包含生成的电池数据。
    """
    data = []
    
    for i in range(num_samples):
        # 初始容量和电阻 (每个电池不同)
        initial_capacity = np.random.uniform(2.0, 3.0)  # Ah
        initial_resistance = np.random.uniform(0.01, 0.05)  # Ohm
        
        # 衰减率 (每个电池不同)
        capacity_fade_rate = np.random.uniform(0.0005, 0.002)
        resistance_increase_rate = np.random.uniform(0.0001, 0.0005)
        
        # 外部条件 (每个电池、每个循环都可能变化)
        temperature = np.random.uniform(15, 40)  # 摄氏度
        discharge_rate = np.random.uniform(0.5, 2.0)  # C-rate
        
        for cycle in range(1, num_cycles + 1):
            # 模拟容量衰减
            capacity = initial_capacity * (1 - capacity_fade_rate * cycle) * (1 - 0.001 * (temperature - 25))
            
            # 模拟内阻增加
            resistance = initial_resistance * (1 + resistance_increase_rate * cycle) * (1 + 0.005 * (temperature - 25))
            
            # 模拟电化学阻抗谱(EIS)特征 (简化模型)
            # R0: 欧姆电阻, Rct: 电荷转移电阻
            R0 = resistance + np.random.normal(0.001, 0.0005)
            Rct = np.random.uniform(0.02, 0.1) * (1 + 0.01 * cycle) * (1 + 0.01 * discharge_rate)
            
            # 计算电池循环寿命 (SOH - State of Health)
            soh = (capacity / initial_capacity) * 100
            
            # 目标：剩余使用寿命 (RUL)
            # 假设当容量衰减到80%时寿命结束
            end_of_life_cycle = (1 - 0.8) / capacity_fade_rate
            rul = end_of_life_cycle - cycle
            
            data.append({
                'battery_id': i,
                'cycle': cycle,
                'initial_capacity_ah': initial_capacity,
                'initial_resistance_ohm': initial_resistance,
                'temperature_c': temperature,
                'discharge_rate_c': discharge_rate,
                'capacity_ah': capacity,
                'resistance_ohm': resistance,
                'R0_ohm': R0,
                'Rct_ohm': Rct,
                'soh_percent': soh,
                'rul': rul
            })
            
    df = pd.DataFrame(data)
    
    # 仅保留 RUL > 0 的数据
    df = df[df['rul'] > 0].reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("正在生成合成电池数据...")
    synthetic_data = generate_battery_data(num_samples=100, num_cycles=200)
    
    # 保存到 CSV 文件
    output_filename = 'battery_performance_data.csv'
    synthetic_data.to_csv(output_filename, index=False)
    
    print(f"数据生成完毕，并已保存到 '{output_filename}'")
    print("数据预览:")
    print(synthetic_data.head())
    print(f"\n总共生成 {len(synthetic_data)} 条数据。")
