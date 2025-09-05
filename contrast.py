import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

from pysr import PySRRegressor
import json

# 创建结果文件夹
os.makedirs('result', exist_ok=True)

def run_multiple_experiments(X_train, y_train, X_test, y_test, enable_linear_optimization, n_runs=20):
    """运行多次实验并返回MSE中位数的结果"""
    results_list = []
    mse_list = []
    
    for i in range(n_runs):
        seed = i + 1  # 随机种子从1到20
        
        # 创建符号回归模型（针对32核CPU优化）
        model = PySRRegressor(
            model_selection="best",
        
            binary_operators=["+", "*", "-", "/"],
            unary_operators=[
                "cos",
                "sin", 
                "exp",
                "log",
                "inv(x) = 1/x",
                "sqrt",
            ],
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            
            # 并行优化参数 - 充分利用32核
            procs=30,  # 使用30个进程，留2个核给系统
            multithreading=True,
            
            # 种群优化 - 增加并行度
            populations=100,  # 增加种群数量以充分利用多核
            population_size=200,  # 适当减少单个种群大小以平衡内存
            
            # 迭代优化
            niterations=1000,  # 减少迭代次数，通过并行补偿
            
            # 复杂度控制
            maxsize=15,  # 适当减少最大复杂度以加速
            
            # 优化器配置
            should_optimize_constants=True,
            enable_linear_optimization=enable_linear_optimization,  # 控制是否启用线性优化
            optimizer_algorithm="BFGS",
            optimizer_nrestarts=3,  # 减少重启次数以加速
            
            # 选择和变异参数优化
            weight_simplify=0.05,
            weight_optimize=0.1,  # 减少优化权重以加速
            parsimony=0.01,
            
            # 选择压力优化
            tournament_selection_n=8,  # 减少锦标赛大小以加速
            tournament_selection_p=0.9,
            
            # 批处理优化
            batch_size=None,  # 启用批处理以提高效率
            
            # 早停机制
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity < 10",
            
            verbosity=1,
            random_state=seed
        )
        
        print(f"运行第 {seed} 次实验，线性优化: {enable_linear_optimization}")
        model.fit(X_train, y_train)
        
        # 在测试集上进行预测
        test_predictions = model.predict(X_test)
        
        # 计算测试集MSE
        test_mse = mean_squared_error(y_test, test_predictions)
        
        # 保存结果
        result = {
            'seed': seed,
            'model': model,
            'test_mse': test_mse,
            'test_predictions': test_predictions,
            'best_equation': str(model.sympy())
        }
        
        results_list.append(result)
        mse_list.append(test_mse)
    
    # 找到MSE中位数的索引
    median_mse_index = np.argsort(mse_list)[len(mse_list) // 2]
    return results_list[median_mse_index]

def process_dataset(dataset_path, dataset_name):
    """处理单个数据集的符号回归"""
    print(f"\n{'='*60}")
    print(f"正在处理数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 读取数据
    data = pd.read_csv(dataset_path, sep='\t')
    print(f"数据集形状: {data.shape}")
    print(f"特征列: {list(data.columns[:-1])}")
    print(f"标签列: {data.columns[-1]}")
    
    # 准备特征和标签
    X = data.iloc[:, :-1].values  # 所有特征列
    y = data.iloc[:, -1].values   # 最后一列作为标签
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签向量形状: {y.shape}")
    
    # 随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 运行开启线性优化的实验
    print("\n开始运行开启线性优化的实验...")
    linear_result = run_multiple_experiments(X_train, y_train, X_test, y_test, enable_linear_optimization=True)
    
    # 运行关闭线性优化的实验
    print("\n开始运行关闭线性优化的实验...")
    no_linear_result = run_multiple_experiments(X_train, y_train, X_test, y_test, enable_linear_optimization=False)
    
    # 在训练集和测试集上进行预测
    linear_train_predictions = linear_result['model'].predict(X_train)
    linear_test_predictions = linear_result['model'].predict(X_test)
    no_linear_train_predictions = no_linear_result['model'].predict(X_train)
    no_linear_test_predictions = no_linear_result['model'].predict(X_test)
    
    # 计算性能指标
    linear_train_mse = mean_squared_error(y_train, linear_train_predictions)
    linear_test_mse = linear_result['test_mse']
    linear_train_r2 = r2_score(y_train, linear_train_predictions)
    linear_test_r2 = r2_score(y_test, linear_test_predictions)
    
    no_linear_train_mse = mean_squared_error(y_train, no_linear_train_predictions)
    no_linear_test_mse = no_linear_result['test_mse']
    no_linear_train_r2 = r2_score(y_train, no_linear_train_predictions)
    no_linear_test_r2 = r2_score(y_test, no_linear_test_predictions)
    
    print(f"\n=== {dataset_name} 性能评估 ===")
    print(f"开启线性优化 - 训练集 MSE: {linear_train_mse:.6f}")
    print(f"开启线性优化 - 测试集 MSE: {linear_test_mse:.6f}")
    print(f"开启线性优化 - 训练集 R²: {linear_train_r2:.6f}")
    print(f"开启线性优化 - 测试集 R²: {linear_test_r2:.6f}")
    print(f"关闭线性优化 - 训练集 MSE: {no_linear_train_mse:.6f}")
    print(f"关闭线性优化 - 测试集 MSE: {no_linear_test_mse:.6f}")
    print(f"关闭线性优化 - 训练集 R²: {no_linear_train_r2:.6f}")
    print(f"关闭线性优化 - 测试集 R²: {no_linear_test_r2:.6f}")
    
    # 获取最佳方程
    linear_best_equation = linear_result['best_equation']
    no_linear_best_equation = no_linear_result['best_equation']
    print(f"\n开启线性优化时学到的最佳方程: {linear_best_equation}")
    print(f"关闭线性优化时学到的最佳方程: {no_linear_best_equation}")
    
    # 保存结果
    results = {
        'dataset_name': dataset_name,
        'dataset_shape': data.shape,
        'feature_columns': list(data.columns[:-1]),
        'target_column': data.columns[-1],
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'linear_optimization': {
            'best_equation': linear_best_equation,
            'performance': {
                'train_mse': float(linear_train_mse),
                'test_mse': float(linear_test_mse),
                'train_r2': float(linear_train_r2),
                'test_r2': float(linear_test_r2)
            },
            'seed': linear_result['seed']
        },
        'no_linear_optimization': {
            'best_equation': no_linear_best_equation,
            'performance': {
                'train_mse': float(no_linear_train_mse),
                'test_mse': float(no_linear_test_mse),
                'train_r2': float(no_linear_train_r2),
                'test_r2': float(no_linear_test_r2)
            },
            'seed': no_linear_result['seed']
        }
    }
    
    # 保存到JSON文件
    result_file = f'result/{dataset_name}_results_new.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
            
    print(f"\n结果已保存到:")
    print(f"- {result_file}")
    
    return results

def main():
    """主函数，处理所有数据集"""
    datasets = [
        # ('data/dataset1_5features.txt', 'dataset1'),
        # ('data/dataset2_5features.txt', 'dataset2'), 
        ('data/dataset3_5features.txt', 'dataset3')
    ]
    
    all_results = {}
    
    for dataset_path, dataset_name in datasets:
        results = process_dataset(dataset_path, dataset_name)
        all_results[dataset_name] = results

    
if __name__ == "__main__":
    main()