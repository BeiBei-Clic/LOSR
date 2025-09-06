import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

from pysr import PySRRegressor
import json
from datetime import datetime

# 创建结果目录
os.makedirs('result', exist_ok=True)

def create_model_params(seed=1):
    """创建基础模型参数"""
    return {
        "model_selection": "best",
        "binary_operators": ["+", "*", "-", "/"],
        "unary_operators": [
            "cos",
            "sin", 
            "exp",
            "log",
            "inv(x) = 1/x",
            "sqrt",
        ],
        "extra_sympy_mappings": {"inv": lambda x: 1 / x},
        "procs": 30,
        "multithreading": True,
        "populations": 90,
        "population_size": 200,
        "niterations": 500,
        "maxsize": 15,
        "should_optimize_constants": True,
        "optimizer_algorithm": "BFGS",
        "optimizer_nrestarts": 3,
        "weight_simplify": 0.05,
        "weight_optimize": 0.1,
        "parsimony": 0.01,
        "tournament_selection_n": 8,
        "tournament_selection_p": 0.9,
        "batch_size": None,
        "early_stop_condition": "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10",
        "verbosity": 1,
        "random_state": seed
    }

def run_single_experiment(X_train, y_train, X_test, y_test, linear_optimization_method="none", seed=1):
    """运行单次实验"""
    model_params = create_model_params(seed)
    
    # 根据线性优化方法设置参数
    if linear_optimization_method == "none":
        model_params["linear_optimization_method"] = None
    elif linear_optimization_method == "correlation":
        model_params["linear_optimization_method"] = "correlation"
    elif linear_optimization_method == "stepwise":
        model_params["linear_optimization_method"] = "stepwise"
    else:
        raise ValueError(f"不支持的线性优化方法: {linear_optimization_method}")
    
    # 创建符号回归模型
    model = PySRRegressor(**model_params)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 在测试集上进行预测
    test_predictions = model.predict(X_test)
    
    # 计算测试集MSE
    test_mse = mean_squared_error(y_test, test_predictions)
    
    # 获取最佳方程
    best_equation = str(model.sympy())
    
    return {
        'seed': seed,
        'model': model,
        'test_mse': test_mse,
        'test_predictions': test_predictions,
        'best_equation': best_equation,
        'linear_optimization_method': linear_optimization_method
    }

def run_multiple_experiments(X_train, y_train, X_test, y_test, linear_optimization_method="none", n_runs=20):
    """运行多次实验并返回MSE中位数的结果"""
    results_list = []
    mse_list = []
    
    for i in range(n_runs):
        seed = i + 1
        result = run_single_experiment(X_train, y_train, X_test, y_test, 
                                     linear_optimization_method, seed)
        results_list.append(result)
        mse_list.append(result['test_mse'])
    
    # 找到MSE中位数的索引
    median_mse_index = np.argsort(mse_list)[len(mse_list) // 2]
    return results_list[median_mse_index]

def calculate_performance_metrics(model, X_train, y_train, X_test, y_test):
    """计算性能指标"""
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    return {
        'train_mse': mean_squared_error(y_train, train_predictions),
        'test_mse': mean_squared_error(y_test, test_predictions),
        'train_r2': r2_score(y_train, train_predictions),
        'test_r2': r2_score(y_test, test_predictions)
    }

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
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签向量形状: {y.shape}")
    
    # 随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 运行不同配置的实验 - 使用一个超参数控制线性优化
    linear_optimization_methods = [
        ("关闭线性优化", "none"),
        # ("传统相关性线性优化", "correlation"),
        ("双向逐步回归线性优化", "stepwise")
    ]
    
    experiment_results = {}
    
    for exp_name, method in linear_optimization_methods:
        print(f"\n开始运行{exp_name}的实验...")
        result = run_multiple_experiments(X_train, y_train, X_test, y_test, method)
        
        # 计算性能指标
        performance = calculate_performance_metrics(result['model'], X_train, y_train, X_test, y_test)
        
        experiment_results[method] = {
            'result': result,
            'performance': performance
        }
    
    # 打印结果
    print(f"\n=== {dataset_name} 性能评估 ===")
    for method, exp_data in experiment_results.items():
        perf = exp_data['performance']
        equation = exp_data['result']['best_equation']
        method_name = {
            'correlation': '传统相关性线性优化',
            'stepwise': '双向逐步回归线性优化',
            'none': '关闭线性优化'
        }[method]
        
        print(f"\n{method_name}:")
        print(f"  训练集 MSE: {perf['train_mse']:.6f}")
        print(f"  测试集 MSE: {perf['test_mse']:.6f}")
        print(f"  训练集 R²: {perf['train_r2']:.6f}")
        print(f"  测试集 R²: {perf['test_r2']:.6f}")
        print(f"  最佳方程: {equation}")
    
    # 保存结果到指定目录
    results = {
        'dataset_name': dataset_name,
        'dataset_shape': data.shape,
        'feature_columns': list(data.columns[:-1]),
        'target_column': data.columns[-1],
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'experiments': {}
    }
    
    # 保存每个实验的结果和超参数
    for method, exp_data in experiment_results.items():
        # 创建可序列化的超参数副本（排除lambda函数）
        serializable_params = create_model_params(exp_data['result']['seed']).copy()
        serializable_params.pop('extra_sympy_mappings', None)  # 移除不可序列化的字段
        
        results['experiments'][method] = {
            'linear_optimization_method': method,
            'best_equation': exp_data['result']['best_equation'],
            'performance': {k: float(v) for k, v in exp_data['performance'].items()},
            'seed': exp_data['result']['seed'],
            'hyperparameters': serializable_params
        }
      
    # 生成带时间戳的文件名，保存到指定目录
    now = datetime.now()
    timestamp = now.strftime("%m%d%H%M")
    # 修改为相对路径保存
    result_file = f'result/{dataset_name}_{timestamp}.json'
    
    # 确保目录存在
    os.makedirs('result', exist_ok=True)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {result_file}")
    return results

def main():
    """主函数"""
    # 确保结果目录存在
    os.makedirs('/home/xyh/LOSR/result', exist_ok=True)
    
    # 动态读取data目录下的所有txt文件
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 不存在")
        return
    
    datasets = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            dataset_path = os.path.join(data_dir, filename)
            dataset_name = os.path.splitext(filename)[0]
            datasets.append((dataset_path, dataset_name))
    
    if not datasets:
        print(f"警告: 在 '{data_dir}' 目录中未找到任何 .txt 文件")
        return
    
    print(f"找到 {len(datasets)} 个数据集文件")
    
    all_results = []
    for dataset_path, dataset_name in datasets:
        if os.path.exists(dataset_path):
            print(f"\n处理数据集: {dataset_name}")
            result = process_dataset(dataset_path, dataset_name)
            all_results.append(result)
        else:
            print(f"警告: 数据集文件不存在: {dataset_path}")
    
    print(f"\n{'='*60}")
    print("所有数据集处理完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()