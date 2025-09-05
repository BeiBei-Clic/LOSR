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
        niterations=100,  # 减少迭代次数，通过并行补偿
        
        # 复杂度控制
        maxsize=15,  # 适当减少最大复杂度以加速
        
        # 优化器配置
        should_optimize_constants=True,
        enable_linear_optimization=True,  # 启用多重线性优化
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
        random_state=42
    )
    
    print("开始符号回归搜索...")
    model.fit(X_train, y_train)
    
    # 在训练集和测试集上进行预测
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # 计算性能指标
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"\n=== {dataset_name} 性能评估 ===")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")
    print(f"训练集 R²: {train_r2:.6f}")
    print(f"测试集 R²: {test_r2:.6f}")
    
    # 获取最佳方程
    best_equation = str(model.sympy())
    print(f"\n学到的最佳方程: {best_equation}")
    
    # 分析方程复杂度
    equations_info = []
    if hasattr(model, 'equations_') and model.equations_ is not None:
        best_eq = model.equations_.iloc[-1]
        print(f"最佳方程复杂度: {best_eq['complexity']}")
        print(f"最佳方程损失: {best_eq['loss']:.6f}")
        
        # 收集所有方程信息（使用简化后的SymPy表达式）
        import sympy as sp
        
        for i, row in model.equations_.iterrows():
            # 获取原始表达式并转换为SymPy可解析的格式
            raw_expr = str(row['equation'])
            
            # 替换PySR表达式中的函数名为SymPy格式
            expr_str = raw_expr.replace('inv(', '1/(')
            
            # 定义符号变量
            x0, x1, x2, x3, x4 = sp.symbols('x0 x1 x2 x3 x4')
            
            # 使用sympy解析并简化表达式
            sympy_expr = sp.sympify(expr_str)
            simplified_expr = str(sp.simplify(sympy_expr))
            
            equations_info.append({
                'complexity': int(row['complexity']),
                'loss': float(row['loss']),
                'equation': simplified_expr
            })
        
        # 显示复杂度分析
        complexities = model.equations_['complexity'].values
        losses = model.equations_['loss'].values
        
        print(f"\n=== 复杂度分析 ===")
        print(f"平均复杂度: {np.mean(complexities):.2f}")
        print(f"最小复杂度: {np.min(complexities)}")
        print(f"最大复杂度: {np.max(complexities)}")
        print(f"复杂度标准差: {np.std(complexities):.2f}")
        
        print(f"\n=== 帕累托前沿（复杂度 vs 损失）===")
        for i, (comp, loss) in enumerate(zip(complexities[-10:], losses[-10:])):  # 显示最后10个
            equation = model.equations_.iloc[-(10-i)]['equation']
            print(f"复杂度 {comp:2d}: 损失 {loss:.6f} - {equation}")
    
    # 保存结果
    results = {
        'dataset_name': dataset_name,
        'dataset_shape': data.shape,
        'feature_columns': list(data.columns[:-1]),
        'target_column': data.columns[-1],
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'best_equation': best_equation,
        'performance': {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2)
        },
        'equations': equations_info
    }
    
    # 保存到JSON文件
    result_file = f'result/{dataset_name}_results.json'
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