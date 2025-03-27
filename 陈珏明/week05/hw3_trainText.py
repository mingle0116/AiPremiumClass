import fasttext
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 数据预处理
def prepare_data(file_path, test_size=0.2):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(lines, test_size=test_size, random_state=30)
    
    # 保存fastText格式数据
    with open(r'week05\cooking.train', 'w') as f:
        f.writelines(train_data)
    
    with open(r'week05\cooking.test', 'w') as f:
        f.writelines(test_data)

# 2. 模型训练
def train_model():
    model = fasttext.train_supervised(
        input=r'week05\cooking.train',
        lr=0.5,            # 学习率
        epoch=50,          # 训练轮次
        wordNgrams=2,      # 使用2-gram特征
        loss='ova',        # 一对多损失函数（适用于多标签分类）
    )
    return model

# 3. 模型评估
def evaluate_model(model):
    results = model.test(
        r'week05\cooking.test'
    )
    print(f'准确率: {results[1]*100:.2f}%')
    print(f'召回率: {results[2]*100:.2f}%')

# 4. 预测示例
def predict_example(model):
    examples = [
        "How to make perfect chocolate cake?",
        "Is it safe to eat raw chicken?",
        "Best way to store fresh herbs"
    ]
    
    for text in examples:
        labels, probs = model.predict(text, k=3)  # 返回top3预测结果
        print(f"输入: {text}")
        for label, prob in zip(labels, probs):
            print(f"→ {label.replace('__label__', '')}: {prob:.2%}")
        print()

if __name__ == '__main__':
    # 数据预处理（首次运行需要执行）
    prepare_data(r'week05\note\cooking.stackexchange.txt')
    
    # 训练模型
    model = train_model()
    
    # 保存模型
    model.save_model(r'week05\cooking_model.bin')
    
    # 评估模型
    evaluate_model(model)
    
    # 示例预测
    predict_example(model)