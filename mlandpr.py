import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# ==========1. pandas读取iris数据集（不需要本地iris.csv文件！）==========
from sklearn.datasets import load_iris
iris_raw = load_iris()
import pandas as pd
# 转为DataFrame，模拟本地csv读取效果
df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
df['target'] = iris_raw.target
print("====pandas读取iris数据集====")
print(df.head())
X_local = df.iloc[:, :-1].values
y_local = df.iloc[:, -1].values

# ==========2. sklearn直接加载iris数据集==========
iris = load_iris()
X = iris.data
y = iris.target
print("\n====sklearn内置iris数据集特征形状====")
print(X.shape, y.shape)

# ==========3. 五折交叉验证==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_list = []
pre_list = []
rec_list = []
f1_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    pre = precision_score(y_val, y_pred, average='macro')
    rec = recall_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')

    acc_list.append(acc)
    pre_list.append(pre)
    rec_list.append(rec)
    f1_list.append(f1)
    print(f"\n第{fold+1}折：")
    print(f"准确度:{acc:.4f}, 精确率:{pre:.4f},召回率:{rec:.4f},F1:{f1:.4f}")

# 五折平均
print("\n====五折交叉验证平均指标====")
print(f"平均准确度:{np.mean(acc_list):.4f}")
print(f"平均精确率:{np.mean(pre_list):.4f}")
print(f"平均召回率:{np.mean(rec_list):.4f}")
print(f"平均F1值:{np.mean(f1_list):.4f}")
