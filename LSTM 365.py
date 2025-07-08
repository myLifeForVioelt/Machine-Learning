import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import layers, models
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import seaborn as sns
# 设置中文显示
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# ------------------------
# 1. 数据加载与预处理
# ------------------------
def load_and_preprocess(path):
    df = pd.read_csv(path, na_values='?')
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.dropna(subset=['DateTime'])

    for col in df.columns:
        if col != 'DateTime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df['hour'] = df['DateTime'].dt.floor('H')
    df['date'] = df['DateTime'].dt.date
    df['RR'] = df['RR'] / 10.0
    df['sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - (
            df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
    )
    daily_df = df.groupby('date').agg({
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'sub_metering_remainder':'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }).reset_index()

    return daily_df

def load_and_preprocess_test(path):
    column_names = [
        'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]
    df = pd.read_csv(path, names=column_names, na_values='?')
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.dropna(subset=['DateTime'])

    for col in column_names[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df['hour'] = df['DateTime'].dt.floor('H')
    df['date'] = df['DateTime'].dt.date
    df['RR'] = df['RR'] / 10.0
    df['sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - (
            df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
    )
    daily_df = df.groupby('date').agg({
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'sub_metering_remainder': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }).reset_index()
    return daily_df

# ------------------------
# 2. 数据准备与标准化
# ------------------------
train_df = load_and_preprocess("train.csv")
test_df = load_and_preprocess_test("test.csv")
test_df = test_df.sort_values('date')

corr_matrix = train_df.drop(columns=['date']).corr(method='pearson')

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={'label': 'Pearson相关系数'}, linewidths=0.5, annot_kws={"size": 8})
plt.title("特征与目标之间的Pearson相关系数热力图")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# 特征选择
X_fs = train_df.drop(columns=['date', 'Global_active_power'])
y_fs = train_df['Global_active_power']
selector = SelectKBest(score_func=f_regression, k=7)
selector.fit(X_fs, y_fs)
f_scores = selector.scores_
feature_names = X_fs.columns

# 绘制特征重要性柱状图
plt.figure(figsize=(12, 6))
plt.bar(feature_names, f_scores, color='skyblue')
plt.xticks(rotation=45)
plt.title("各特征的F检验分数（用于特征选择）")
plt.ylabel("F-score")
plt.tight_layout()
plt.grid(True)
plt.show()
selected_columns = X_fs.columns[selector.get_support()].tolist()
print("选中的特征：", selected_columns)
# 保留选中列
train_df = train_df[['date', 'Global_active_power'] + selected_columns]
test_df = test_df[['date', 'Global_active_power'] + selected_columns]

# 特征和标签分开缩放
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(train_df[selected_columns])
y_train = scaler_y.fit_transform(train_df[['Global_active_power']])

X_test = scaler_X.transform(test_df[selected_columns])
y_test = scaler_y.transform(test_df[['Global_active_power']])


def create_sequences(data, target, input_len, output_len):
    """
    生成滑动窗口输入数据和目标值
    """
    x, y = [], []
    for i in range(len(data) - input_len - output_len+1):
        x.append(data[i:i + input_len])
        y.append(target[i + input_len:i + input_len + output_len])
    return np.array(x), np.array(y)


input_len = 90 # 输入时间步
output_len = 365 # 输出时间步

# 使用 X_train 和 y_train 创建训练序列
x_train, y_train_seq = create_sequences(X_train, y_train, input_len, output_len)
x_test,y_test_seq = create_sequences(X_test, y_test, input_len, output_len)


# ===========================
# 构建 LSTM 模型
# ===========================

mse_scores = []
mae_scores = []
best_history = None
best_val_loss = float('inf')

num_experiments = 5

for i in range(num_experiments):
    print(f"Starting Experiment {i + 1}/{num_experiments}...")

    model = models.Sequential([
        layers.Input(shape=(input_len, x_train.shape[2])),
        # layers.LSTM(128, return_sequences=True, activation='tanh'),
        # layers.Dropout(0.1),
        # layers.LSTM(128, return_sequences=True, activation='tanh'),
        # layers.Dropout(0.1),
        layers.LSTM(200, return_sequences=False, activation='tanh'),
        layers.Dropout(0.1),
        layers.Dense(output_len)
    ])
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=100,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train_seq,
        validation_data=(x_test, y_test_seq),
        epochs=20000,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )
    current_min_val_loss = min(history.history['val_loss'])
    if current_min_val_loss < best_val_loss:
        best_val_loss = current_min_val_loss
        best_history = history
    mse, mae = model.evaluate(x_test, y_test_seq, verbose=0)
    print(f"Experiment {i + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    mse_scores.append(mse)
    mae_scores.append(mae)

mse_mean = np.mean(mse_scores)
mse_std = np.std(mse_scores)
mae_mean = np.mean(mae_scores)
mae_std = np.std(mae_scores)
print("\nFinal Results:")
print("MSE Scores:", mse_scores)
print(f"Mean MSE: {mse_mean:.4f}, Std MSE: {mse_std:.4f}")
print("MAE Scores:", mae_scores)
print(f"Mean MAE: {mae_mean:.4f}, Std MAE: {mae_std:.4f}")

preds = model.predict(x_test)  # preds.shape = (num_windows, output_len)
print(len(preds)," ",len(preds[0]))
preds_real_all = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
final_predictions = np.zeros_like(test_df['Global_active_power'].values)
print(len(final_predictions),len(preds_real_all),preds_real_all)
counts = np.zeros_like(test_df['Global_active_power'].values)
for i in range(len(preds_real_all)):
    start_idx = i + input_len
    end_idx = start_idx + output_len
    for j in range(output_len):
        final_predictions[start_idx + j] = preds_real_all[i][j]
        counts[start_idx + j] += 1
counts[counts == 0] = 1
#final_predictions /= counts
preds_val = final_predictions[input_len:]
print(len(preds_val),preds_val)

y_real = test_df['Global_active_power'].values[input_len:]

mse_unscaled = mean_squared_error(y_real, preds_val)
mae_unscaled = mean_absolute_error(y_real, preds_val)

print(f"反归一化后的 MSE: {mse_unscaled:.4f}")
print(f"反归一化后的 MAE: {mae_unscaled:.4f}")
plt.figure(figsize=(16, 8))
plt.plot(y_real, label='真实值', color='blue')
plt.plot(preds_val, label='预测值', color='orange')
plt.title("总有功功率预测")
plt.xlabel("时间步")
plt.ylabel("总有功功率 (kW)")
plt.legend()
plt.grid(True)
plt.show()
# 绘制训练过程的 loss 和 val_loss 曲线
plt.figure(figsize=(10, 6))
plt.plot(best_history.history['loss'], label='训练损失')
plt.title("最佳实验的训练损失变化")
plt.xlabel("训练轮次")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()