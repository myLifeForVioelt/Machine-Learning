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
import seaborn as sns
from collections import Counter
import heapq

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

def extract_topk_sketch_feature(df, k=3):
    topk_features = []
    for idx in range(len(df)):
        row = df.iloc[max(0, idx - 6):idx + 1]
        powers = row['Global_active_power'].values
        counter = Counter(np.round(powers, 2))
        topk = heapq.nlargest(k, counter.items(), key=lambda x: x[1])
        feature = [val for val, cnt in topk]
        feature += [0] * (k - len(feature))
        topk_features.append(feature)
    return np.array(topk_features)

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
        df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
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

def load_and_preprocess_test(path):
    column_names = [
        'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
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
        df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
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

train_df = load_and_preprocess("train.csv")
test_df = load_and_preprocess_test("test.csv")
test_df = test_df.sort_values('date')

X_fs = train_df.drop(columns=['date', 'Global_active_power'])
y_fs = train_df['Global_active_power']
selector = SelectKBest(score_func=f_regression, k=7)
selector.fit(X_fs, y_fs)
selected_columns = X_fs.columns[selector.get_support()].tolist()
train_df = train_df[['date', 'Global_active_power'] + selected_columns]
test_df = test_df[['date', 'Global_active_power'] + selected_columns]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_base = train_df[selected_columns].values
X_test_base = test_df[selected_columns].values

topk_train_feats = extract_topk_sketch_feature(train_df, k=3)
topk_test_feats = extract_topk_sketch_feature(test_df, k=3)

X_train = scaler_X.fit_transform(np.hstack([X_train_base, topk_train_feats]))
y_train = scaler_y.fit_transform(train_df[['Global_active_power']])

X_test = scaler_X.transform(np.hstack([X_test_base, topk_test_feats]))
y_test = scaler_y.transform(test_df[['Global_active_power']])

def create_sequences(data, target, input_len, output_len):
    x, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        x.append(data[i:i + input_len])
        y.append(target[i + input_len:i + input_len + output_len])
    return np.array(x), np.array(y)

input_len = 90
output_len = 360
x_train, y_train_seq = create_sequences(X_train, y_train, input_len, output_len)
x_test, y_test_seq = create_sequences(X_test, y_test, input_len, output_len)

mse_scores = []
mae_scores = []
best_history = None
best_val_loss = float('inf')

for i in range(5):
    model = Sequential([
        LSTM(200, return_sequences=False, activation='tanh', input_shape=(input_len, x_train.shape[2])),
        Dropout(0.1),
        Dense(output_len)
    ])
    optimizer = Adam(learning_rate=ExponentialDecay(1e-3, decay_steps=10000, decay_rate=0.9))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    history = model.fit(x_train, y_train_seq,
                        validation_data=(x_test, y_test_seq),
                        epochs=100,
                        batch_size=64,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)],
                        verbose=1)
    if min(history.history['val_loss']) < best_val_loss:
        best_val_loss = min(history.history['val_loss'])
        best_history = history
    mse, mae = model.evaluate(x_test, y_test_seq, verbose=0)
    mse_scores.append(mse)
    mae_scores.append(mae)

print("MSE Scores:", mse_scores)
print("MAE Scores:", mae_scores)
print("Mean MSE:", np.mean(mse_scores), "Std:", np.std(mse_scores))
print("Mean MAE:", np.mean(mae_scores), "Std:", np.std(mae_scores))

preds = model.predict(x_test)
preds_real_all = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
final_predictions = np.zeros_like(test_df['Global_active_power'].values)
counts = np.zeros_like(test_df['Global_active_power'].values)
for i in range(len(preds_real_all)):
    for j in range(output_len):
        idx = i + input_len + j
        if idx < len(final_predictions):
            final_predictions[idx] += preds_real_all[i][j]
            counts[idx] += 1
counts[counts == 0] = 1
preds_val = final_predictions[input_len:] / counts[input_len:]
y_real = test_df['Global_active_power'].values[input_len:]

print("\n反归一化后的 MSE:", mean_squared_error(y_real, preds_val))
print("反归一化后的 MAE:", mean_absolute_error(y_real, preds_val))

plt.figure(figsize=(16, 8))
plt.plot(y_real, label='真实值', color='blue')
plt.plot(preds_val, label='预测值', color='orange')
plt.title("总有功功率预测")
plt.xlabel("时间步")
plt.ylabel("总有功功率 (kW)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(best_history.history['loss'], label='训练损失')
plt.title("最佳实验的训练损失变化")
plt.xlabel("训练轮次")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
