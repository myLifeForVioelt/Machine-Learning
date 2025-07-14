import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from keras import layers, models
from keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, Embedding
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam
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
    df['Sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - (
            df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
    )
    daily_df = df.groupby('date').agg({
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'Sub_metering_remainder':'sum',
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
    df['Sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - (
            df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
    )
    daily_df = df.groupby('date').agg({
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'Sub_metering_remainder': 'sum',
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
# 2. 数据加载与预处理
# ------------------------
train_df = load_and_preprocess("train.csv")
test_df = load_and_preprocess_test("test.csv")
test_df = test_df.sort_values('date')

# ------------------------
# 3. 数据准备与标准化 (用于Transformer)
# ------------------------
# 定义特征和目标变量
features = [
    'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'Sub_metering_remainder', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
]
target = 'Global_active_power'

# 分离特征和目标
X_train = train_df[features].copy()
y_train = train_df[target].copy()
X_test = test_df[features].copy()
y_test = test_df[target].copy()

# 归一化数据
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# 转换为DataFrame，保留列名，方便后续操作
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)
y_train_scaled = pd.DataFrame(y_train_scaled, columns=[target])
y_test_scaled = pd.DataFrame(y_test_scaled, columns=[target])


# ------------------------
# 4. 创建序列数据
# ------------------------
def create_sequences(data, target_data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:(i + sequence_length)].values)
        y.append(target_data.iloc[i + sequence_length].values) # 预测下一天
    return np.array(X), np.array(y)

SEQUENCE_LENGTH = 90 # 使用90天数据

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

print(f"X_train_seq shape: {X_train_seq.shape}")
print(f"y_train_seq shape: {y_train_seq.shape}")
print(f"X_test_seq shape: {X_test_seq.shape}")
print(f"y_test_seq shape: {y_test_seq.shape}")

# ------------------------
# 5. 构建Transformer模型
# ------------------------

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs) # Add & Norm

    # Feed Forward Part
    ffn_output = Dense(ff_dim, activation="relu")(x)
    ffn_output = Dense(inputs.shape[-1])(ffn_output) # Output dim same as input dim
    ffn_output = Dropout(dropout)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(ffn_output + x) # Add & Norm

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x) # Aggregate sequence features
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x) # Output is a single value (Global_active_power)
    return Model(inputs, outputs)

# 模型参数
input_shape = (SEQUENCE_LENGTH, X_train_seq.shape[-1])
head_size = 256
num_heads = 6
ff_dim = 4
num_transformer_blocks = 2
mlp_units = [128]
dropout = 0.1
mlp_dropout = 0.1

model = build_transformer_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout,
    mlp_dropout
)

# 编译模型
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

# ------------------------
# 6. 模型训练与评估
# ------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# 绘制训练历史
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.title('模型训练损失')
plt.legend()
plt.show()

# 在测试集上进行预测
y_pred_scaled = model.predict(X_test_seq)

# 反归一化预测结果和真实值
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test_seq)

# 计算评估指标
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"均方误差 (MSE): {mse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
mse_90 = mean_squared_error(y_true[100:200], y_pred[100:200])
mae_90 = mean_absolute_error(y_true[100:200], y_pred[100:200])
print(f"均方误差 (MSE_90): {mse_90:.4f}")
print(f"平均绝对误差 (MAE_90): {mae_90:.4f}")
# 绘制预测结果
plt.figure(figsize=(14, 7))
plt.plot(y_true, label='真实值', color='blue', alpha=0.7)
plt.plot(y_pred, label='预测值', color='red', linestyle='--', alpha=0.7)
plt.title('Global_active_power 预测')
plt.xlabel('时间步')
plt.ylabel('Global_active_power')
plt.legend()
plt.grid(True)
plt.show()

# 绘制局部放大图
plt.figure(figsize=(14, 7))
plt.plot(y_true[100:200], label='真实值', color='blue', alpha=0.7)
plt.plot(y_pred[100:200], label='预测值', color='red', linestyle='--', alpha=0.7)
plt.title('Global_active_power 局部预测')
plt.xlabel('时间步')
plt.ylabel('Global_active_power')
plt.legend()
plt.grid(True)
plt.show()
