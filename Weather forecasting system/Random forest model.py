import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
import tensorflow as tf
import numpy as np

# 加载数据集
file_path = 'C:/Users/31606/Desktop/weatherHistory.csv'
weather_data = pd.read_csv(file_path)

# 处理缺失值
weather_data['Precip Type'].fillna(weather_data['Precip Type'].mode()[0], inplace=True)

# 编码分类变量
le_summary = LabelEncoder()
le_precip_type = LabelEncoder()
weather_data['Summary'] = le_summary.fit_transform(weather_data['Summary'])
weather_data['Precip Type'] = le_precip_type.fit_transform(weather_data['Precip Type'])
weather_data['Daily Summary'] = weather_data['Daily Summary'].astype('category').cat.codes

# 转换日期列并提取日期特征
weather_data['Formatted Date'] = pd.to_datetime(weather_data['Formatted Date'], utc=True)
weather_data['Year'] = weather_data['Formatted Date'].dt.year
weather_data['Month'] = weather_data['Formatted Date'].dt.month
weather_data['Day'] = weather_data['Formatted Date'].dt.day
weather_data['Hour'] = weather_data['Formatted Date'].dt.hour
weather_data.drop(columns=['Formatted Date'], inplace=True)

# 定义特征和目标变量
X = weather_data.drop(columns=['Summary', 'Precip Type', 'Temperature (C)', 'Daily Summary'])
y_summary = weather_data['Summary']
y_precip_type = weather_data['Precip Type']
y_temperature = weather_data['Temperature (C)']
y_daily_summary = weather_data['Daily Summary']

# 拆分数据集
X_train, X_test, y_train_summary, y_test_summary = train_test_split(X, y_summary, test_size=0.2, random_state=42)
_, _, y_train_precip_type, y_test_precip_type = train_test_split(X, y_precip_type, test_size=0.2, random_state=42)
_, _, y_train_temperature, y_test_temperature = train_test_split(X, y_temperature, test_size=0.2, random_state=42)
_, _, y_train_daily_summary, y_test_daily_summary = train_test_split(X, y_daily_summary, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义神经网络模型
input_layer = keras.layers.Input(shape=(X_train_scaled.shape[1],))
dense_layer = keras.layers.Dense(128, activation='relu')(input_layer)
dense_layer = keras.layers.Dense(64, activation='relu')(dense_layer)

output_summary = keras.layers.Dense(len(le_summary.classes_), activation='softmax', name='summary_output')(dense_layer)
output_precip_type = keras.layers.Dense(len(le_precip_type.classes_), activation='softmax', name='precip_type_output')(dense_layer)
output_temperature = keras.layers.Dense(1, name='temperature_output')(dense_layer)
output_daily_summary = keras.layers.Dense(weather_data['Daily Summary'].nunique(), activation='softmax', name='daily_summary_output')(dense_layer)

model = keras.models.Model(inputs=input_layer, outputs=[output_summary, output_precip_type, output_temperature, output_daily_summary])

# 编译模型，使用字符串形式的损失函数定义
model.compile(optimizer='adam',
              loss={'summary_output': 'sparse_categorical_crossentropy',
                    'precip_type_output': 'sparse_categorical_crossentropy',
                    'temperature_output': 'mean_squared_error',
                    'daily_summary_output': 'sparse_categorical_crossentropy'},
              metrics={'summary_output': 'accuracy',
                       'precip_type_output': 'accuracy',
                       'temperature_output': 'mae',
                       'daily_summary_output': 'accuracy'})

# 训练模型
history = model.fit(X_train_scaled,
                    {'summary_output': y_train_summary,
                     'precip_type_output': y_train_precip_type,
                     'temperature_output': y_train_temperature,
                     'daily_summary_output': y_train_daily_summary},
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)

# 保存神经网络模型
model.save('multi_output_weather_model.h5')

# 将模型转换为TensorFlow Lite格式
model = tf.keras.models.load_model('multi_output_weather_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存TensorFlow Lite模型到microSD卡的F盘
with open('F:/weather_model.tflite', 'wb') as f:
    f.write(tflite_model)
