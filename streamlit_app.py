import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import json

# 定义逆变换函数
def inverse_transform(scaler, data, n_features):
    data_extended = np.zeros((data.shape[0], n_features))
    data_extended[:, 0] = data.flatten()
    return scaler.inverse_transform(data_extended)[:, 0]

# 定义预测发电量的函数
def predict_power_generation(input_data, model, scaler):
    input_data = input_data.sort_values(by='日期')
    input_data = input_data[['总发电量', '星期', '天数']]
    scaled_input_data = scaler.transform(input_data)
    X_input = np.array([scaled_input_data])
    X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], X_input.shape[2])
    predictions = model.predict(X_input)
    n_features = scaled_input_data.shape[1]
    predictions_inv = inverse_transform(scaler, predictions[0], n_features)
    return predictions_inv

# 加载模型和归一化参数
model_path = 'lstm_power_generation_model.h5'
scaler_path = 'scaler.joblib'
model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Streamlit界面
st.title("Power Generation Prediction")
uploaded_file = st.file_uploader("Choose a JSON file", type="json")

if uploaded_file is not None:
    input_data_list = json.load(uploaded_file)
    input_data = pd.DataFrame(input_data_list)
    input_data['日期'] = pd.to_datetime(input_data['日期'])
    input_data['星期'] = input_data['日期'].dt.weekday
    input_data['天数'] = input_data['日期'].dt.dayofyear
    input_data = input_data.tail(15)
    predictions = predict_power_generation(input_data, model, scaler)
    st.write("Predicted Power Generation:", predictions)
