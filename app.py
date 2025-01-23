import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

file_path = "simplified_stride_data.csv"
stride_data = pd.read_csv(file_path)

stride_data['Group'] = stride_data['Group'].map({'Elderly': 0, 'Young Adults': 1, "Parkinson's Disease": 2})

X = stride_data[['Step Time (s)']]
y = stride_data['Group']

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)

joblib.dump(dt_model, "decision_tree_model.pkl")


st.title("보행 데이터 분류기")

step_time = st.number_input("Step Time (s) 값을 입력하세요:", min_value=0.1, max_value=3.0, step=0.01)

col1, col2, col3 = st.columns(3)  

with col1:
    if st.button("보행 데이터 보기", key="data_button"):
        st.image("3.png", caption="Stride Data Overview", use_column_width=True)

with col2:
    if st.button("분포 그래프 보기 (1)", key="graph_button_1"):
        st.image("1.png", caption="Step Time Distribution 1", use_column_width=True)

with col3:
    if st.button("분포 그래프 보기 (2)", key="graph_button_2"):
        st.image("2.png", caption="Step Time Distribution 2", use_column_width=True)

if st.button("예측하기"):
    model = joblib.load("decision_tree_model.pkl")
    prediction = model.predict([[step_time]])
    
    class_labels = {0: "Elderly (노인)", 1: "Young Adults (청년)", 2: "Parkinson's Disease (파킨슨병이 의심됩니다)"}
    st.write(f"예측 결과: **{class_labels[prediction[0]]}**") 
