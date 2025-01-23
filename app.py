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

if st.button("예측하기", key="predict_button"):
    model = joblib.load("decision_tree_model.pkl")
    prediction = model.predict([[step_time]])
    
    class_labels = {0: "Elderly (노인)", 1: "Young Adults (청년)", 2: "Parkinson's Disease (파킨슨병이 의심됩니다)"}
    st.write(f"예측 결과: **{class_labels[prediction[0]]}**") 

st.markdown("---") 

if "show_data_image" not in st.session_state:
    st.session_state.show_data_image = False
if "show_graph1" not in st.session_state:
    st.session_state.show_graph1 = False
if "show_graph2" not in st.session_state:
    st.session_state.show_graph2 = False


col1, col2, col3 = st.columns(3)  

with col1:
    if st.button("보행 데이터 보기", key="data_button"):
        st.session_state.show_data_image = not st.session_state.show_data_image 
    if st.session_state.show_data_image:
        st.image("3.png", caption="Stride Data Overview", use_container_width=True)

with col2:
    if st.button("이중 분류일때 \nconfusion Matrix", key="graph_button_1"):
        st.session_state.show_graph1 = not st.session_state.show_graph1 
    if st.session_state.show_graph1:
        st.image("1.png", caption="Step Time Distribution 1", use_container_width=True)

with col3:
    if st.button("이중 분류일때 \nROC Curve", key="graph_button_2"):
        st.session_state.show_graph2 = not st.session_state.show_graph2 
    if st.session_state.show_graph2:
        st.image("2.png", caption="Step Time Distribution 2", use_container_width=True)
