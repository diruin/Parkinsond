import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

# CSS 스타일 적용
st.markdown(
    """
    <style>
    body {
        background-color: #4D0F0F;
    }
    
    .stApp {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
    }

    h1 {
        color: black;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
    }

    .stButton>button {
        background-color: #7F7F7F;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #2980B9;
    }

    .stNumberInput>div>div>input {
        font-size: 18px; 
        border-radius: 5px; 
        border: 2px solid #BDC3C7;
        padding: 5px;
    }
    
    .stImage {
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }

    .stMarkdown {
        text-align: center;
    }

    .red-text {
        color: red;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 데이터 불러오기 및 모델 학습
file_path = "simplified_stride_data.csv"
stride_data = pd.read_csv(file_path)

stride_data['Group'] = stride_data['Group'].map({'Elderly': 0, 'Young Adults': 1, "Parkinson's Disease": 2})

X = stride_data[['Step Time (s)']]
y = stride_data['Group']

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)

joblib.dump(dt_model, "decision_tree_model.pkl")

st.title("🚶‍♂️ 파킨슨병 진단 시스템")

step_time = st.number_input("📏 Step Time (s) 값을 입력하세요:", min_value=0.1, max_value=3.0, step=0.01)

if st.button("🔍 예측하기", key="predict_button"):
    model = joblib.load("decision_tree_model.pkl")
    prediction = model.predict([[step_time]])

    class_labels = {
        0: "Elderly (노인)",
        1: "Young Adults (청년)",
        2: '<span class="red-text">⚠️ Parkinson\'s Disease (파킨슨병이 의심됩니다)</span>'
    }
    
    st.markdown(f"**예측 결과: {class_labels[prediction[0]]}**", unsafe_allow_html=True)

st.markdown("---")

if "show_data_image" not in st.session_state:
    st.session_state.show_data_image = False
if "show_graph1" not in st.session_state:
    st.session_state.show_graph1 = False
if "show_graph2" not in st.session_state:
    st.session_state.show_graph2 = False

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 보행 데이터 보기", key="data_button"):
        st.session_state.show_data_image = not st.session_state.show_data_image
    if st.session_state.show_data_image:
        st.image("3.png", caption="DataSet", use_container_width=True, output_format="auto")

with col2:
    if st.button("📉 Confusion Matrix", key="graph_button_1"):
        st.session_state.show_graph1 = not st.session_state.show_graph1
    if st.session_state.show_graph1:
        st.image("1.png", caption="Confusion Matrix", use_container_width=True, output_format="auto")

with col3:
    if st.button("📈 ROC Curve", key="graph_button_2"):
        st.session_state.show_graph2 = not st.session_state.show_graph2
    if st.session_state.show_graph2:
        st.image("2.png", caption="ROC Curve", use_container_width=True, output_format="auto")
