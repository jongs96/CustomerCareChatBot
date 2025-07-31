import streamlit as st

st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stTextInput > div > div > input {border-radius: 10px; padding: 10px;}
    </style>
    """, unsafe_allow_html=True)

def simple_chatbot(user_input):
    user_input = user_input.lower().strip()
    responses = {
        "제품이 고장": "죄송합니다. 제품 고장에 대해 안내드리겠습니다. 구매하신 날짜와 증상을 말씀해 주세요.",
        "환불": "환불 절차를 안내드리겠습니다. 주문 번호를 알려주시겠습니까?",
        "배송이 늦": "배송 지연에 대해 사과드립니다. 주문 번호로 확인 후 빠르게 처리하겠습니다.",
        "잘못 배송": "잘못된 배송에 대해 사과드립니다. 주문 번호와 잘못 배송된 제품 정보를 알려주세요.",
        "결제 오류": "결제 오류에 대해 확인드리겠습니다. 결제 시도한 시간과 방법을 알려주세요."
    }

    for key, response in responses.items():
        if key in user_input:
            return response
    return "죄송합니다. 질문이 불분명합니다. 불만 사항을 구체적으로 말씀해 주시면 더 빠르게 도와드리겠습니다!"

st.title("고객 불만 처리 챗봇")
st.write("불만 사항을 입력하세요:")

user_input = st.text_input("입력", key="user_input")

if user_input:
    response = simple_chatbot(user_input)
    st.write("챗봇 응답:", response)