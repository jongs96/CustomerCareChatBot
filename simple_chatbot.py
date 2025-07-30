def simple_chatbot(user_input):
    # 사용자 입력을 소문자로 변환
    user_input = user_input.lower().strip()
    # 간단한 규칙 기반 응답
    responses = {
        "제품이 고장": "죄송합니다. 제품 고장에 대해 안내드리겠습니다. 구매하신 날짜와 증상을 말씀해 주세요.",
        "환불": "환불 절차를 안내드리겠습니다. 주문 번호를 알려주시겠습니까?",
        "배송이 늦": "배송 지연에 대해 사과드립니다. 주문 번호로 확인 후 빠르게 처리하겠습니다.",
        "잘못 배송": "잘못된 배송에 대해 사과드립니다. 주문 번호와 잘못 배송된 제품 정보를 알려주세요."
    }
    # 입력이 정확히 일치하지 않아도 키워드 기반으로 응답 찾기
    for key, response in responses.items():
        if key in user_input:
            return response
    return "죄송합니다. 다시 말씀해 주세요."

# 챗봇 실행
print("고객 불만 처리 챗봇입니다. 불만 사항을 말씀해 주세요 (종료하려면 '종료' 입력):")
while True:
    user_input = input("> ")
    if user_input == "종료":
        print("챗봇을 종료합니다.")
        break
    response = simple_chatbot(user_input)
    print(response)