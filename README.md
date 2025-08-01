# 🧿 정책 문서 기반 고객 불만 자동 분류 & 대응 챗봇

---

## 🎯 주제  
정책 문서를 기반으로 Retrieval-Augmented Generation(RAG)을 활용해 고객 불만을 자동으로 분류하고, 시나리오 기반 맞춤 응답을 제공하는 챗봇 시스템

---

## 🎉 서비스 내용  
- **문서 기반 Q&A**: 사내 정책 문서(FAQ, 가이드라인 등)를 벡터화하여 RAG 엔진으로 정확한 답변 제공  
- **불만 유형 자동 분류**: 환불, 배송 지연, 서비스 불만, 기타 항목으로 자동 태깅  
- **시나리오 분기 응답**: 분류된 유형별로 미리 설계한 응답 플로우에 따라 대화 분기  
- **Streamlit UI**: 카카오톡·GPT 스타일의 직관적인 웹 인터페이스 지원  
- **인덱스 자동 갱신**: `policy_docs/` 폴더 내용 변경 시마다 FAISS 인덱스 자동 재생성

---

## 🗓️ 프로젝트 수행 기간  
2025.07.28 ~ 2025.08.08

---

## 📚 Tech Stack  

![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)  
![LangChain](https://img.shields.io/badge/langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)  
![FAISS](https://img.shields.io/badge/FAISS-000000?style=for-the-badge&logo=faiss&logoColor=white)  
![OpenAI API](https://img.shields.io/badge/OpenAI%20API-412991?style=for-the-badge&logo=openai&logoColor=white)  
![dotenv](https://img.shields.io/badge/dotenv-212121?style=for-the-badge&logo=dotenv&logoColor=white)

---

## 🛠 Tools  

![GitHub](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)  
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)  
![Anaconda](https://img.shields.io/badge/anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)  
![Git](https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white)

---

## 🔗 링크  
- **코드 리포지토리**: https://github.com/jongs96/CustomerCareChatBot
---
