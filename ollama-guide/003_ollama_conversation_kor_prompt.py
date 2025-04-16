# ollama 멀티턴 질의응답 - 한글 답변 강화

import ollama

model = "deepcoder"

# 한글 응답 강화를 위한 시스템 프롬프트
system_prompt = """당신은 매우 유능한 한국어 AI 도우미입니다. 
항상 한국어로 자연스럽게 답변하며, 한국어 맞춤법과 문법을 정확하게 사용합니다.
전문적인 내용이라도 한국인이 이해하기 쉽게 설명해주세요.
영어 단어는 필요한 경우에만 사용하고, 가능한 한국어로 설명합니다."""

# 대화 시작하기
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "머신러닝이란 무엇인가요?"},
]

response = ollama.chat(model=model, messages=messages)
print("AI:", response["message"]["content"])

# 대화 계속하기
messages.append(response["message"])
messages.append({"role": "user", "content": "예시를 들어서 설명해주세요"})

response = ollama.chat(model=model, messages=messages)
print("AI:", response["message"]["content"])
