# ollama 멀티턴 질의응답

import ollama

model = "deepcoder"

# 대화 시작하기
messages = [{"role": "user", "content": "머신러닝이란 무엇인가요?"}]

response = ollama.chat(model=model, messages=messages)
print("AI:", response["message"]["content"])

# 대화 계속하기
messages.append(response["message"])
messages.append({"role": "user", "content": "예시를 들어서 설명해주세요"})

response = ollama.chat(model=model, messages=messages)
print("AI:", response["message"]["content"])
