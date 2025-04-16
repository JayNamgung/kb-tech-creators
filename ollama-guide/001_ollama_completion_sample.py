# ollama 싱글턴 질의응답

import ollama

# 텍스트 생성
response = ollama.generate(
    model="deepcoder", prompt="파이썬으로 머신러닝 샘플 코드를 작성해주세요"
)

print(response["response"])
