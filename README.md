- 개괄짜기

```
from openai import OpenAI

# 로컬 LM 서버 주소
client = OpenAI(
    base_url="http://100.119.179.1:1234/api/v1",  # 이 부분만 수정
    api_key="lm-studio"  # LM Studio나 Ollama는 보통 키가 필요 없음
)

response = client.chat.completions.create(
    model="qwen2.5-1.5b-instruct-mlx",  # curl에서 봤던 모델 key
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain reinforcement learning simply."}
    ]
)

print(response.choices[0].message.content)

```
