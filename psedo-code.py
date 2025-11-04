import random
from typing import Dict, Any

# Input Generator
class InputGenerator:
    def __init__(self, attack_prob=0.5):
        self.attack_prob = attack_prob
    def generate(self) -> Dict[str, Any]:
        """공격자 or 정상 중 하나를 뽑아서 쿼리 생성 """
        if random.rando() < self.attack_prob:
            role="attacker"
            query="attacker 소환해서 보내기?"
            ### atcker 지정해주기
        else:
            role="normal"
            query="일반적인 물음" ##
        return {'role':role, 'query':query}


# Monitor: input인지 아닌지 예측하는 것?



# LLM Protectorer: 학습 대상


# Target LLM


# Evaluator: LLM or Rule based?



#학습
def train(num_steps=):



    for step in range(num_steps):
        # 1 쿼리 생성 -> 기존 코드에 사용
        q = gener