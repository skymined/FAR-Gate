import random
from typing import Dict, Any

class SimpleEnv:
    def __init__(self, generator, monitor, protector, target_llm, evaluator):
        self.generator = generator
        self.monitor = monitor
        self.protector = protector
        self.target_llm = target_llm
        self.evaluator = evaluator
        self.t = 0

    def step(self) -> Dict[str, Any]:
        # 1) 쿼리 만들기 (attacker or normal)
        q = self.generator.generate()          # {role, query, ...}

        # 2) 모니터로 추가정보 얻기
        meta = self.monitor.observe(q)         # {freq_score, suspicious_pattern, ...}

        # 3) protector가 행동 선택
        act = self.protector.decide(q, meta)   # {protect, score, extra_prompt}

        # 4) LLM이 실제 답 생성
        output = self.target_llm.generate(q["query"], act["extra_prompt"])

        # 5) evaluator가 보상 계산
        reward = self.evaluator.evaluate(q["role"], act, output)

        # 6) protector에 보상 전달해서 업데이트
        self.protector.update(reward)

        # 관찰을 다음 step에도 넘기고 싶으면 여기서 리턴
        self.t += 1
        return {
            "query": q,
            "meta": meta,
            "action": act,
            "output": output,
            "reward": reward
        }
