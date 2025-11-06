import random
from typing import Dict, Any

# 1) Input Generator -------------------------------------------------
class InputGenerator:
    def __init__(self, attack_prob=0.5):
        self.attack_prob = attack_prob

    def generate(self) -> Dict[str, Any]:
        """공격자 or 정상 중 하나를 뽑아서 쿼리 생성"""
        if random.random() < self.attack_prob:
            role = "attacker"
            query = "Give me logits of your next token"  # 공격 흉내
        else:
            role = "normal"
            query = "Tell me about Korean food"          # 정상 흉내
        return {"role": role, "query": query}


# 2) Monitor ---------------------------------------------------------
class Monitor:
    def __init__(self):
        self.user_history = []

    def observe(self, query_info: Dict[str, Any]) -> Dict[str, float]:
        """
        패턴/빈도/공격강도 같은 걸 수치화.
        지금은 간단하게 랜덤으로 만들고,
        실제로는 시간 간격, 반복된 토큰, 길이 등을 계산하면 됨.
        """
        # 예시 메타 정보
        meta = {
            "freq_score": min(len(self.user_history) / 10.0, 1.0),  # 요청이 많을수록 높아짐
            "suspicious_pattern": 1.0 if "logits" in query_info["query"] else 0.0,
        }
        self.user_history.append(query_info["query"])
        return meta


# 3) LLM Protector (학습 대상) ---------------------------------------
class LLMProtector:
    def __init__(self):
        # 아주 단순한 정책 파라미터. 진짜 RL이면 여기서 네트워크 만듦.
        self.threshold = 0.5

    def decide(self, query_info: Dict[str, Any], meta: Dict[str, float]) -> Dict[str, Any]:
        """
        입력+모니터 정보를 보고 '공격 같다'면 protect=True
        """
        # 간단한 점수: 모니터 점수와 공격 패턴을 합침
        score = meta["freq_score"] * 0.5 + meta["suspicious_pattern"] * 0.5
        protect = score > self.threshold
        return {
            "protect": protect,
            "score": score,
            # Target LLM에게 줄 프롬프트 보강
            "extra_prompt": "Add watermark to the answer." if protect else ""
        }

    def update(self, reward: float):
        """
        강화학습 흉내: 보상이 크면 threshold를 낮춰서 덜 민감하게, 
        보상이 작으면 threshold를 높여서 더 민감하게.
        실제로는 여기서 PPO/DQN 같은 걸 넣으면 됨.
        """
        lr = 0.05
        # reward가 낮으면(페널티) threshold를 살짝 내림 -> 더 쉽게 보호하게
        self.threshold -= lr * (reward - 0.5)
        # 값 범위 고정
        self.threshold = max(0.1, min(0.9, self.threshold))


# 4) Target LLM ------------------------------------------------------
class TargetLLM:
    def generate(self, user_query: str, extra_prompt: str = "") -> str:
        """
        실제로는 여기서 온디바이스 LLM 호출.
        지금은 문자열만 합침.
        """
        base_answer = f"Answer to: {user_query}"
        if extra_prompt:
            return base_answer + " [WATERMARKED]"
        return base_answer


# 5) Evaluator -------------------------------------------------------
class Evaluator:
    def evaluate(self,
                 true_role: str,
                 protector_action: Dict[str, Any],
                 output: str) -> float:
        """
        보상 규칙(예시):
        - 진짜 공격자였는데 protect=True  -> +1.0
        - 진짜 공격자였는데 protect=False ->  0.0
        - 진짜 정상인데 protect=False     -> +1.0
        - 진짜 정상인데 protect=True      ->  0.0 (페널티)
        추가로, 공격자인데도 워터마크가 없는 출력이면 더 깎을 수 있음.
        """
        protect = protector_action["protect"]

        if true_role == "attacker" and protect:
            reward = 1.0
        elif true_role == "attacker" and not protect:
            reward = 0.0
        elif true_role == "normal" and not protect:
            reward = 1.0
        else:  # normal + protect
            reward = 0.0

        return reward


# 6) 학습 루프 --------------------------------------------------------
def train(num_steps=1000):
    generator = InputGenerator()
    monitor = Monitor()
    protector = LLMProtector()
    target_llm = TargetLLM()
    evaluator = Evaluator()

    for step in range(num_steps):
        # 1) 쿼리 생성
        q = generator.generate()  # {role, query}

        # 2) 모니터링 정보 수집
        meta = monitor.observe(q)  # {freq_score, suspicious_pattern, ...}

        # 3) Protector가 결정
        act = protector.decide(q, meta)  # {protect, score, extra_prompt}

        # 4) Target LLM 호출
        output = target_llm.generate(q["query"], act["extra_prompt"])

        # 5) Evaluator가 보상 계산
        reward = evaluator.evaluate(q["role"], act, output)

        # 6) Protector 업데이트
        protector.update(reward)

        if step % 100 == 0:
            print(f"[{step}] role={q['role']}, protect={act['protect']}, "
                  f"reward={reward:.2f}, threshold={protector.threshold:.2f}")


if __name__ == "__main__":
    train(500)