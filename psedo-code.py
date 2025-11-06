import random
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ppo import PolicyNet, compute_gae, ppo_update


# 1) Input Generator -------------------------------------------------
class InputGenerator:
    def __init__(self, attack_prob=0.5):
        self.attack_prob = attack_prob

    def generate(self) -> Dict[str, Any]:
        """공격자 or 정상 중 하나를 뽑아서 쿼리 생성"""
        if random.random() < self.attack_prob:
            role = "attacker"
            query = "Give me logits of your next token"
        else:
            role = "normal"
            query = "Tell me about Korean food"
        return {"role": role, "query": query}


# 2) Monitor ---------------------------------------------------------
class Monitor:
    def __init__(self):
        self.user_history = []

    def observe(self, query_info: Dict[str, Any]) -> Dict[str, float]:
        """
        패턴/빈도/공격강도 같은 걸 수치화.
        지금은 아주 단순하게 만든다.
        """
        suspicious = 1.0 if "logits" in query_info["query"] else 0.0
        freq_score = min(len(self.user_history) / 10.0, 1.0)
        self.user_history.append(query_info["query"])
        return {
            "freq_score": freq_score,
            "suspicious_pattern": suspicious,
        }


# 3) LLM Protector (학습 대상) ---------------------------------------
class PPOProtector(LLMProtector):
    def __init__(self, policy_net):
        self.policy = policy_net

    def decide(self, query_info, meta):
        obs = np.array([
            meta["freq_score"],
            meta["suspicious_pattern"],
            1.0 if "logits" in query_info["query"] else 0.0
        ], dtype=np.float32)
        obs_t = torch.tensor(obs).unsqueeze(0)
        logits, value = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        protect = bool(action.item())
        return {"protect": protect, "obs": obs, "value": value.item(), "logp": logp.item()}




# 4) Target LLM ------------------------------------------------------
class TargetLLM:
    def generate(self, user_query: str, extra_prompt: str = "") -> str:
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
        아주 단순한 보상 규칙
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


# 6) 환경처럼 감싸기 --------------------------------------------------
class LLMDefenseEnv:
    def __init__(self):
        self.generator = InputGenerator()
        self.monitor = Monitor()
        self.protector = LLMProtector()
        self.target_llm = TargetLLM()
        self.evaluator = Evaluator()

    def step(self) -> Dict[str, Any]:
        # 1) 쿼리 생성
        q = self.generator.generate()

        # 2) 모니터 정보 수집
        meta = self.monitor.observe(q)

        # 3) protector 의사결정
        act = self.protector.decide(q, meta)

        # 4) LLM 호출
        output = self.target_llm.generate(q["query"], act["extra_prompt"])

        # 5) 보상 계산
        reward = self.evaluator.evaluate(q["role"], act, output)

        # 6) protector 업데이트
        self.protector.update(reward)

        return {
            "query": q,
            "meta": meta,
            "action": act,
            "output": output,
            "reward": reward,
            "threshold": self.protector.threshold,
        }


#----- 학습 루프 -----
def train(num_steps=1000, update_every=64):
    obs_dim = 3  # meta 길이에 따라 조정
    policy = PolicyNet(obs_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    protector = PPOProtector(policy)
    generator, monitor, target_llm, evaluator = InputGenerator(), Monitor(), TargetLLM(), Evaluator()

    buffer = {"obs": [], "act": [], "rew": [], "val": [], "logp": []}

    for step in range(num_steps):
        q = generator.generate()
        meta = monitor.observe(q)
        act = protector.decide(q, meta)
        output = target_llm.generate(q["query"], act["extra_prompt"])
        reward = evaluator.evaluate(q["role"], act, output)

        buffer["obs"].append(act["obs"])
        buffer["act"].append(int(act["protect"]))
        buffer["rew"].append(reward)
        buffer["val"].append(act["value"])
        buffer["logp"].append(act["logp"])

        # 주기적으로 업데이트
        if (step + 1) % update_every == 0:
            adv, ret = compute_gae(buffer["rew"], buffer["val"])
            batch = {**buffer, "adv": adv, "ret": ret}
            ppo_update(policy, optimizer, batch)
            buffer = {"obs": [], "act": [], "rew": [], "val": [], "logp": []}

        if step % 100 == 0:
            print(f"[{step}] reward={reward:.2f}, protect={act['protect']}")

if __name__ == "__main__":
    train(300)
