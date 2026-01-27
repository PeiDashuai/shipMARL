# tests/test_term_reason.py
def test_term_reason_present(env_factory):
    # 保持默认 dt，给个较小 Tmax 让它很快结束
    env = env_factory(dict(T_max=1.0))
    env.reset(seed=0)

    reason = None
    for _ in range(20):  # 最多尝试 20 步
        acts = {aid: env.action_space(aid).sample() for aid in env.agents}
        obs, rew, term, trunc, infos = env.step(acts)

        ended = any(term.values()) or any(trunc.values())
        if ended:
            reasons = {infos[aid].get("term_reason") for aid in env.agents}
            # 允许 timeout / collision / success 任一
            assert any(r in {"timeout", "collision", "success"} for r in reasons)
            reason = next(iter(reasons))
            break

    assert reason is not None, "Episode did not end within 20 steps"