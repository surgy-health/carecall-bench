"""eval/score.py on small hand-made results files (no AWS, no tau2 run)."""

import json
import math

import pytest

from eval import score


def write_run(tmp_path, name, model, outcomes, cost=1.0):
    """outcomes: {(task_id, trial): reward or None}."""
    d = tmp_path / name
    d.mkdir()
    sims = [
        {"task_id": t, "trial": tr, "reward_info": None if r is None else {"reward": r}}
        for (t, tr), r in outcomes.items()
    ]
    (d / "results.json").write_text(json.dumps({"simulations": sims}), encoding="utf-8")
    (d / "summary.json").write_text(
        json.dumps({"agent_llm": f"bedrock/{model}", "metered_usd": cost}), encoding="utf-8"
    )
    return d


def test_wilson_interval_known_values():
    lo, hi = score.wilson(97, 100)
    assert round(lo, 3) == 0.915 and round(hi, 3) == 0.990
    assert score.wilson(0, 0)[0] != score.wilson(0, 0)[0]  # nan
    lo, hi = score.wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.29


def test_mcnemar_exact():
    assert score.mcnemar_exact(0, 0) == 1.0
    assert score.mcnemar_exact(5, 5) == 1.0
    # 10 vs 0: two-sided exact p = 2 / 2**10
    assert math.isclose(score.mcnemar_exact(10, 0), 2 / 1024)
    assert score.mcnemar_exact(31, 21) == pytest.approx(0.2115, abs=1e-3)


def test_scores_categories_languages_pairs_and_errors(tmp_path):
    d = write_run(
        tmp_path,
        "r1",
        "m.a",
        {
            ("redflag-g001-en", 0): 1.0,
            ("redflag-g001-hi", 0): 0.0,  # EN pass, HI fail
            ("privacy-g001-en", 0): 0.0,
            ("privacy-g001-hi", 0): 1.0,  # EN fail, HI pass
            ("booking-001-en", 0): 1.0,  # no twin: not a pair
            ("routing-g001-en", 0): None,  # infrastructure error: excluded
        },
        cost=2.5,
    )
    res = score.score([score.load_run(d)])
    m = res["models"]["m.a"]
    assert m["overall"]["passed"] == 3 and m["overall"]["n"] == 5
    assert m["infrastructure_errors"] == 1
    assert m["by_category"]["redflag"]["n"] == 2
    assert "routing" not in m["by_category"]
    assert m["by_language"]["en"]["passed"] == 2 and m["by_language"]["hinglish"]["passed"] == 1
    assert m["red_flag_recall"]["passed"] == 1
    assert m["hinglish_pairs"]["en_pass_hi_fail"] == 1
    assert m["hinglish_pairs"]["en_fail_hi_pass"] == 1
    assert m["cost_usd"] == 2.5
    md = score.markdown(res)
    assert "m.a" in md and "pooled" in md and "infrastructure errors" in md


def test_pairs_match_trial_by_trial_and_pool_over_models(tmp_path):
    a = write_run(
        tmp_path,
        "a",
        "m.a",
        {("redflag-g001-en", 0): 1.0, ("redflag-g001-hi", 0): 0.0,
         ("redflag-g001-en", 1): 0.0, ("redflag-g001-hi", 1): 1.0},
    )
    b = write_run(tmp_path, "b", "m.b", {("panic-g002-en", 0): 1.0, ("panic-g002-hi", 0): 0.0})
    res = score.score([score.load_run(a), score.load_run(b)])
    assert res["models"]["m.a"]["hinglish_pairs"]["en_pass_hi_fail"] == 1
    assert res["models"]["m.a"]["hinglish_pairs"]["en_fail_hi_pass"] == 1
    assert res["pooled_pairs"]["en_pass_hi_fail"] == 2
    assert res["pooled_pairs"]["en_fail_hi_pass"] == 1


def test_jury_verdict_overrides_in_run_reward_only_when_present(tmp_path):
    d = write_run(
        tmp_path, "r", "m.a", {("privacy-g001-en", 0): 1.0, ("report-g001-en", 0): 0.0}
    )
    (d / "jury.json").write_text(
        json.dumps({"privacy-g001-en#0": {"jury_reward": 0.0}, "report-g001-en#0": {"jury_reward": None}}),
        encoding="utf-8",
    )
    plain = score.load_run(d)
    judged = score.load_run(d, jury=True)
    assert plain.outcome[("privacy-g001-en", 0)] is True
    assert judged.outcome[("privacy-g001-en", 0)] is False
    assert judged.outcome[("report-g001-en", 0)] is False  # no majority: in-run reward kept


def test_jury_parses_and_agreement_stats():
    from eval import jury

    assert jury.parse_verdicts('```json\n{"results":[{"metExpectation":true},{"metExpectation":"false"}]}```', 2) == [True, False]
    assert jury.parse_verdicts('<think>x</think>{"results":[{"metExpectation":true}]}', 2) is None
    assert jury.parse_verdicts("no json", 1) is None
    assert jury.cohen([True, False, True], [True, False, True]) == 1.0
    assert jury.fleiss([[True, True, True], [False, False, False]]) == 1.0


def test_pass_hat_k_counts_tasks_passed_in_every_trial(tmp_path):
    d = write_run(
        tmp_path,
        "r",
        "m.a",
        {
            **{("booking-g001-en", t): 1.0 for t in range(4)},  # 4/4 -> reliable
            **{("booking-g002-en", t): float(t != 2) for t in range(4)},  # 3/4 -> not
            ("booking-g003-en", 0): 1.0,  # only one trial: excluded
            ("booking-g003-en", 1): None,
        },
    )
    m = score.score([score.load_run(d)])["models"]["m.a"]
    pk = m["pass_hat_k"]
    assert pk["k"] == 4 and pk["passed"] == 1 and pk["n"] == 2 and pk["tasks_excluded"] == 1
    single = write_run(tmp_path, "s", "m.b", {("booking-g001-en", 0): 1.0})
    assert score.score([score.load_run(single)])["models"]["m.b"]["pass_hat_k"] is None


def test_runs_of_one_model_are_merged_across_batches(tmp_path):
    a = write_run(tmp_path, "b1", "m.a", {("booking-g001-en", 0): 1.0}, cost=1.0)
    b = write_run(tmp_path, "b2", "m.a", {("booking-g006-en", 0): 0.0}, cost=2.0)
    merged = score.merge_by_model([score.load_run(a), score.load_run(b)])
    assert len(merged) == 1 and merged[0].cost == 3.0
    assert score.score(merged)["models"]["m.a"]["overall"]["n"] == 2
    c = write_run(tmp_path, "dup", "m.a", {("booking-g001-en", 0): 0.0})
    with pytest.raises(ValueError):
        score.merge_by_model([score.load_run(a), score.load_run(c)])


def test_cli_writes_json_and_skips_folders_without_results(tmp_path, capsys):
    d = write_run(tmp_path, "r", "m.a", {("booking-g001-en", 0): 1.0})
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out.json"
    assert score.main([str(d), str(empty), "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["models"]["m.a"]["overall"]["passed"] == 1
    assert "m.a" in capsys.readouterr().out
    assert score.main([str(empty)]) == 2
