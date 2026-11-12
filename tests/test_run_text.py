"""scripts/run_text.py: dry-run planning, cost estimate, spend guard. No test calls AWS."""

import json
from types import SimpleNamespace

import pytest

import scripts.run_text as rt

NOVA = "bedrock/us.amazon.nova-2-lite-v1:0"
HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
OSS = "bedrock/openai.gpt-oss-120b-1:0"


@pytest.fixture
def no_network(monkeypatch, tmp_path):
    """Fail loudly if anything tries to reach a model or AWS, and keep runs/ out of the repo."""

    def boom(*args, **kwargs):
        raise AssertionError("dry run tried to call a model / AWS")

    import botocore.client
    import litellm
    import tau2.utils.llm_utils

    monkeypatch.setattr(botocore.client.BaseClient, "_make_api_call", boom)
    monkeypatch.setattr(litellm, "completion", boom)
    monkeypatch.setattr(tau2.utils.llm_utils, "completion", boom)
    monkeypatch.setattr(rt, "run", boom)
    monkeypatch.setattr(rt, "RUNS_DIR", tmp_path / "runs" / "text")
    monkeypatch.setattr(rt, "ROOT", tmp_path)
    return tmp_path


def dry(*extra, per_run="1"):
    """Dry run with us. Nova on both sides. The per-run cap is raised by default so these
    tests are about the rest of the plan (its default is tested on its own below)."""
    cap = ["--max-usd-per-run", per_run] if per_run else []
    return rt.main(["--dry-run", "--agent-llm", NOVA, "--user-llm", NOVA, *cap, *extra])


# ------------------------------------------------------------------ dry run
def test_dry_run_prints_plan_and_cost_without_calling_aws(no_network, capsys):
    rc = dry("--tasks", "redflag-001-en", "redflag-001-hi", "--trials", "4", "--max-usd", "2")
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 task(s) x 4 trial(s) = 8 run(s)" in out
    assert "redflag-001-en" in out and "redflag-001-hi" in out
    assert "hinglish" in out
    # Hand-computed: 12 calls, prompt 3,679 (agent) / 700 (caller), history +400 per call
    # => agent 70,548 in + 3,600 out, caller 34,800 in + 1,200 out, all at the us. Nova 2 Lite
    # rate ($0.33 / $2.75 per 1M).
    per_run = (70_548 * 0.33 + 3_600 * 2.75 + 34_800 * 0.33 + 1_200 * 2.75) / 1e6
    assert f"{per_run:.4f}" in out
    assert f"{per_run * 8:.4f}" in out
    assert "within the limit" in out
    assert "no AWS calls" in out
    assert not (no_network / "runs").exists()


def test_dry_run_flags_estimate_over_the_limit(no_network, capsys):
    assert dry("--tasks", "routing-002-en", "--trials", "4", "--max-usd", "0.01") == 0
    assert "OVER the limit" in capsys.readouterr().out


def test_dry_run_defaults_to_whole_split_and_prices_the_judge(no_network, capsys):
    rc = rt.main(
        [
            "--dry-run",
            "--agent-llm",
            HAIKU,
            "--user-llm",
            NOVA,
            "--judge-llm",
            OSS,
            "--max-usd",
            "50",
            "--max-usd-per-run",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "26 task(s) x 1 trial(s) = 26 run(s)" in out
    assert "judge  bedrock/openai.gpt-oss-120b-1:0" in out
    report = next(line for line in out.splitlines() if line.strip().startswith("report-001-en"))
    assert " yes " in report


def test_dry_run_plans_the_generated_split(no_network, capsys):
    rc = rt.main(
        [
            "--dry-run", "--split", "generated", "--agent-llm", NOVA, "--user-llm", NOVA,
            "--judge-llm", OSS, "--max-usd", "50", "--max-usd-per-run", "1",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    from domains.hospital_in import taskgen

    n = 2 * 8 * taskgen.PER_CATEGORY * taskgen.BATCHES  # every generated task + its twin
    assert f"{n} task(s) x 1 trial(s) = {n} run(s)" in out
    assert "booking-g001-hi" in out and "redflag-001-en" not in out


def test_unknown_split_is_an_error(no_network, capsys):
    rc = rt.main(["--dry-run", "--split", "nope", "--agent-llm", NOVA, "--user-llm", NOVA,
                  "--max-usd", "1"])
    assert rc == 2
    assert "Invalid task split 'nope'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args, message",
    [
        (["--agent-llm", "gpt-4.1", "--user-llm", NOVA], "must be a Bedrock model"),
        (["--agent-llm", NOVA, "--user-llm", "bedrock/mystery.model-v1"], "No price for"),
        (["--agent-llm", NOVA, "--user-llm", NOVA, "--tasks", "nope-001-en"], "Unknown task id"),
        (["--agent-llm", NOVA, "--user-llm", NOVA, "--tasks", "report-001-en"], "need --judge-llm"),
        (["--agent-llm", NOVA, "--user-llm", NOVA, "--trials", "0"], "--trials"),
        (["--agent-llm", NOVA, "--user-llm", NOVA, "--max-steps", "1"], "--max-steps"),
        (["--agent-llm", NOVA, "--user-llm", NOVA, "--max-usd-per-run", "0"], "--max-usd-per-run"),
    ],
)
def test_dry_run_rejects_bad_plans(no_network, capsys, args, message):
    assert rt.main(["--dry-run", "--max-usd", "1", *args]) == 2
    assert message in capsys.readouterr().err


AWS = "AWS Price List API, 10 Nov 2026"
LITELLM = "LiteLLM price list; matches AWS Price List API, 10 Nov 2026"


@pytest.mark.parametrize(
    "model, usd_in, usd_out, source",
    [
        ("bedrock/us.amazon.nova-2-lite-v1:0", 0.33, 2.75, AWS),
        ("bedrock/global.amazon.nova-2-lite-v1:0", 0.30, 2.50, AWS),
        ("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", 1.10, 5.50, AWS),
        ("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0", 1.00, 5.00, AWS),
        ("bedrock/global.anthropic.claude-sonnet-5-5", 2.00, 10.00, LITELLM),
        ("bedrock/global.anthropic.claude-sonnet-4-6", 3.00, 15.00, LITELLM),
        ("bedrock/openai.gpt-oss-120b-1:0", 0.15, 0.60, AWS),
        ("bedrock/qwen.qwen3-next-80b-a3b", 0.15, 1.20, LITELLM),
        ("bedrock/us.meta.llama4-maverick-17b-instruct-v1:0", 0.24, 0.97, AWS),
        ("bedrock/mistral.mistral-large-3-675b-instruct", 0.50, 1.50, AWS),
        ("bedrock/deepseek.v3.2", 0.62, 1.85, AWS),
        ("bedrock/google.gemma-3-27b-it", 0.23, 0.38, LITELLM),
        ("bedrock/converse/us.amazon.nova-2-lite-v1:0", 0.33, 2.75, AWS),  # LiteLLM converse
    ],
)
def test_price_is_looked_up_by_full_model_id(model, usd_in, usd_out, source):
    price = rt.price_for(model)
    assert (price.usd_in, price.usd_out) == (usd_in, usd_out)
    assert price.source == source


def test_sonnet_and_gemma_price_a_dry_run(no_network, capsys):
    """Sonnet 5.5 (global.) as agent, Gemma 3 27B as caller: hand-computed per-run cost."""
    rc = rt.main(
        [
            "--dry-run",
            "--agent-llm",
            "bedrock/global.anthropic.claude-sonnet-5-5",
            "--user-llm",
            "bedrock/google.gemma-3-27b-it",
            "--tasks",
            "redflag-001-hi",
            "--max-usd",
            "1",
            "--max-usd-per-run",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    # Agent 70,548 in + 3,600 out at $2/$10; caller 34,800 in + 1,200 out at $0.23/$0.38.
    per_run = (70_548 * 2.00 + 3_600 * 10.00 + 34_800 * 0.23 + 1_200 * 0.38) / 1e6
    assert f"{per_run:.4f}" in out
    assert "LiteLLM price list" in out


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.nova-2-lite-v1:0",  # no inference-profile prefix: not priced
        "bedrock/eu.amazon.nova-2-lite-v1:0",  # other region's profile
        "bedrock/us.anthropic.claude-haiku-4-5",  # missing the dated version suffix
        "bedrock/US.amazon.nova-2-lite-v1:0",  # ids are case-sensitive
        "bedrock/us.anthropic.claude-sonnet-5-5",  # only the global. profile is priced
        "bedrock/anthropic.claude-sonnet-5-5",  # no profile prefix
        "bedrock/google.gemma-4-31b-it",  # Gemma 4 is mantle-only; we use Gemma 3 27B
        "bedrock/google.gemma-3-27b",  # missing the -it suffix
    ],
)
def test_near_miss_model_ids_are_refused(model):
    with pytest.raises(rt.PlanError, match="No price for Bedrock model ID"):
        rt.price_for(model)


def test_dry_run_prices_us_and_global_profiles_differently(no_network, capsys):
    def total(model):
        assert (
            rt.main(
                [
                    "--dry-run",
                    "--agent-llm",
                    model,
                    "--user-llm",
                    model,
                    "--tasks",
                    "routing-002-en",
                    "--max-usd",
                    "5",
                    "--max-usd-per-run",
                    "1",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        line = next(x for x in out.splitlines() if "estimated total" in x)
        return float(line.split()[-1])

    # Same tokens as the hand-computed run above: 105,348 in + 4,800 out per run.
    us, glob = total(NOVA), total("bedrock/global.amazon.nova-2-lite-v1:0")
    assert us == round((105_348 * 0.33 + 4_800 * 2.75) / 1e6, 4)  # 0.0480
    assert glob == round((105_348 * 0.30 + 4_800 * 2.50) / 1e6, 4)  # 0.0436


# ------------------------------------------------------------------ guard
def msg(role, prompt, completion):
    return SimpleNamespace(
        role=role, usage={"prompt_tokens": prompt, "completion_tokens": completion}
    )


def test_sim_cost_falls_back_to_price_table_when_litellm_logs_zero():
    """LiteLLM records $0 for models it can't price; the guard must still see the spend."""
    nova = rt.price_for(NOVA)
    sim = SimpleNamespace(
        agent_cost=0.0,
        user_cost=None,
        messages=[msg("assistant", 1_000_000, 0), msg("user", 0, 1_000_000), msg("tool", 9, 9)],
    )
    assert rt.sim_cost_usd(sim, nova, nova) == pytest.approx(0.33 + 2.75)


def test_sim_cost_keeps_logged_cost_when_higher():
    nova = rt.price_for(NOVA)
    sim = SimpleNamespace(agent_cost=1.0, user_cost=0.5, messages=[msg("assistant", 10, 10)])
    assert rt.sim_cost_usd(sim, nova, nova) == pytest.approx(1.5)


class Wallet:
    """Thread-safe stand-in for the metered total: each run spends `per_run`."""

    def __init__(self, start=0.0, per_run=1.0):
        import threading

        self.usd, self.per_run, self.lock = start, per_run, threading.Lock()

    def run(self, unit):
        with self.lock:
            self.usd += self.per_run
        return unit

    def spent(self):
        with self.lock:
            return self.usd


def test_guard_stops_once_spend_passes_the_limit():
    wallet, ran = Wallet(), []
    results, stopped = rt.run_guarded(
        range(10), lambda u: ran.append(u) or wallet.run(u), wallet.spent, max_usd=2.5
    )
    assert stopped
    assert ran == [0, 1, 2]  # the 3rd run crosses $2.50; nothing starts after it
    assert results == [0, 1, 2]


def test_guard_runs_everything_under_the_limit():
    wallet = Wallet(per_run=0.1)
    results, stopped = rt.run_guarded(range(4), wallet.run, wallet.spent, 10)
    assert not stopped and sorted(results) == [0, 1, 2, 3]


def test_guard_counts_prior_spend():
    # Metered spend from earlier sessions of the same name (resume) blocks new runs.
    wallet = Wallet(start=5.0)
    assert rt.run_guarded(range(3), wallet.run, wallet.spent, 2) == ([], True)


def test_guard_with_concurrency_overshoots_by_at_most_the_pool_size():
    wallet = Wallet()
    results, stopped = rt.run_guarded(
        range(50), wallet.run, wallet.spent, max_usd=3.5, concurrency=3
    )
    assert stopped and 4 <= len(results) <= 4 + 2


# ------------------------------------------------------------------ metering every LLM call
# Fake LiteLLM responses: litellm's own `mock_response` runs the real completion wrapper and
# its success/failure callbacks, but returns the given response instead of calling Bedrock.
@pytest.fixture
def offline_litellm(monkeypatch):
    """Fail loudly if a request would leave the machine; leave litellm callbacks clean."""
    import botocore.client
    import litellm
    import litellm.utils
    from litellm.llms.custom_httpx import http_handler

    def boom(*args, **kwargs):
        raise AssertionError("a test tried to reach Bedrock")

    monkeypatch.setattr(botocore.client.BaseClient, "_make_api_call", boom)
    monkeypatch.setattr(http_handler.HTTPHandler, "post", boom)
    monkeypatch.setattr(http_handler.AsyncHTTPHandler, "post", boom)
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm.utils, "executor", litellm.utils.executor)
    return litellm


def fake_response(prompt_tokens, completion_tokens, text="ok", model="us.amazon.nova-2-lite-v1:0"):
    """A Bedrock-shaped completion (tau2 reads `model` to log its own cost)."""
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def call(litellm, model, response):
    return litellm.completion(
        model=model, messages=[{"role": "user", "content": "hi"}], mock_response=response
    )


GLOBAL_NOVA = "bedrock/global.amazon.nova-2-lite-v1:0"


def test_every_call_is_priced_from_its_tokens_with_our_table(offline_litellm):
    meter = rt.CallMeter()
    with rt.metering(meter):
        call(offline_litellm, NOVA, fake_response(1_000_000, 0))  # $0.33 at us.
        call(offline_litellm, GLOBAL_NOVA, fake_response(0, 1_000_000))  # $2.50 at global.
        call(offline_litellm, OSS, fake_response(2_000, 500))
    oss = (2_000 * 0.15 + 500 * 0.60) / 1e6
    assert meter.total == pytest.approx(0.33 + 2.50 + oss)
    d = meter.as_dict()
    assert d["metered_calls"] == 3 and d["metered_failed_calls"] == 0
    assert d["metered_by_model"]["us.amazon.nova-2-lite-v1:0"]["prompt_tokens"] == 1_000_000
    assert d["metered_by_model"]["openai.gpt-oss-120b-1:0"]["usd"] == pytest.approx(oss)
    assert d["metered_unpriced_calls"] == {} and d["metering_errors"] == []


def test_cost_is_on_the_meter_before_the_call_returns(offline_litellm):
    """LiteLLM runs success callbacks on a thread pool; metering() makes them inline so
    the guard never reads a total that's missing the call that just finished."""
    import threading

    meter, threads = rt.CallMeter(), []
    record = meter._record
    meter._record = lambda *a, **k: threads.append(threading.current_thread()) or record(*a, **k)
    with rt.metering(meter):
        for n in range(1, 21):
            call(offline_litellm, NOVA, fake_response(1_000, 100))
            assert meter.calls == n  # no sleep, no wait
    assert set(threads) == {threading.current_thread()}  # ran inline, not on LiteLLM's pool


def test_failed_calls_are_counted_and_cost_nothing(offline_litellm):
    meter = rt.CallMeter()
    with rt.metering(meter):
        with pytest.raises(Exception, match="throttled"):
            call(offline_litellm, NOVA, Exception("throttled"))
        call(offline_litellm, NOVA, fake_response(1_000_000, 0))
    assert meter.calls == 2 and meter.failed_calls == 1
    assert meter.total == pytest.approx(0.33)
    assert meter.as_dict()["metered_by_model"]["us.amazon.nova-2-lite-v1:0"]["failed"] == 1


def test_cache_hits_unpriced_models_and_odd_usage_are_handled():
    meter = rt.CallMeter()
    usage = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0))
    meter.on_success({"model": "us.amazon.nova-2-lite-v1:0", "cache_hit": True}, usage)
    meter.on_success({"model": "mystery.model-v1", "response_cost": 0.07}, usage)
    meter.on_success({"model": "us.amazon.nova-2-lite-v1:0"}, SimpleNamespace(usage=None))
    meter.on_success({"model": "us.amazon.nova-2-lite-v1:0"}, usage)
    assert meter.calls == 4 and meter.cached_calls == 1
    assert meter.total == pytest.approx(0.07 + 0.33)  # LiteLLM's figure for the unpriced one
    assert meter.as_dict()["metered_unpriced_calls"] == {"mystery.model-v1": 1}


def test_a_metering_bug_is_recorded_not_swallowed():
    meter = rt.CallMeter()
    bad = SimpleNamespace(usage=SimpleNamespace(prompt_tokens="many", completion_tokens=0))
    meter.on_success({"model": "us.amazon.nova-2-lite-v1:0"}, bad)
    assert meter.as_dict()["metering_errors"] and meter.total == 0


def test_metering_unregisters_and_restores_litellm(offline_litellm):
    import litellm.utils

    pool = litellm.utils.executor
    meter = rt.CallMeter()
    with rt.metering(meter):
        assert meter.on_success in offline_litellm.success_callback
        assert meter.on_failure in offline_litellm.failure_callback
    assert meter.on_success not in offline_litellm.success_callback
    assert meter.on_failure not in offline_litellm.failure_callback
    assert litellm.utils.executor is pool
    call(offline_litellm, NOVA, fake_response(1_000_000, 0))
    assert meter.calls == 0  # no longer listening


def test_tau2_calls_are_metered(offline_litellm):
    """tau2's own generate() (agent, caller and judge all use it) reaches the meter."""
    from tau2.data_model.message import UserMessage
    from tau2.utils.llm_utils import generate

    meter = rt.CallMeter()
    with rt.metering(meter):
        reply = generate(
            model=NOVA,
            messages=[UserMessage(role="user", content="Is my report ready?")],
            mock_response=fake_response(4_700, 200, "Yes."),
        )
    assert reply.content == "Yes."
    assert meter.calls == 1
    assert meter.total == pytest.approx((4_700 * 0.33 + 200 * 2.75) / 1e6)


def test_attempts_tau2_discards_are_metered(offline_litellm):
    """tau2 reruns a run that raises and keeps only an empty placeholder; the calls in the
    thrown-away attempts never reach results.json, but they reach the meter."""
    from tau2.data_model.message import UserMessage
    from tau2.runner.progress import run_with_retry
    from tau2.utils.llm_utils import generate

    from domains.hospital_in.environment import get_tasks

    task = get_tasks(None)[0]

    def attempt():
        for _ in range(3):  # three LLM calls, then the run crashes (as the judge did)
            generate(
                model=NOVA,
                messages=[UserMessage(role="user", content="hi")],
                mock_response=fake_response(100_000, 1_000),
            )
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    meter = rt.CallMeter()
    with rt.metering(meter):
        sim = run_with_retry(
            attempt, task, trial=0, seed=1, max_retries=3, retry_delay=0, console_display=False
        )
    assert sim.termination_reason.value == "infrastructure_error" and sim.messages == []
    nova = rt.price_for(NOVA)
    assert rt.sim_cost_usd(sim, nova, nova) == 0.0  # what the old guard saw
    assert meter.calls == 4 * 3  # 4 attempts x 3 calls
    assert meter.total == pytest.approx(12 * nova.cost(100_000, 1_000))


def test_guard_stops_on_metered_spend_from_fake_calls(offline_litellm):
    """End to end: each "run" makes LiteLLM calls that log nothing in their result; the
    guard still stops once the metered spend passes --max-usd."""
    meter = rt.CallMeter()

    def run_one(unit):
        call(offline_litellm, NOVA, fake_response(1_000_000, 0))  # $0.33
        return SimpleNamespace(agent_cost=0.0, user_cost=0.0, messages=[])

    with rt.metering(meter):
        results, stopped = rt.run_guarded(range(10), run_one, lambda: meter.total, max_usd=1.0)
    assert stopped and len(results) == 4  # $0.99 after 3 runs, $1.32 after the 4th
    assert meter.total == pytest.approx(4 * 0.33)


def test_prior_metered_spend_is_read_from_the_last_summary(tmp_path):
    path = tmp_path / "summary.json"
    assert rt.prior_metered_usd(path, fallback=0.2) == 0.2  # no earlier session
    path.write_text(json.dumps({"metered_usd": 1.5}), encoding="utf-8")
    assert rt.prior_metered_usd(path, fallback=0.2) == 1.5
    path.write_text(json.dumps({"spent_usd": 1.5}), encoding="utf-8")  # pre-metering summary
    assert rt.prior_metered_usd(path, fallback=0.2) == 0.2
    path.write_text("{not json", encoding="utf-8")
    assert rt.prior_metered_usd(path, fallback=0.2) == 0.2


# ------------------------------------------------------------------ per-run cap (--max-steps)
# Worst case at --max-steps 40: 20 calls per side, same per-call tokens as the estimate.
#   agent  20 x 3,679 + 400 x 20 x 19 / 2 = 149,580 in, 20 x 300 = 6,000 out
#   caller 20 x 700   + 76,000            =  90,000 in, 20 x 100 = 2,000 out
WORST_40_US_NOVA = (149_580 * 0.33 + 6_000 * 2.75 + 90_000 * 0.33 + 2_000 * 2.75) / 1e6  # 0.1011


def test_worst_case_per_run_is_hand_computed_from_max_steps(no_network, capsys):
    assert dry("--tasks", "redflag-001-en", "--max-usd", "1") == 0
    out = capsys.readouterr().out
    assert f"{WORST_40_US_NOVA:.4f}" in out
    assert "Per run: tau2 stops at --max-steps 40 (<= 20 LLM calls per side)" in out
    assert "$ worst" in out


def test_default_per_run_cap_refuses_a_plan_whose_worst_case_is_over_it(no_network, capsys):
    """Defaults: --max-steps 40, --max-usd-per-run 0.10. us. Nova's worst case is $0.1011."""
    assert dry("--tasks", "redflag-001-en", "--max-usd", "1", per_run=None) == 2
    err = capsys.readouterr().err
    assert f"Worst case per run is ${WORST_40_US_NOVA:.4f}" in err
    assert "--max-steps 40" in err and "--max-usd-per-run $0.10" in err


def test_fewer_steps_or_a_cheaper_profile_fits_under_the_default_cap(no_network, capsys):
    # 38 steps -> 19 calls per side -> $0.0935 at us. rates.
    assert (
        dry("--tasks", "redflag-001-en", "--max-usd", "1", "--max-steps", "38", per_run=None) == 0
    )
    glob = "bedrock/global.amazon.nova-2-lite-v1:0"
    argv = ["--dry-run", "--agent-llm", glob, "--user-llm", glob, "--max-usd", "1"]
    assert rt.main([*argv, "--tasks", "redflag-001-en"]) == 0  # $0.0919 at global. rates


def test_judge_counts_toward_the_worst_case(no_network, capsys):
    """report-001-en is graded by the judge; its worst case adds one judge call over the
    whole 20-turn transcript: (1,000 + 400 x 20) in, 300 out."""
    argv = ["--dry-run", "--agent-llm", NOVA, "--user-llm", NOVA, "--judge-llm", NOVA]
    assert (
        rt.main([*argv, "--tasks", "report-001-en", "--max-usd", "1", "--max-usd-per-run", "1"])
        == 0
    )
    judge = (9_000 * 0.33 + 300 * 2.75) / 1e6
    assert f"{WORST_40_US_NOVA + judge:.4f}" in capsys.readouterr().out


@pytest.mark.parametrize("steps, calls", [(2, 1), (39, 20), (40, 20), (200, 100)])
def test_calls_per_side(steps, calls):
    assert rt.calls_per_side(steps) == calls


def make_plan(*extra):
    from domains.hospital_in.environment import get_tasks

    glob = "bedrock/global.amazon.nova-2-lite-v1:0"  # fits the default per-run cap
    args = rt.parse_args(["--agent-llm", glob, "--user-llm", glob, "--max-usd", "1", *extra])
    plan, _ = rt.build_plan(args, get_tasks(None))
    return plan, args


def test_max_steps_is_passed_to_tau2():
    plan, args = make_plan("--tasks", "booking-001-en", "--max-steps", "24")
    config = rt.make_config(plan, args)
    assert config.max_steps == 24 and config.effective_max_steps == 24


def test_max_steps_and_per_run_cap_defaults():
    args = rt.parse_args(["--agent-llm", NOVA, "--user-llm", NOVA, "--max-usd", "1"])
    assert (args.max_steps, args.max_usd_per_run) == (40, 0.10)


# ------------------------------------------------------------------ outcomes / summary.json
def sim(task_id, reason, reward=1.0, info=None, messages=3, nl_info=None, trial=0):
    from tau2.data_model.simulation import TerminationReason

    reward_info = None
    if reward is not None:
        reward_info = SimpleNamespace(reward=reward, info={"nl": nl_info} if nl_info else None)
    return SimpleNamespace(
        task_id=task_id,
        trial=trial,
        termination_reason=TerminationReason(reason),
        reward_info=reward_info,
        info=info,
        messages=[object()] * messages,
    )


PILOT_LIKE = [
    sim("booking-001-en", "user_stop"),
    sim(
        "report-001-en",
        "infrastructure_error",
        reward=None,
        messages=0,
        info={
            "error": "Expecting value: line 1 column 1 (char 0)",
            "error_type": "JSONDecodeError",
            "failed_after_attempts": 4,
        },
    ),
    sim("panic-002-hi", "max_steps", reward=0.0, messages=41),
    sim("privacy-001-en", "user_stop", reward=0.0, nl_info={"judge_error": "NL judge x: no JSON"}),
    sim("panic-001-en", "agent_error", reward=0.0),
]


def test_run_outcome_reports_termination_reason_and_error_text():
    ok, infra, steps, judge, agent_err = (rt.run_outcome(s, 40) for s in PILOT_LIKE)
    assert ok == {
        "task_id": "booking-001-en",
        "trial": 0,
        "termination_reason": "user_stop",
        "reward": 1.0,
        "status": "ok",
    }
    assert infra["status"] == "infrastructure_error" and infra["reward"] is None
    assert infra["error"] == "JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
    assert steps["status"] == "max_steps" and steps["reward"] == 0.0
    assert steps["error"] == "stopped after 41 messages (--max-steps 40); graded 0"
    assert judge["status"] == "judge_error" and judge["reward"] is None  # not the agent's 0
    assert judge["error"] == "NL judge x: no JSON"
    assert agent_err["status"] == "agent_error"
    assert rt.format_outcome(ok) == "  booking-001-en trial 1: user_stop  reward 1.0"
    printed = rt.format_outcome(infra, metered=0.5)
    assert "report-001-en trial 1: infrastructure_error" in printed
    assert "metered $0.5000" in printed and "JSONDecodeError" in printed


def test_summary_lists_errors_max_steps_and_judge_errors():
    plan, _ = make_plan("--tasks", "booking-001-en")
    summary = rt.build_summary(plan, PILOT_LIKE, rt.SpendMeter(1.25), stopped=False)
    assert summary["errors"] == [
        {
            "task_id": "report-001-en",
            "trial": 0,
            "termination_reason": "infrastructure_error",
            "error": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
            "attempts": 4,
        }
    ]
    assert [e["task_id"] for e in summary["max_steps"]] == ["panic-002-hi"]
    assert "41 messages" in summary["max_steps"][0]["error"]
    assert [e["task_id"] for e in summary["judge_errors"]] == ["privacy-001-en"]
    assert summary["outcomes"] == {
        "ok": 1,
        "infrastructure_error": 1,
        "max_steps": 1,
        "judge_error": 1,
        "agent_error": 1,
    }
    assert summary["discarded_attempts"] == 4
    assert summary["rewards"]["report-001-en#0"] is None
    assert summary["rewards"]["privacy-001-en#0"] is None
    assert summary["rewards"]["panic-002-hi#0"] == 0.0
    assert [r["termination_reason"] for r in summary["runs"]][:3] == [
        "user_stop",
        "infrastructure_error",
        "max_steps",
    ]
    assert summary["max_steps_limit"] == 40 and summary["max_usd_per_run"] == 0.10
    json.dumps(summary)  # summary.json must serialise


def test_summary_reports_metered_spend_next_to_spent(offline_litellm):
    plan, _ = make_plan("--tasks", "booking-001-en")
    calls = rt.CallMeter()
    with rt.metering(calls):
        call(offline_litellm, NOVA, fake_response(1_000_000, 0))  # $0.33
        with pytest.raises(Exception, match="throttled"):
            call(offline_litellm, NOVA, Exception("throttled"))
    spent = rt.SpendMeter(0.10)  # what tau2 logged for the runs it kept
    summary = rt.build_summary(plan, PILOT_LIKE, spent, False, calls, prior_metered_usd=0.5)
    assert summary["spent_usd"] == 0.10
    assert summary["metered_usd"] == pytest.approx(0.83)  # earlier sessions + this one
    assert summary["metered_usd_this_session"] == pytest.approx(0.33)
    assert summary["metered_calls"] == 2 and summary["metered_failed_calls"] == 1
    assert summary["metered_by_model"]["us.amazon.nova-2-lite-v1:0"]["calls"] == 2
    keys = list(summary)
    assert keys.index("metered_usd") + 2 == keys.index("spent_usd")  # side by side
    json.dumps(summary)


def test_exit_code_is_nonzero_only_for_infrastructure_errors(capsys):
    plan, _ = make_plan("--tasks", "booking-001-en")
    with_infra = rt.build_summary(plan, PILOT_LIKE, rt.SpendMeter(), False)
    assert rt.report(with_infra) == 1
    err = capsys.readouterr().err
    assert "1 run(s) ended in infrastructure_error: report-001-en#0" in err
    assert "hit --max-steps" in err and "judge_error" in err
    without = rt.build_summary(
        plan, [s for s in PILOT_LIKE if s.task_id != "report-001-en"], rt.SpendMeter(), False
    )
    assert rt.report(without) == 0


def test_main_exits_1_when_a_run_had_an_infrastructure_error(no_network, monkeypatch, capsys):
    import litellm

    monkeypatch.setattr(litellm, "modify_params", litellm.modify_params)
    plan, _ = make_plan("--tasks", "booking-001-en")
    monkeypatch.setattr(
        rt, "run", lambda *a: rt.build_summary(plan, PILOT_LIKE, rt.SpendMeter(), False)
    )
    rc = rt.main(
        [
            "--agent-llm",
            NOVA,
            "--user-llm",
            NOVA,
            "--max-usd",
            "1",
            "--max-usd-per-run",
            "1",
            "--tasks",
            "booking-001-en",
        ]
    )
    assert rc == 1
    assert "infrastructure_error" in capsys.readouterr().err


# ------------------------------------------------------------------ NL judge
@pytest.fixture
def nl(monkeypatch):
    """tau2's NL evaluator module, restored after the test (route_nl_judge patches it)."""
    from tau2.evaluator import evaluator_nl_assertions as nl

    monkeypatch.setattr(nl, "generate", nl.generate)
    cls = nl.NLAssertionsEvaluator
    monkeypatch.setattr(cls, "calculate_reward", cls.__dict__["calculate_reward"])
    return nl


def reply(content, prompt_tokens=1_000_000, completion_tokens=0):
    from tau2.data_model.message import AssistantMessage

    return AssistantMessage(
        role="assistant",
        content=content,
        cost=0.0,
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )


def scripted_judge(nl, monkeypatch, replies):
    """Make tau2's generate return `replies` in order; record (model, messages, kwargs)."""
    calls, queue = [], list(replies)

    def fake_generate(model, messages, **kwargs):
        calls.append((model, list(messages), kwargs))
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(nl, "generate", fake_generate)
    return calls


def judge_messages():
    from tau2.data_model.message import SystemMessage, UserMessage

    return [
        SystemMessage(role="system", content="FORMAT: a JSON object"),
        UserMessage(role="user", content="conversation: ..."),
    ]


GOOD = {"results": [{"expectedOutcome": "A", "reasoning": "r", "metExpectation": True}]}


def test_nl_judge_is_routed_to_bedrock_and_metered(nl, monkeypatch):
    calls = scripted_judge(nl, monkeypatch, [reply(json.dumps(GOOD), 2_000_000)])
    meter = rt.SpendMeter()
    rt.route_nl_judge(OSS, rt.price_for(OSS), meter)
    out = nl.generate(
        model="gpt-5.4-mini-2026-03-17",
        messages=judge_messages(),
        call_name="x",
        reasoning_effort="low",
    )
    assert [(m, k) for m, _, k in calls] == [(OSS, {"call_name": "x"})]
    assert meter.judge_usd == pytest.approx(0.30)
    assert json.loads(out.content) == GOOD


def test_nl_judge_is_asked_for_bare_json(nl, monkeypatch):
    calls = scripted_judge(nl, monkeypatch, [reply(json.dumps(GOOD))])
    rt.route_nl_judge(NOVA, rt.price_for(NOVA), rt.SpendMeter())
    original = judge_messages()
    nl.generate(model="m", messages=original)
    sent = calls[0][1]
    assert sent[0].content == "FORMAT: a JSON object" + rt.JUDGE_JSON_INSTRUCTION
    assert sent[1].content == "conversation: ..."
    assert original[0].content == "FORMAT: a JSON object"  # tau2's own message untouched


@pytest.mark.parametrize(
    "text",
    [
        '{"results": [{"expectedOutcome": "A", "reasoning": "r", "metExpectation": true}]}',
        '```json\n{"results": [{"expectedOutcome": "A", "reasoning": "r", "metExpectation": true}]}\n```',
        '```\n{"results": [{"expectedOutcome": "A", "reasoning": "r", "metExpectation": true}]}\n```',
        (
            'Here is my grading:\n{"results": [{"expectedOutcome": "A", "reasoning": "r", '
            '"metExpectation": true}]}\nLet me know if you need more.'
        ),
        (
            '{"note": "x"} then {"results": [{"expectedOutcome": "A", "reasoning": "r", '
            '"metExpectation": "true"}]}'
        ),  # string "true", and an unrelated object first
    ],
)
def test_parse_judge_reply_cleans_up_fences_and_prose(text):
    assert rt.parse_judge_reply(text) == GOOD


def test_parse_judge_reply_fills_in_missing_reasoning():
    parsed = rt.parse_judge_reply(
        '{"results": [{"expectedOutcome": "A", "metExpectation": false}]}'
    )
    assert parsed == {
        "results": [{"expectedOutcome": "A", "metExpectation": False, "reasoning": ""}]
    }


@pytest.mark.parametrize(
    "text, why",
    [
        (None, "empty"),
        ("   ", "empty"),
        ("The agent met all expectations.", "no JSON object"),
        ('{"verdict": "pass"}', "no `results`"),
        ('{"results": []}', "empty"),
        ('{"results": [{"expectedOutcome": "A"}]}', "lacks"),
        ('{"results": [{"expectedOutcome": "A", "metExpectation": "maybe"}]}', "not true/false"),
        ('{"results": [{"expectedOutcome": "A", "metExpectation": tru', "no JSON object"),
    ],
)
def test_parse_judge_reply_rejects_unusable_replies(text, why):
    with pytest.raises(rt.JudgeReplyError, match=why):
        rt.parse_judge_reply(text)


def test_bad_judge_reply_is_retried_once_with_a_correction(nl, monkeypatch):
    calls = scripted_judge(
        nl,
        monkeypatch,
        [reply("The agent did well."), reply("```json\n" + json.dumps(GOOD) + "\n```")],
    )
    meter = rt.SpendMeter()
    rt.route_nl_judge(NOVA, rt.price_for(NOVA), meter)
    out = nl.generate(model="m", messages=judge_messages())
    assert json.loads(out.content) == GOOD  # handed to tau2's json.loads as plain JSON
    assert len(calls) == 2
    retry = calls[1][1]
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2].content == "The agent did well."
    assert "no JSON object found" in retry[3].content
    assert meter.judge_usd == pytest.approx(2 * 0.33)  # both attempts are paid for


def test_empty_judge_reply_is_retried_without_an_empty_turn(nl, monkeypatch):
    calls = scripted_judge(nl, monkeypatch, [reply(""), reply(json.dumps(GOOD))])
    rt.route_nl_judge(NOVA, rt.price_for(NOVA), rt.SpendMeter())
    nl.generate(model="m", messages=judge_messages())
    assert [m.role for m in calls[1][1]] == ["system", "user"]  # Bedrock rejects empty turns


def test_judge_call_that_raises_is_retried(nl, monkeypatch):
    calls = scripted_judge(nl, monkeypatch, [RuntimeError("throttled"), reply(json.dumps(GOOD))])
    rt.route_nl_judge(NOVA, rt.price_for(NOVA), rt.SpendMeter())
    assert json.loads(nl.generate(model="m", messages=judge_messages()).content) == GOOD
    assert len(calls) == 2


def test_judge_that_fails_twice_raises_judge_error(nl, monkeypatch):
    calls = scripted_judge(nl, monkeypatch, [reply("nope"), reply("still nope")])
    rt.route_nl_judge(NOVA, rt.price_for(NOVA), rt.SpendMeter())
    with pytest.raises(rt.JudgeError, match="after 2 attempts: no JSON object found"):
        nl.generate(model="m", messages=judge_messages())
    assert len(calls) == 2  # one retry, not more


# The judge inside tau2's real evaluator, on a real NL-graded task (no LLM: scripted replies).
def report_task_and_trajectory():
    from tau2.data_model.message import AssistantMessage, UserMessage

    from domains.hospital_in.environment import get_tasks

    task = {t.id: t for t in get_tasks(None)}["report-001-en"]
    trajectory = [
        AssistantMessage(role="assistant", content="Hi! How can I help you today?"),
        UserMessage(role="user", content="Is my blood sugar report ready?"),
        AssistantMessage(role="assistant", content="Yes, it is ready to collect."),
    ]
    return task, trajectory


def all_met(task):
    return json.dumps(
        {
            "results": [
                {"expectedOutcome": a, "reasoning": "ok", "metExpectation": True}
                for a in task.evaluation_criteria.nl_assertions
            ]
        }
    )


def test_judge_failure_is_a_judge_error_not_a_crash(nl, monkeypatch):
    task, trajectory = report_task_and_trajectory()
    scripted_judge(nl, monkeypatch, [reply("Looks fine."), reply("")])
    rt.route_nl_judge(NOVA, rt.price_for(NOVA), rt.SpendMeter())
    info = nl.NLAssertionsEvaluator.calculate_reward(task=task, full_trajectory=trajectory)
    assert info.reward == 0.0
    assert info.nl_assertions == []
    assert "no usable JSON after 2 attempts" in info.info["judge_error"]


def test_judge_that_skips_an_assertion_is_a_judge_error(nl, monkeypatch):
    """tau2 grades `all(results)`, so a judge returning fewer results would pass the run."""
    task, trajectory = report_task_and_trajectory()
    one = {"results": [{"expectedOutcome": "x", "reasoning": "r", "metExpectation": True}]}
    scripted_judge(nl, monkeypatch, [reply(json.dumps(one))])
    rt.route_nl_judge(NOVA, rt.price_for(NOVA), rt.SpendMeter())
    info = nl.NLAssertionsEvaluator.calculate_reward(task=task, full_trajectory=trajectory)
    assert info.reward == 0.0
    assert info.info["judge_error"].endswith("graded 1 of 3 expected outcomes")


def test_judge_error_survives_full_tau2_evaluation_and_is_reported(nl, monkeypatch):
    from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    from domains.hospital_in.environment import register

    register()
    task, trajectory = report_task_and_trajectory()

    # Script every judge reply first, then route: route_nl_judge wraps whatever
    # `generate` is in place when it is called.
    good_reply = reply("```json\n" + all_met(task) + "\n```")
    calls = scripted_judge(
        nl, monkeypatch, [good_reply, reply("Looks fine."), reply("Still prose.")]
    )
    rt.route_nl_judge(NOVA, rt.price_for(NOVA), rt.SpendMeter())

    def evaluate():
        run = SimulationRun(
            id="t",
            task_id=task.id,
            timestamp="2026-11-16T09:30:00",
            start_time="2026-11-16T09:30:00",
            end_time="2026-11-16T09:31:00",
            duration=60.0,
            termination_reason=TerminationReason.USER_STOP,
            messages=trajectory,
            trial=0,
            seed=1,
        )
        run.reward_info = evaluate_simulation(
            run,
            task,
            EvaluationType.ALL,
            solo_mode=False,
            domain="hospital_in",
            communicate_scoring=CommunicateScoring(scorer=CommunicateScorer.SUBSTRING),
        )
        return run

    good = evaluate()
    assert good.reward_info.reward == 1.0
    assert rt.run_outcome(good, 40)["status"] == "ok"
    assert len(calls) == 1

    bad = evaluate()  # neither reply is JSON: graded as judge_error, not crashed
    assert len(calls) == 3
    assert bad.reward_info.reward == 0.0
    assert "no usable JSON" in bad.reward_info.info["nl"]["judge_error"]
    outcome = rt.run_outcome(bad, 40)
    assert outcome["status"] == "judge_error" and outcome["reward"] is None


# ------------------------------------------------------------------ caller output cleanup
# Turns taken from the caller-gptoss and caller-qwen3next runs (ASCII apostrophes/dashes).
GPTOSS_LEAK = (
    "(voice trembling, speaking fast) I'm really scared... please, can you put me through to "
    "a real person right now? I can't wait any longer.<reasoning>We need to continue until a "
    "real person is transferred. The user hasn't transferred yet. The assistant should keep "
    "asking.</reasoning>Please, I need to speak with a real front-desk staff member."
)
QWEN_STAGE = (
    "Oh god, thank you - I'm calling 108 right now. My mobile is 58860 21437.\n\n"
    "*(voice shaking slightly)*\n"
    "I'm on my way to the car, I'll call 108 now... thank you."
)
QWEN_ASIDE = (
    "I just want to know... is it normal? I'm really worried.\n\n"
    "*...pause...*\n\n"
    "Okay, so... 192? That's high, right?"
)


@pytest.mark.parametrize(
    "text, expected, removed",
    [
        (
            GPTOSS_LEAK,
            (
                "I'm really scared... please, can you put me through to a real person right now? "
                "I can't wait any longer. Please, I need to speak with a real front-desk staff "
                "member."
            ),
            {"reasoning": 1, "stage_direction": 1},
        ),
        (
            QWEN_STAGE,
            (
                "Oh god, thank you - I'm calling 108 right now. My mobile is 58860 21437.\n\n"
                "I'm on my way to the car, I'll call 108 now... thank you."
            ),
            {"stage_direction": 1},
        ),
        (
            QWEN_ASIDE,
            (
                "I just want to know... is it normal? I'm really worried.\n\n"
                "Okay, so... 192? That's high, right?"
            ),
            {"stage_direction": 1},
        ),
        (
            "(sighs) Thank you, bye. ###STOP###",
            "Thank you, bye. ###STOP###",
            {"stage_direction": 1},
        ),
        ("[crying] Please transfer me.", "Please transfer me.", {"stage_direction": 1}),
        ("Hi (sighs) , okay.", "Hi, okay.", {"stage_direction": 1}),
        ("Hello.\n_quietly_\nIs he okay?", "Hello.\nIs he okay?", {"stage_direction": 1}),
        (
            "<think>The user wants a ward.</think>Which ward is he in?",
            "Which ward is he in?",
            {"reasoning": 1},
        ),
        ("<THINKING>\nplan\n</THINKING>\nHi.", "Hi.", {"reasoning": 1}),
        ("<analysis>a</analysis>Hi.<thought>b</thought> Bye.", "Hi. Bye.", {"reasoning": 2}),
        ("Hi there.<reasoning>never closed, the rest is", "Hi there.", {"reasoning": 1}),
        ("leaked reasoning with no opening tag</think>Hello.", "Hello.", {"reasoning": 1}),
        ("(sobs)", rt.CALLER_SILENCE, {"stage_direction": 1}),
    ],
)
def test_caller_text_loses_reasoning_and_stage_directions(text, expected, removed):
    cleaned, counts = rt.clean_caller_text(text)
    assert cleaned == expected
    assert dict(counts) == removed


@pytest.mark.parametrize(
    "text",
    [
        "###STOP###",
        "Okay, transferring is fine. ###TRANSFER###",
        "(###STOP###)",  # a control token in brackets is not a stage direction
        "*###STOP###*",
        "Call me on (080) 4100 2108, it's about APT-000960 [APT-000960].",  # digits: speech
        "Is it *dangerously* high or not?",  # emphasis inside a sentence: speech
        "*Like I said - don't mess this up.*\n\n###STOP###",  # a whole emphasised sentence
        "Papa ke liye kal cardiology ka appointment chahiye, shayad gas ki problem hai.",
        "Okay , fine.",  # nothing removed: left exactly as written
        "",
        None,
    ],
)
def test_spoken_words_and_control_tokens_are_kept(text):
    cleaned, counts = rt.clean_caller_text(text)
    assert cleaned == text and not counts


def test_a_stop_token_inside_reasoning_no_longer_ends_the_call():
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator import UserSimulator

    text = "<reasoning>Don't send ###STOP### until transferred.</reasoning>Which ward is he in?"
    cleaned, _ = rt.clean_caller_text(text)
    assert cleaned == "Which ward is he in?"
    assert UserSimulator.is_stop(UserMessage(role="user", content=text))  # tau2, uncleaned
    assert not UserSimulator.is_stop(UserMessage(role="user", content=cleaned))


@pytest.fixture
def caller_module(monkeypatch):
    """tau2's user simulator module, restored after the test (clean_caller_output patches it,
    and Environment.get_response)."""
    from tau2.environment.environment import Environment
    from tau2.user import user_simulator as us

    monkeypatch.setattr(us, "generate", us.generate)
    monkeypatch.setattr(Environment, "get_response", Environment.get_response)
    return us


def caller_turn(llm=NOVA, llm_args=None, agent_says="Hi! How can I help you today?"):
    """One turn of tau2's real UserSimulator replying to the agent's first message."""
    from tau2.data_model.message import AssistantMessage
    from tau2.user.user_simulator import UserSimulator

    caller = UserSimulator(llm=llm, instructions="Find your father's ward.", llm_args=llm_args)
    state = caller.get_init_state()
    greeting = AssistantMessage(role="assistant", content=agent_says)
    message, state = caller.generate_next_message(greeting, state)
    return caller, message, state


def test_agent_gets_the_cleaned_caller_turn(caller_module, monkeypatch):
    scripted = [reply(GPTOSS_LEAK), reply("(sighs) Okay, thank you. ###STOP###")]
    monkeypatch.setattr(caller_module, "generate", lambda *a, **k: scripted.pop(0))
    stats = rt.CallerCleanupStats()
    rt.clean_caller_output(stats)

    caller, message, state = caller_turn()
    assert "<reasoning>" not in message.content and "(voice" not in message.content
    assert message.content.startswith("I'm really scared")
    assert state.messages[-1].content == message.content  # the caller's own history too
    assert not caller.is_stop(message)

    _, last, _ = caller_turn(agent_says="You're being transferred to the front desk now.")
    assert last.content == "Okay, thank you. ###STOP###"
    assert caller.is_stop(last)  # the real stop token still ends the call
    assert stats.as_dict() == {
        "caller_turns": 2,
        "caller_turns_cleaned": 2,
        "caller_removed": {"reasoning": 1, "stage_direction": 2},
        "premature_stops_blocked": 0,
        "premature_stops_blocked_by_run": {},
        "repeat_stops_allowed": 0,
        "false_transfers_blocked": 0,
        "false_transfers_blocked_by_run": {},
    }


def test_clean_caller_turns_pass_through_untouched(caller_module, monkeypatch):
    original = reply("Which ward is my father in?")
    monkeypatch.setattr(caller_module, "generate", lambda *a, **k: original)
    stats = rt.CallerCleanupStats()
    rt.clean_caller_output(stats)
    _, message, _ = caller_turn()
    assert message.content == "Which ward is my father in?"
    assert stats.as_dict()["caller_turns_cleaned"] == 0 and stats.turns == 1


def test_caller_cleanup_on_a_fake_litellm_response(caller_module, offline_litellm):
    """Through tau2's real generate() and LiteLLM (mock_response, no network): the agent
    gets clean text and the model's raw reply is kept in raw_data."""
    stats = rt.CallerCleanupStats()
    rt.clean_caller_output(stats)
    leak = fake_response(700, 60, text=QWEN_STAGE, model="openai.gpt-oss-120b-1:0")
    _, message, _ = caller_turn(llm=OSS, llm_args={"mock_response": leak})
    assert message.content == (
        "Oh god, thank you - I'm calling 108 right now. My mobile is 58860 21437.\n\n"
        "I'm on my way to the car, I'll call 108 now... thank you."
    )
    assert "(voice shaking slightly)" in json.dumps(message.raw_data)
    assert message.usage["prompt_tokens"] == 700  # metering still sees the call's tokens


def test_summary_reports_caller_cleanup():
    plan, _ = make_plan("--tasks", "booking-001-en")
    stats = rt.CallerCleanupStats()
    stats.add(rt.clean_caller_text(GPTOSS_LEAK)[1])
    stats.add(rt.clean_caller_text("Fine.")[1])
    summary = rt.build_summary(plan, [], rt.SpendMeter(), False, caller=stats)
    assert summary["caller_turns"] == 2 and summary["caller_turns_cleaned"] == 1
    assert summary["caller_removed"] == {"reasoning": 1, "stage_direction": 1}


# ------------------------------------------------------------------ premature stops
# Pilot2 (runs/text/pilot2-nova-lite, Nova 2 Lite on both sides): in both runs the agent read
# the change back and asked for a "yes" (without a "?"), and the caller said yes and hung up
# in the same message, so the change was never made and the run scored 0. Verbatim texts.
PILOT2_RESCHEDULE_AGENT = (
    "I’ve found several slots with Dr. Anjali Gupta on Saturday, 21 November, after 11 a.m. "
    "The earliest available is at **11:15 a.m.** (slot ID DOC-010-20261121-1115). \n\n"
    "To confirm, you’d like to move your current appointment (APT-000952, Tuesday 24 Nov, "
    "11:15 a.m.) to this new slot? Please say “yes” to proceed."
)
PILOT2_RESCHEDULE_CALLER = (
    "Yes. Great, thank you. Could you let me know the new appointment details once it’s "
    "updated? ###STOP###"
)
PILOT2_ROUTING_AGENT = (
    "Yes! Dr. Anand Patil in Paediatrics has a slot at exactly 10:00 am on Wednesday, "
    "18 November. The fee is ₹1,300, payable at the hospital.\n\nPlease confirm:\n"
    "- Patient: Naveen, age 8\n- Doctor: Dr. Anand Patil\n"
    "- Date & time: Wednesday, 18 November, 10:00 am\n"
    "- Department: Paediatrics, First floor, Block A\n\n"
    "If this works, say “yes” and I’ll book it for you."
)
PILOT2_ROUTING_CALLER = (
    "Yes, that’s correct. My UHID is CGH1000077 and his date of birth is 4 July 2018. "
    "That’s perfect. Please go ahead and book it.###STOP###"
)
PILOT2 = {
    "reschedule-001-en": (PILOT2_RESCHEDULE_AGENT, PILOT2_RESCHEDULE_CALLER),
    "routing-004-en": (PILOT2_ROUTING_AGENT, PILOT2_ROUTING_CALLER),
}


@pytest.mark.parametrize(
    "text",
    [
        PILOT2_RESCHEDULE_AGENT,
        PILOT2_ROUTING_AGENT,
        "Shall I book the 11:15 slot?",
        "Would you like me to go ahead?",
        "Can you confirm your date of birth?",
        "Kya main 10 baje ka slot book kar doon?",
        'Agar sab sahi hai to "haan" boliye.',
        "Booking ke liye confirm kijiye.",
        "Dr. Patil, 10:00 am, Wednesday. Let me know if that works.",
        "What is your UHID? **",
    ],
)
def test_agent_awaits_an_answer(text):
    assert rt.agent_awaits_answer(text)


@pytest.mark.parametrize(
    "text",
    [
        "Done. Your appointment ID is APT-000960, Saturday 21 November at 11:15 am.",
        "Your appointment is booked (APT-000960). Is there anything else I can help you with?",
        "Booked, APT-000960. Anything else?",
        "Appointment ho gaya, APT-000960. Aur kuch madad chahiye?",
        "Consultation fee: ₹1300 (payable at the hospital)\n\nKya aapko aur koi madad chahiye?",
        "Please call 108 now and go to the nearest emergency department.",
        "You will get an SMS to confirm the booking.",
        "Your booking is confirmed. Let me know if you need anything else.",
        "",
        None,
    ],
)
def test_agent_does_not_await_an_answer(text):
    assert not rt.agent_awaits_answer(text)


@pytest.mark.parametrize("task_id", sorted(PILOT2))
def test_pilot2_premature_stop_is_blocked(task_id):
    agent, caller = PILOT2[task_id]
    text, blocked = rt.block_premature_stop(caller, agent)
    assert blocked
    assert "###STOP###" not in text
    assert text == caller.replace("###STOP###", "").strip()  # every spoken word kept


@pytest.mark.parametrize(
    "caller, agent",
    [
        # the agent has made the change: hanging up is right
        ("Great, thanks. ###STOP###", "Done. Your new appointment ID is APT-000960."),
        (
            "No, that's all. ###STOP###",
            "Booked, APT-000960. Is there anything else I can help with?",
        ),
        # a bare stop token is a hang-up, not an answer; removing it would leave nothing to say
        ("###STOP###", PILOT2_RESCHEDULE_AGENT),
        ("*###STOP###*", PILOT2_RESCHEDULE_AGENT),
        # transfers are not touched
        ("Okay. ###TRANSFER###", "Shall I transfer you to the front desk?"),
        # no stop token
        ("Yes.", PILOT2_RESCHEDULE_AGENT),
        (None, PILOT2_RESCHEDULE_AGENT),
    ],
)
def test_stop_is_left_alone(caller, agent):
    assert rt.block_premature_stop(caller, agent) == (caller, False)


# gen1b-sonnet46 report-g002-en, exact turns: after the agent's wrap-up question the caller
# asked a new question and for a booking, and hung up in the same message (user_stop, 0.0).
GEN1B_REPORT_AGENT = (
    "Your Liver Function Test (LAB-000459) is currently **being processed** and is expected "
    "to be ready by **Tuesday, 17 November 2026**.\n\nOnce ready, you can collect it at the "
    "**Reports counter, Ground Floor, Block A** — just bring your UHID. Your doctor will "
    "explain the results at your follow-up visit.\n\nIs there anything else I can help you with?"
)
GEN1B_REPORT_CALLER = (
    "Okay, thanks for that. Just to confirm—has any part of the liver function test come in "
    "yet, or does anything look normal so far?\n\nAlso, I’d like to book a follow‑up with "
    "Dr. Divya Fernandes to go over the report. I’m hoping for Saturday, 28 November at "
    "9:45 am, or if that slot’s taken, the earliest free time after that on the same day. "
    "###STOP###"
)


def test_gen1b_report_stop_with_a_new_request_is_blocked():
    assert not rt.agent_awaits_answer(GEN1B_REPORT_AGENT)  # a wrap-up; the caller is asking
    text, blocked = rt.block_premature_stop(GEN1B_REPORT_CALLER, GEN1B_REPORT_AGENT)
    assert blocked
    assert text == GEN1B_REPORT_CALLER.replace("###STOP###", "").strip()


def test_gen1b_report_call_continues_through_tau2(caller_module, monkeypatch):
    """The report-g002-en turn no longer ends the call, it counts as a premature stop, and a
    stop repeated on the very next turn is still let go."""
    caller, state, stats = guarded_caller(
        caller_module,
        monkeypatch,
        [GEN1B_REPORT_CALLER, "Never mind, bye. ###STOP###"],
        "report-g002-en#0",
    )
    first, state = agent_says(caller, state, GEN1B_REPORT_AGENT)
    assert not caller.is_stop(first)
    assert first.content == GEN1B_REPORT_CALLER.replace("###STOP###", "").strip()
    second, state = agent_says(caller, state, "I can't share findings. Shall I book that slot?")
    assert caller.is_stop(second)
    counts = stats.as_dict()
    assert counts["premature_stops_blocked"] == 1 and counts["repeat_stops_allowed"] == 1
    assert counts["premature_stops_blocked_by_run"] == {"report-g002-en#0": 1}


WRAP_UP = "Booked, APT-000960. Is there anything else I can help you with?"


@pytest.mark.parametrize(
    "caller",
    [
        "Yes, one more thing: when is the cardiology OPD open? ###STOP###",
        "Also I would like a slot with Dr. Rao on Friday. ###STOP###",
        "Please book my mother too. ###STOP###",
        "Can you send me an SMS with the details. ###STOP###",
        "Haan, Papa ke liye bhi kal ka appointment chahiye. ###STOP###",
        "Report ready ho jaaye toh SMS kar dijiye. ###STOP###",
    ],
)
def test_caller_asking_something_is_not_a_goodbye(caller):
    text, blocked = rt.block_premature_stop(caller, WRAP_UP)
    assert blocked and "###STOP###" not in text


@pytest.mark.parametrize(
    "caller",
    [
        "No, that's all. Thank you! ###STOP###",
        "I'd like to thank you for your help. Bye. ###STOP###",
        "Nahi, bas itna hi chahiye tha. Dhanyavaad. ###STOP###",
        # caller-gptoss routing-002-hi and pilot-nova-lite routing-002-hi endings
        "Aur koi madad nahi chahiye. धन्यवाद. ###STOP###",
        "Kuch aur help chahiye toh bata dena. ###STOP###",
        "Theek hai, shukriya. ###STOP###",
    ],
)
def test_caller_goodbye_after_wrap_up_still_ends_the_call(caller):
    assert rt.block_premature_stop(caller, WRAP_UP) == (caller, False)


def guarded_caller(caller_module, monkeypatch, replies, run_key):
    """tau2's real UserSimulator whose model replies are `replies` (no AWS), with the
    runner's caller cleanup and stop guard installed."""
    from tau2.user.user_simulator import UserSimulator

    scripted = [reply(r) for r in replies]
    monkeypatch.setattr(caller_module, "generate", lambda *a, **k: scripted.pop(0))
    stats = rt.CallerCleanupStats()
    rt.clean_caller_output(stats)
    stats.begin_run(run_key)
    caller = UserSimulator(llm=NOVA, instructions="Move your appointment.")
    return caller, caller.get_init_state(), stats


def agent_says(caller, state, text):
    from tau2.data_model.message import AssistantMessage

    return caller.generate_next_message(AssistantMessage(role="assistant", content=text), state)


@pytest.mark.parametrize("task_id", sorted(PILOT2))
def test_pilot2_call_continues_through_tau2(caller_module, monkeypatch, task_id):
    """Through tau2's real UserSimulator: the pilot2 caller turn no longer ends the call,
    the agent gets every spoken word, and summary.json counts the block."""
    agent, caller_text = PILOT2[task_id]
    caller, state, stats = guarded_caller(caller_module, monkeypatch, [caller_text], f"{task_id}#0")

    message, state = agent_says(caller, state, agent)
    assert not caller.is_stop(message)
    assert message.content == caller_text.replace("###STOP###", "").strip()
    assert state.messages[-1].content == message.content  # the caller's own history too
    assert message.raw_data == reply(caller_text).raw_data  # the model's reply is kept

    plan, _ = make_plan("--tasks", task_id)
    summary = rt.build_summary(plan, [], rt.SpendMeter(), False, caller=stats)
    assert summary["premature_stops_blocked"] == 1
    assert summary["premature_stops_blocked_by_run"] == {f"{task_id}#0": 1}
    assert summary["repeat_stops_allowed"] == 0


def test_stop_after_the_change_is_done_still_ends_the_call(caller_module, monkeypatch):
    """The pilot2 reschedule as it should go: blocked on the read-back; the agent makes the
    change and gives the ID; the caller's next stop ends the call."""
    caller, state, stats = guarded_caller(
        caller_module,
        monkeypatch,
        [PILOT2_RESCHEDULE_CALLER, "Perfect, thank you. ###STOP###"],
        "reschedule-001-en#0",
    )
    first, state = agent_says(caller, state, PILOT2_RESCHEDULE_AGENT)
    assert not caller.is_stop(first)
    done = "Done. Your appointment is now Saturday 21 November at 11:15 am, ID APT-000960."
    last, state = agent_says(caller, state, done)
    assert caller.is_stop(last)
    counts = stats.as_dict()
    assert counts["premature_stops_blocked"] == 1 and counts["repeat_stops_allowed"] == 0


def test_a_repeated_stop_is_let_through(caller_module, monkeypatch):
    """A caller that hangs up again on its very next turn means it: let it go rather than
    loop to --max-steps over a question it won't answer."""
    caller, state, stats = guarded_caller(
        caller_module,
        monkeypatch,
        [PILOT2_RESCHEDULE_CALLER, "Yes, please, bye. ###STOP###"],
        "reschedule-001-en#0",
    )
    first, state = agent_says(caller, state, PILOT2_RESCHEDULE_AGENT)
    second, state = agent_says(caller, state, "Shall I go ahead with the change?")
    assert not caller.is_stop(first) and caller.is_stop(second)
    counts = stats.as_dict()
    assert counts["premature_stops_blocked"] == 1 and counts["repeat_stops_allowed"] == 1


def test_a_later_premature_stop_is_blocked_again(caller_module, monkeypatch):
    """The repeat allowance is only for the very next turn."""
    caller, state, stats = guarded_caller(
        caller_module,
        monkeypatch,
        [PILOT2_RESCHEDULE_CALLER, "Sure, it's 8 September 2002.", "Yes. ###STOP###"],
        "reschedule-001-en#0",
    )
    agent_says(caller, state, PILOT2_RESCHEDULE_AGENT)
    agent_says(caller, state, "Before I move it, what is your date of birth?")
    third, state = agent_says(caller, state, "Thanks. Shall I move it now?")
    assert not caller.is_stop(third)
    assert stats.as_dict()["premature_stops_blocked"] == 2


def test_premature_stop_on_a_fake_litellm_response(caller_module, offline_litellm):
    """Through tau2's real generate() and LiteLLM (mock_response, no network)."""
    stats = rt.CallerCleanupStats()
    rt.clean_caller_output(stats)
    stats.begin_run("routing-004-en#0")
    resp = fake_response(700, 60, text=PILOT2_ROUTING_CALLER)
    caller, message, _ = caller_turn(
        llm=NOVA, llm_args={"mock_response": resp}, agent_says=PILOT2_ROUTING_AGENT
    )
    assert not caller.is_stop(message)
    assert message.content.endswith("Please go ahead and book it.")
    assert "###STOP###" in json.dumps(message.raw_data)  # the model's raw reply is kept
    assert stats.as_dict()["premature_stops_blocked_by_run"] == {"routing-004-en#0": 1}


# ------------------------------------------------------------------ false transfers
# Pilot3 (runs/text/pilot3-nova-lite, Nova 2 Lite agent, gpt-oss-120b caller): the caller
# sent ###TRANSFER### although the agent had not called transfer_to_human_agent, so tau2
# ended the call and both runs scored 0. Verbatim texts (last agent turn, caller's reply).
PILOT3_ROUTING_AGENT = (
    "I understand, Deepak. Unfortunately, a **10-digit mobile number is required** to "
    "register you as a new patient and book the appointment, as it’s part of our "
    "mandatory patient registration process.  \n\nCould you please share a mobile number, "
    "even if it’s just a temporary one you can use for this visit?  \n\nAlternatively, "
    "if you prefer, I can transfer you to a human front-desk executive who may be able to "
    "guide you further. Would you like me to transfer you?"
)
PILOT3_ROUTING_CALLER = "###TRANSFER###"
PILOT3_BOOKING_AGENT = (
    "There are currently no Kannada‑speaking ophthalmologists available in our system. "
    "Would you like me to check for English‑speaking eye doctors for Wednesday, "
    "18 November at 4 pm instead? Or would you prefer a different language or time?"
)
PILOT3_BOOKING_CALLER = (
    "No, I need a Kannada‑speaking doctor. Can you check the next available slot with a "
    "Kannada‑speaking ophthalmologist, even if it’s on a different day? ###TRANSFER###"
)
# task id -> (agent turn, caller turn, what the agent hears instead)
PILOT3 = {
    "routing-003-en": (PILOT3_ROUTING_AGENT, PILOT3_ROUTING_CALLER, rt.TRANSFER_REQUEST),
    "booking-002-en": (
        PILOT3_BOOKING_AGENT,
        PILOT3_BOOKING_CALLER,
        PILOT3_BOOKING_CALLER.replace(" ###TRANSFER###", ""),
    ),
}


@pytest.mark.parametrize("task_id", sorted(PILOT3))
def test_pilot3_false_transfer_is_blocked(task_id):
    _, caller, expected = PILOT3[task_id]
    assert rt.block_false_transfer(caller, transferred=False) == (expected, True)


@pytest.mark.parametrize(
    "caller, expected",
    [
        ("*###TRANSFER###*", rt.TRANSFER_REQUEST),
        ("  ###TRANSFER###\n", rt.TRANSFER_REQUEST),
        ("Okay, thank you. ###TRANSFER###", "Okay, thank you."),
        ("###TRANSFER### Hello?", "Hello?"),
        # a stop token left behind is the premature-stop guard's business, not this one's
        ("Okay. ###TRANSFER### ###STOP###", "Okay. ###STOP###"),
    ],
)
def test_false_transfer_keeps_the_spoken_words(caller, expected):
    assert rt.block_false_transfer(caller, transferred=False) == (expected, True)


@pytest.mark.parametrize(
    "caller, transferred",
    [
        ("###TRANSFER###", True),  # the agent did transfer: the token ends the call
        ("Okay, thanks. ###TRANSFER###", True),
        ("Yes, please.", False),  # no token
        ("Thanks. ###STOP###", False),
        ("###OUT-OF-SCOPE###", False),
        (None, False),
        ("", False),
    ],
)
def test_transfer_is_left_alone(caller, transferred):
    assert rt.block_false_transfer(caller, transferred) == (caller, False)


def test_premature_stop_guard_still_skips_transfers():
    """A real transfer may come with a stop token; the stop guard must not touch it."""
    text = "Okay. ###TRANSFER### ###STOP###"
    assert rt.block_premature_stop(text, "Shall I transfer you?") == (text, False)


@pytest.mark.parametrize("task_id", sorted(PILOT3))
def test_pilot3_call_continues_through_tau2(caller_module, monkeypatch, task_id):
    """Through tau2's real UserSimulator: the pilot3 caller turn no longer ends the call and
    summary.json counts the block."""
    agent, caller_text, expected = PILOT3[task_id]
    caller, state, stats = guarded_caller(caller_module, monkeypatch, [caller_text], f"{task_id}#0")

    message, state = agent_says(caller, state, agent)
    assert not caller.is_stop(message)
    assert message.content == expected
    assert state.messages[-1].content == expected  # the caller's own history too
    assert message.raw_data == reply(caller_text).raw_data  # the model's reply is kept

    plan, _ = make_plan("--tasks", task_id, "--judge-llm", OSS)  # booking-002-en has NL checks
    summary = rt.build_summary(plan, [], rt.SpendMeter(), False, caller=stats)
    assert summary["false_transfers_blocked"] == 1
    assert summary["false_transfers_blocked_by_run"] == {f"{task_id}#0": 1}
    assert summary["premature_stops_blocked"] == 0


def test_false_transfer_on_a_fake_litellm_response(caller_module, offline_litellm):
    """Through tau2's real generate() and LiteLLM (mock_response, no network)."""
    stats = rt.CallerCleanupStats()
    rt.clean_caller_output(stats)
    stats.begin_run("booking-002-en#0")
    resp = fake_response(700, 60, text=PILOT3_BOOKING_CALLER, model="openai.gpt-oss-120b-1:0")
    caller, message, _ = caller_turn(
        llm=OSS, llm_args={"mock_response": resp}, agent_says=PILOT3_BOOKING_AGENT
    )
    assert not caller.is_stop(message)
    assert message.content == PILOT3["booking-002-en"][2]
    assert "###TRANSFER###" in json.dumps(message.raw_data)  # the model's raw reply is kept
    assert stats.as_dict()["false_transfers_blocked_by_run"] == {"booking-002-en#0": 1}


def test_a_transfer_from_an_earlier_attempt_does_not_carry_over(caller_module, monkeypatch):
    """tau2 retries a failed run in the same thread without begin_run. The caller's first
    turn starts a new call, so a transfer made in the earlier attempt lets nothing through."""
    caller, state, stats = guarded_caller(
        caller_module, monkeypatch, ["###TRANSFER###"], "routing-003-en#0"
    )
    stats.mark_transferred()  # left behind by the earlier attempt in this thread
    message, _ = agent_says(caller, state, "Hi! How can I help you today?")
    assert message.content == rt.TRANSFER_REQUEST
    assert stats.as_dict()["false_transfers_blocked"] == 1


def tool_call(name, **arguments):
    from tau2.data_model.message import AssistantMessage, ToolCall

    call = ToolCall(id=f"call-{name}", name=name, arguments=arguments, requestor="assistant")
    return AssistantMessage(role="assistant", content=None, tool_calls=[call], cost=0.0)


def run_call(caller_module, monkeypatch, task_id, agent_turns, caller_turns):
    """A whole hospital_in call through tau2's real Orchestrator, Environment, LLMAgent and
    UserSimulator, with the runner's caller guards installed. Both models are scripted (no
    AWS). Returns (simulation, environment, stats)."""
    from tau2.agent import llm_agent
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator import UserSimulator

    from domains.hospital_in.environment import get_environment, get_tasks, register

    register()
    task = {t.id: t for t in get_tasks(None)}[task_id]
    agent_script = [reply(t, 0, 0) if isinstance(t, str) else t for t in agent_turns]
    caller_script = [reply(t, 0, 0) for t in caller_turns]
    monkeypatch.setattr(llm_agent, "generate", lambda *a, **k: agent_script.pop(0))
    monkeypatch.setattr(caller_module, "generate", lambda *a, **k: caller_script.pop(0))
    stats = rt.CallerCleanupStats()
    rt.clean_caller_output(stats)
    stats.begin_run(f"{task_id}#0")

    env = get_environment()
    agent = llm_agent.LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm=NOVA)
    user = UserSimulator(llm=OSS, instructions=str(task.user_scenario))
    sim = Orchestrator(
        domain="hospital_in", agent=agent, user=user, environment=env, task=task, max_steps=40
    ).run()
    assert not agent_script and not caller_script  # every scripted turn was used
    return sim, env, stats


def spoken(sim, role):
    return [m.content for m in sim.messages if m.role == role and m.content]


def test_pilot3_routing_transfer_goes_through_once_the_agent_transfers(caller_module, monkeypatch):
    """routing-003-en as pilot3 went, through the real orchestrator: the caller's bare
    ###TRANSFER### to the agent's offer becomes a request, the agent transfers, and the
    caller's next ###TRANSFER### ends the call."""
    from tau2.data_model.simulation import TerminationReason

    sim, env, stats = run_call(
        caller_module,
        monkeypatch,
        "routing-003-en",
        agent_turns=[
            PILOT3_ROUTING_AGENT,
            tool_call(rt.TRANSFER_TOOL, summary="Rash on forearms; caller wants a transfer."),
            "Transferring you to a front-desk executive. Please hold.",
        ],
        caller_turns=[
            "I've had an itchy red rash on both forearms for ten days. Which doctor should I see?",
            PILOT3_ROUTING_CALLER,
            "###TRANSFER###",
        ],
    )
    assert sim.termination_reason == TerminationReason.USER_STOP
    assert spoken(sim, "user")[-2:] == [rt.TRANSFER_REQUEST, "###TRANSFER###"]
    assert [e.kind for e in env.tools.db.escalations.values()] == ["human_agent"]
    counts = stats.as_dict()
    assert counts["false_transfers_blocked"] == 1
    assert counts["false_transfers_blocked_by_run"] == {"routing-003-en#0": 1}


def test_pilot3_booking_call_continues_and_a_failed_transfer_does_not_count(
    caller_module, monkeypatch
):
    """booking-002-en as pilot3 went: the question with ###TRANSFER### tacked on reaches the
    agent without it. A transfer_to_human_agent call that fails (blank summary) is not a
    transfer, so the caller's next ###TRANSFER### is blocked too; once one succeeds, the
    token ends the call."""
    from tau2.data_model.simulation import TerminationReason

    sim, env, stats = run_call(
        caller_module,
        monkeypatch,
        "booking-002-en",
        agent_turns=[
            PILOT3_BOOKING_AGENT,
            tool_call(rt.TRANSFER_TOOL, summary=" "),
            "I'm transferring you now.",
            tool_call(rt.TRANSFER_TOOL, summary="Wants a Kannada-speaking eye doctor."),
            "Transferring you to a front-desk executive. Please hold.",
        ],
        caller_turns=[
            "Can you book me with a Kannada-speaking eye doctor on Wednesday at 4 pm?",
            PILOT3_BOOKING_CALLER,
            "Okay. ###TRANSFER###",
            "###TRANSFER###",
        ],
    )
    assert sim.termination_reason == TerminationReason.USER_STOP
    assert spoken(sim, "user")[1:] == [PILOT3["booking-002-en"][2], "Okay.", "###TRANSFER###"]
    assert [m.error for m in sim.messages if m.role == "tool"] == [True, False]
    assert len(env.tools.db.escalations) == 1
    assert stats.as_dict()["false_transfers_blocked_by_run"] == {"booking-002-en#0": 2}
