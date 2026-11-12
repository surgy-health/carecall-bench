"""Run hospital_in in text mode (LLM agent <-> LLM caller) on AWS Bedrock, with a spend guard.

    # See what would run and what it should cost. Makes no AWS calls.
    uv run python scripts/run_text.py --dry-run \
        --agent-llm bedrock/global.amazon.nova-2-lite-v1:0 \
        --user-llm  bedrock/openai.gpt-oss-120b-1:0 \
        --judge-llm bedrock/qwen.qwen3-next-80b-a3b \
        --tasks redflag-001-en redflag-001-hi report-001-en --trials 4 --max-usd 2

    # Same command without --dry-run runs it. Results: runs/text/<name>/results.json
    # (tau2's format, so `tau2 view` works), plus summary.json with spend and rewards.

Models must be Bedrock (`bedrock/<model-id>`), so all spend lands on the AWS bill. Every
model needs a row in PRICES below. That row is used for the dry-run estimate and to
price every call for the guard.

The guard: every LLM call (agent, caller, judge) is priced from its token usage as it
finishes, by a LiteLLM success/failure callback (`CallMeter`). That includes calls in
attempts tau2 throws away and reruns after an exception. Once the metered total passes
--max-usd, no new simulation starts. Simulations already in flight finish, so the
overshoot is at most --concurrency simulations (default 1). Metered spend from earlier
sessions of the same --save-to name counts when resuming. summary.json reports it as
metered_usd, next to spent_usd (what tau2 logged for the runs it kept).

Per run: tau2 stops a conversation after --max-steps messages (default 40; tau2's own
default of 200 let one pilot run loop for $0.58). The plan is refused if any run's worst
case, a conversation that uses every step, could cost more than --max-usd-per-run.

Tasks graded by NL assertions need an LLM judge. tau2 hard-codes an OpenAI judge, so this
script requires --judge-llm bedrock/<id> for those tasks and routes the judge to it. The
judge is asked for bare JSON, its reply is cleaned up before tau2 parses it, and it gets
one retry. A judge that still fails marks the run `judge_error` (reward not counted)
instead of crashing the simulation.

The caller's turns are cleaned before the agent sees them: reasoning blocks
(<reasoning>...</reasoning> and similar) and stage directions such as "(voice shaking)"
are removed, keeping the spoken words and the ###STOP### / ###TRANSFER### tokens.
A caller turn that ends the call (###STOP###) while the agent's last turn asked a question
or for a yes/confirmation has the token removed, so the call goes on (pilot2: callers said
"Yes ###STOP###" to the read-back, before the change was made), and so does one whose own
words ask something ("?", "I'd like", "can you", "chahiye"...; gen1b report-g002-en asked a
new question and for a booking, then hung up). summary.json counts these
as premature_stops_blocked. A caller that ends the call again on its very next turn is let
go (repeat_stops_allowed), so a misread question costs one exchange, not a loop.
A caller ###TRANSFER### sent before the agent's transfer_to_human_agent call has succeeded
in this call is removed too, and the call goes on (pilot3: callers sent it when the agent
only offered a transfer, or tacked it onto an unanswered question). A turn that was only the
token becomes "Please transfer me." summary.json counts these as false_transfers_blocked.

Every run's termination reason is printed and recorded in summary.json, with runs that
ended in infrastructure_error, max_steps or judge_error listed separately. The exit code
is 1 if any run ended in infrastructure_error, 2 for a refused plan, else 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOMAIN = "hospital_in"
RUNS_DIR = ROOT / "runs" / "text"
AWS_REGION = "us-east-1"  # CLAUDE.md: everything runs in us-east-1

# ------------------------------------------------------------------ prices
PRICE_LIST = "AWS Price List API, 10 Nov 2026"
LITELLM_LIST = "LiteLLM price list; matches AWS Price List API, 10 Nov 2026"


@dataclass(frozen=True)
class Price:
    """On-demand USD per 1M tokens for one Bedrock model ID."""

    usd_in: float
    usd_out: float
    source: str

    def cost(self, prompt_tokens: float, completion_tokens: float) -> float:
        return (prompt_tokens * self.usd_in + completion_tokens * self.usd_out) / 1e6


# Keyed by the full Bedrock model ID, inference-profile prefix included: the prefix changes
# the price (us. cross-region costs 10% more than global. for Nova 2 Lite and Haiku 4.5), so
# an ID that isn't listed exactly is refused rather than priced by a near match.
# Gemma: Gemma 4 31B is only on the bedrock_mantle endpoint, not the standard Bedrock
# runtime this script (LiteLLM bedrock/) calls, so the open-weight Google model is
# Gemma 3 27B instead.
# No caching discount: the estimate and the fallback are upper bounds when caching is on.
PRICES: dict[str, Price] = {
    "us.amazon.nova-2-lite-v1:0": Price(0.33, 2.75, PRICE_LIST),
    "global.amazon.nova-2-lite-v1:0": Price(0.30, 2.50, PRICE_LIST),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": Price(1.10, 5.50, PRICE_LIST),
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": Price(1.00, 5.00, PRICE_LIST),
    "global.anthropic.claude-sonnet-5-5": Price(2.00, 10.00, LITELLM_LIST),
    # Sonnet 5.5 / Sonnet 5 / Opus 5.5 return AccessDenied for this account (11 Nov 2026);
    # Sonnet 4.6 is the frontier Claude we can call.
    "global.anthropic.claude-sonnet-4-6": Price(3.00, 15.00, LITELLM_LIST),
    "google.gemma-3-27b-it": Price(0.23, 0.38, LITELLM_LIST),
    "openai.gpt-oss-120b-1:0": Price(0.15, 0.60, PRICE_LIST),
    # Qwen3 235B (qwen.qwen3-235b-a22b-2507-v1:0) is not offered on demand in us-east-1
    # (`aws bedrock list-foundation-models`, 10 Nov 2026); Qwen3-Next 80B is the largest
    # text Qwen there. AWS lists $0.14 in; LiteLLM $0.15 -- keep the higher.
    "qwen.qwen3-next-80b-a3b": Price(0.15, 1.20, LITELLM_LIST),
    # Open-weight candidates to replace Gemma 3 27B, which on Bedrock writes tool calls as
    # JSON text instead of native tool use (pilot3-gemma3: 0 tool calls in 26 runs).
    # IDs checked with `aws bedrock list-foundation-models` (on demand, us-east-1, 11 Nov 2026).
    "mistral.mistral-large-3-675b-instruct": Price(0.50, 1.50, PRICE_LIST),
    "deepseek.v3.2": Price(0.62, 1.85, PRICE_LIST),
    # Llama 4 Maverick is inference-profile only in us-east-1; the bare ID is invalid there.
    "us.meta.llama4-maverick-17b-instruct-v1:0": Price(0.24, 0.97, PRICE_LIST),
}


class PlanError(ValueError):
    """The requested run is invalid; nothing was started."""


def check_bedrock(model: str, flag: str) -> None:
    if not model.startswith("bedrock/") or model == "bedrock/":
        raise PlanError(
            f"{flag} must be a Bedrock model, written bedrock/<model-id>; got {model!r}"
        )


def bedrock_model_id(model: str) -> str:
    """`bedrock/<id>` or LiteLLM's `bedrock/converse/<id>` -> `<id>`."""
    return model.removeprefix("bedrock/").removeprefix("converse/")


def price_for(model: str) -> Price:
    model_id = bedrock_model_id(model)
    if model_id in PRICES:
        return PRICES[model_id]
    raise PlanError(
        f"No price for Bedrock model ID {model_id!r}. Add it, with its inference-profile "
        f"prefix, to PRICES in scripts/run_text.py (known: {', '.join(PRICES)})."
    )


# ------------------------------------------------------------------ estimate
@dataclass(frozen=True)
class Assumptions:
    """Per-run token model. Defaults follow the Bedrock plan doc (Nov 10, 2026)."""

    agent_prompt_tokens: int = 3679  # measured: policy + 14 tool schemas, sent on every call
    turns: int = 12  # LLM calls per side per run
    history_growth_tokens: int = 400  # conversation added per call
    agent_output_tokens: int = 300  # middle of the plan's 150-500
    caller_prompt_tokens: int = 700  # measured ~500-670 (tau2 caller guidelines + task script)
    caller_output_tokens: int = 100
    judge_overhead_tokens: int = 1000  # NL-judge instructions + assertions, plus the transcript
    judge_output_tokens: int = 300


DEFAULT_ASSUMPTIONS = Assumptions()


def _conversation_tokens(prompt: int, a: Assumptions, output: int) -> tuple[int, int]:
    """(input, output) tokens for `a.turns` calls whose history grows each call."""
    history = a.history_growth_tokens * a.turns * (a.turns - 1) // 2
    return a.turns * prompt + history, a.turns * output


@dataclass(frozen=True)
class RunEstimate:
    agent_usd: float
    caller_usd: float
    judge_usd: float

    @property
    def total_usd(self) -> float:
        return self.agent_usd + self.caller_usd + self.judge_usd


def estimate_run(
    agent: Price, caller: Price, judge: Price | None, a: Assumptions = DEFAULT_ASSUMPTIONS
) -> RunEstimate:
    agent_usd = agent.cost(*_conversation_tokens(a.agent_prompt_tokens, a, a.agent_output_tokens))
    caller_usd = caller.cost(
        *_conversation_tokens(a.caller_prompt_tokens, a, a.caller_output_tokens)
    )
    judge_usd = 0.0
    if judge is not None:
        transcript = a.history_growth_tokens * a.turns
        judge_usd = judge.cost(a.judge_overhead_tokens + transcript, a.judge_output_tokens)
    return RunEstimate(agent_usd, caller_usd, judge_usd)


def calls_per_side(max_steps: int) -> int:
    """Most LLM calls one side can make in a run of `max_steps` tau2 steps.

    tau2 counts every message as a step, and the agent and caller take turns (a tool
    result also takes a step), so each side gets at most ceil(steps / 2). The pilot's
    200-step panic-002-hi made 100 agent and 99 caller calls.
    """
    return math.ceil(max_steps / 2)


def worst_case_run(
    agent: Price,
    caller: Price,
    judge: Price | None,
    max_steps: int,
    a: Assumptions = DEFAULT_ASSUMPTIONS,
) -> RunEstimate:
    """A run that uses every step: same per-call tokens, history growing the whole way."""
    return estimate_run(agent, caller, judge, replace(a, turns=calls_per_side(max_steps)))


# ------------------------------------------------------------------ plan
@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    category: str
    lang: str
    needs_judge: bool
    estimate: RunEstimate
    worst: RunEstimate


@dataclass(frozen=True)
class Plan:
    agent_llm: str
    user_llm: str
    judge_llm: str | None
    trials: int
    max_usd: float
    max_steps: int
    max_usd_per_run: float
    concurrency: int
    save_path: Path
    tasks: tuple[PlannedTask, ...]

    @property
    def worst_task(self) -> PlannedTask:
        return max(self.tasks, key=lambda t: t.worst.total_usd)

    @property
    def n_runs(self) -> int:
        return len(self.tasks) * self.trials

    @property
    def total_usd(self) -> float:
        return sum(t.estimate.total_usd for t in self.tasks) * self.trials


def _note(task, key: str) -> str:
    for part in (task.description.notes or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == key:
            return v
    return "?"


def needs_nl_judge(task) -> bool:
    from tau2.data_model.tasks import RewardType

    ec = task.evaluation_criteria
    return bool(ec and ec.reward_basis and RewardType.NL_ASSERTION in ec.reward_basis)


def build_plan(args: argparse.Namespace, all_tasks: Sequence) -> tuple[Plan, list]:
    """Validate the request and price it. Raises PlanError; touches no network."""
    check_bedrock(args.agent_llm, "--agent-llm")
    check_bedrock(args.user_llm, "--user-llm")
    if args.judge_llm:
        check_bedrock(args.judge_llm, "--judge-llm")
    agent_p, user_p = price_for(args.agent_llm), price_for(args.user_llm)
    judge_p = price_for(args.judge_llm) if args.judge_llm else None
    if args.trials < 1:
        raise PlanError("--trials must be at least 1")
    if args.max_usd <= 0:
        raise PlanError("--max-usd must be positive")
    if args.concurrency < 1:
        raise PlanError("--concurrency must be at least 1")
    if args.max_steps < 2:
        raise PlanError("--max-steps must be at least 2 (one caller and one agent turn)")
    if args.max_usd_per_run <= 0:
        raise PlanError("--max-usd-per-run must be positive")

    by_id = {t.id: t for t in all_tasks}
    wanted = args.tasks or list(by_id)
    unknown = [t for t in wanted if t not in by_id]
    if unknown:
        raise PlanError(f"Unknown task id(s): {', '.join(unknown)}. Valid: {', '.join(by_id)}")
    if len(set(wanted)) != len(wanted):
        raise PlanError("--tasks lists a task more than once")
    tasks = [by_id[t] for t in wanted]

    with_comm = [
        t.id for t in tasks if t.evaluation_criteria and t.evaluation_criteria.communicate_info
    ]
    if with_comm:
        raise PlanError(
            "Tasks with communicate_info need tau2's semantic judge, which this script does "
            f"not route to Bedrock or meter yet: {', '.join(with_comm)}"
        )
    nl = [t.id for t in tasks if needs_nl_judge(t)]
    if nl and not args.judge_llm:
        raise PlanError(
            "These tasks are graded with NL assertions and need --judge-llm bedrock/<id> "
            f"(tau2's default judge is an OpenAI model): {', '.join(nl)}"
        )

    planned = tuple(
        PlannedTask(
            task_id=t.id,
            category=_note(t, "category"),
            lang=_note(t, "lang"),
            needs_judge=needs_nl_judge(t),
            estimate=estimate_run(agent_p, user_p, judge_p if needs_nl_judge(t) else None),
            worst=worst_case_run(
                agent_p, user_p, judge_p if needs_nl_judge(t) else None, args.max_steps
            ),
        )
        for t in tasks
    )
    name = args.save_to or default_run_name(args.agent_llm, args.user_llm)
    plan = Plan(
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        judge_llm=args.judge_llm if nl else None,
        trials=args.trials,
        max_usd=args.max_usd,
        max_steps=args.max_steps,
        max_usd_per_run=args.max_usd_per_run,
        concurrency=args.concurrency,
        save_path=RUNS_DIR / name / "results.json",
        tasks=planned,
    )
    worst = plan.worst_task
    if worst.worst.total_usd > plan.max_usd_per_run:
        raise PlanError(
            f"Worst case per run is ${worst.worst.total_usd:.4f} ({worst.task_id}: "
            f"{calls_per_side(plan.max_steps)} calls per side at --max-steps {plan.max_steps}), "
            f"over --max-usd-per-run ${plan.max_usd_per_run:.2f}. "
            "Lower --max-steps or raise --max-usd-per-run."
        )
    return plan, tasks


def default_run_name(agent_llm: str, user_llm: str) -> str:
    def short(m: str) -> str:
        return re.sub(r"[^a-z0-9.-]+", "-", m.removeprefix("bedrock/").lower()).strip("-")

    return f"{datetime.now():%Y%m%d-%H%M%S}_{short(agent_llm)}_vs_{short(user_llm)}"


def format_plan(plan: Plan, a: Assumptions = DEFAULT_ASSUMPTIONS) -> str:
    def price_line(role: str, model: str | None) -> str:
        if model is None:
            return f"  {role:<7}none (no selected task is graded by NL assertions)"
        p = price_for(model)
        return f"  {role:<7}{model}  ${p.usd_in:.2f} in / ${p.usd_out:.2f} out per 1M  [{p.source}]"

    lines = [
        "Planned hospital_in text runs",
        price_line("agent", plan.agent_llm),
        price_line("caller", plan.user_llm),
        price_line("judge", plan.judge_llm),
        (
            f"  {len(plan.tasks)} task(s) x {plan.trials} trial(s) = {plan.n_runs} run(s), "
            f"concurrency {plan.concurrency}"
        ),
        f"  results: {plan.save_path.relative_to(ROOT).as_posix()}",
        "",
        (
            f"  {'task':<18}{'category':<12}{'lang':<10}{'trials':>6}  "
            f"{'judge':<6}{'$/run':>9}{'$ worst':>9}{'$ total':>10}"
        ),
    ]
    for t in plan.tasks:
        lines.append(
            f"  {t.task_id:<18}{t.category:<12}{t.lang:<10}{plan.trials:>6}  "
            f"{'yes' if t.needs_judge else '-':<6}{t.estimate.total_usd:>9.4f}"
            f"{t.worst.total_usd:>9.4f}{t.estimate.total_usd * plan.trials:>10.4f}"
        )
    total = plan.total_usd
    verdict = "within" if total <= plan.max_usd else "OVER"
    lines += [
        f"  {'estimated total':<63}{total:>10.4f}",
        "",
        (
            "Guard: stop starting new runs once metered spend (every LLM call, priced from "
            f"its tokens) passes ${plan.max_usd:.2f} (estimate is {verdict} the limit)."
        ),
        (
            f"Per run: tau2 stops at --max-steps {plan.max_steps} "
            f"(<= {calls_per_side(plan.max_steps)} LLM calls per side); worst case "
            f"${plan.worst_task.worst.total_usd:.4f} <= --max-usd-per-run "
            f"${plan.max_usd_per_run:.2f}."
        ),
        (
            f"Estimate assumes per run: {a.turns} agent calls on a "
            f"{a.agent_prompt_tokens:,}-token prompt, history +{a.history_growth_tokens} "
            f"tokens/call, {a.agent_output_tokens} output tokens/call; caller {a.turns} calls "
            f"on {a.caller_prompt_tokens} tokens, {a.caller_output_tokens} out. "
            "No prompt caching (upper bound)."
        ),
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ guard
def _usage_cost(messages: Iterable, role: str, price: Price) -> float:
    total = 0.0
    for m in messages:
        if getattr(m, "role", None) == role and getattr(m, "usage", None):
            total += price.cost(
                m.usage.get("prompt_tokens") or 0, m.usage.get("completion_tokens") or 0
            )
    return total


def sim_cost_usd(sim, agent: Price, caller: Price) -> float:
    """Spend for one simulation: tau2's logged cost, or our price table if that's higher.

    tau2 prices calls through LiteLLM. When LiteLLM has no price for a model it logs an
    error and records $0, which would make a guard on logged cost useless, so each side
    is also priced from its logged token usage and the larger figure is used.
    """
    messages = sim.messages or []
    agent_usd = max(sim.agent_cost or 0.0, _usage_cost(messages, "assistant", agent))
    caller_usd = max(sim.user_cost or 0.0, _usage_cost(messages, "user", caller))
    return agent_usd + caller_usd


class _InlineExecutor:
    """Stand-in for LiteLLM's callback thread pool that runs each task at once."""

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001 -- handed back on the future, as a pool would
            future.set_exception(e)
        return future


class CallMeter:
    """Prices every LiteLLM call as it finishes, from its token usage and PRICES.

    This sees each call tau2 makes (agent, caller, judge), including calls in attempts
    tau2 throws away and reruns after an exception, which never reach results.json and
    so are invisible to `sim_cost_usd`. Failed calls are counted; they carry no usage,
    so they cost $0 (Bedrock doesn't bill a failed request). Cache hits cost $0.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.usd = 0.0
        self.calls = 0
        self.failed_calls = 0
        self.cached_calls = 0
        self.by_model: dict[str, dict] = {}
        self.unpriced: Counter = Counter()  # model id -> calls not in PRICES
        self.errors: list[str] = []  # our callback failed; LiteLLM would swallow it

    @property
    def total(self) -> float:
        with self._lock:
            return self.usd

    def on_success(self, kwargs, response, start_time=None, end_time=None) -> None:
        self._record(kwargs, response, failed=False)

    def on_failure(self, kwargs, response, start_time=None, end_time=None) -> None:
        self._record(kwargs, response, failed=True)

    def _record(self, kwargs, response, failed: bool) -> None:
        try:
            kwargs = kwargs or {}
            model = bedrock_model_id(str(kwargs.get("model") or "unknown"))
            usage = getattr(response, "usage", None)
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion = getattr(usage, "completion_tokens", 0) or 0
            cached = bool(kwargs.get("cache_hit"))
            usd, unpriced = 0.0, False
            if not cached and (prompt or completion):
                try:
                    usd = price_for(model).cost(prompt, completion)
                except PlanError:  # build_plan prices every model, so this shouldn't happen
                    unpriced = True
                    usd = float(kwargs.get("response_cost") or 0.0)  # LiteLLM's figure, if any
            with self._lock:
                row = self.by_model.setdefault(
                    model,
                    {
                        "calls": 0,
                        "failed": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "usd": 0.0,
                    },
                )
                row["calls"] += 1
                row["failed"] += failed
                row["prompt_tokens"] += prompt
                row["completion_tokens"] += completion
                row["usd"] += usd
                self.calls += 1
                self.failed_calls += failed
                self.cached_calls += cached
                self.usd += usd
                if unpriced:
                    self.unpriced[model] += 1
        except Exception as e:  # noqa: BLE001 -- recorded and reported, never lost silently
            with self._lock:
                self.errors.append(f"{type(e).__name__}: {e}")

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "metered_calls": self.calls,
                "metered_failed_calls": self.failed_calls,
                "metered_cached_calls": self.cached_calls,
                "metered_by_model": {
                    m: {**r, "usd": round(r["usd"], 6)} for m, r in self.by_model.items()
                },
                "metered_unpriced_calls": dict(self.unpriced),
                "metering_errors": list(self.errors),
            }


@contextmanager
def metering(meter: CallMeter):
    """Register `meter` as a LiteLLM success and failure callback for the duration.

    LiteLLM runs success callbacks on a background thread pool (failure callbacks run
    inline), so a call could return before its cost was counted and the guard would read
    a stale total. Inside this block that pool is replaced by an inline executor: LiteLLM
    looks `litellm.utils.executor` up on every call, so each call's cost is on the meter
    before the call returns to tau2.
    """
    import litellm
    import litellm.utils as litellm_utils

    old_executor = litellm_utils.executor
    litellm_utils.executor = _InlineExecutor()
    litellm.success_callback.append(meter.on_success)
    litellm.failure_callback.append(meter.on_failure)
    try:
        yield meter
    finally:
        for name, fn in (
            ("success_callback", meter.on_success),
            ("failure_callback", meter.on_failure),
        ):
            callbacks = getattr(litellm, name)
            while fn in callbacks:
                callbacks.remove(fn)
        litellm_utils.executor = old_executor


class SpendMeter:
    """Thread-safe running total of the spend logged in results.json (and the judge's)."""

    def __init__(self, start_usd: float = 0.0):
        self._lock = threading.Lock()
        self.sims_usd = start_usd
        self.judge_usd = 0.0

    def add_sim(self, usd: float) -> float:
        with self._lock:
            self.sims_usd += usd
            return self.total

    def add_judge(self, usd: float) -> float:
        with self._lock:
            self.judge_usd += usd
            return self.total

    @property
    def total(self) -> float:
        return self.sims_usd + self.judge_usd


def run_guarded(
    units: Sequence,
    run_one: Callable,
    spent: Callable[[], float],
    max_usd: float,
    concurrency: int = 1,
    on_result: Callable | None = None,
) -> tuple[list, bool]:
    """Run units until done or until `spent()` passes `max_usd`.

    `spent` is the metered total (every LLM call so far, see CallMeter). Returns
    (results, stopped_by_guard). A unit starts only while spend is within the limit, so
    at most `concurrency` units run after the limit is crossed.
    """
    stop = threading.Event()
    if spent() > max_usd:
        return [], True
    lock = threading.Lock()
    pending = iter(units)
    results: list = []

    def worker() -> None:
        while not stop.is_set():
            with lock:
                if stop.is_set():
                    return
                unit = next(pending, None)
            if unit is None:
                return
            result = run_one(unit)
            total = spent()
            with lock:
                results.append(result)
                if on_result:
                    on_result(unit, result, total)
                if total > max_usd:
                    stop.set()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for f in [pool.submit(worker) for _ in range(concurrency)]:
            f.result()
    return results, stop.is_set()


# ------------------------------------------------------------------ caller output
# The simulated caller is an LLM and sometimes writes more than a caller would say: gpt-oss
# leaked <reasoning>...</reasoning> mid-turn, and both gpt-oss and Qwen3-Next wrote stage
# directions like "(voice shaking)" or "*(sighs)*" (caller-gptoss / caller-qwen3next runs).
# The agent must only get the spoken words, so each caller turn is cleaned first.
_REASONING_TAGS = r"(?:reasoning|think|thinking|thought|thoughts|analysis|scratchpad)"
_REASONING_BLOCK = re.compile(
    rf"<\s*{_REASONING_TAGS}\b[^>]*>.*?<\s*/\s*{_REASONING_TAGS}\s*>", re.IGNORECASE | re.DOTALL
)
_REASONING_OPEN = re.compile(rf"<\s*{_REASONING_TAGS}\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
_REASONING_CLOSE = re.compile(rf"\A.*?<\s*/\s*{_REASONING_TAGS}\s*>", re.IGNORECASE | re.DOTALL)
# (voice shaking) / [sighs], optionally in Markdown emphasis: *(sighs)*. No digits inside,
# so a spoken "(080) 4100 2108" or "[APT-000960]" is left alone, and no "#", so a control
# token the caller put in brackets, "(###STOP###)", survives.
_STAGE_DIRECTION = re.compile(r"[*_]{0,3}(?:\([^()\d#\n]{1,80}\)|\[[^\[\]\d#\n]{1,80}\])[*_]{0,3}")
# A line that is only a short emphasised aside: "*…pause…*", "*quietly*", "_sighs_".
_ASIDE_LINE = re.compile(r"^[ \t]*([*_]{1,3})([^*_\n]+)\1[ \t]*$", re.MULTILINE)
CALLER_SILENCE = "..."  # what the agent gets if a turn was nothing but stage directions


def clean_caller_text(text: str | None) -> tuple[str | None, Counter]:
    """The caller's turn with reasoning blocks and stage directions removed.

    Returns (cleaned text, counts of what was removed). Spoken words, emphasis inside a
    sentence, and the ###STOP### / ###TRANSFER### / ###OUT-OF-SCOPE### tokens are kept.
    Reasoning is removed first, so a stop token *mentioned in reasoning* no longer ends
    the call.
    """
    removed: Counter = Counter()
    if not text:
        return text, removed

    cut = "\x00"  # marks removed text, so a line left holding only cuts can be dropped

    def drop(pattern: re.Pattern, kind: str, s: str) -> str:
        s, n = pattern.subn(cut, s)
        removed[kind] += n
        return s

    out = drop(_REASONING_BLOCK, "reasoning", text)
    out = drop(_REASONING_CLOSE, "reasoning", out)  # reasoning, then a closing tag only
    out = drop(_REASONING_OPEN, "reasoning", out)  # opened, never closed: cut to the end
    out = drop(_STAGE_DIRECTION, "stage_direction", out)

    def aside(m: re.Match) -> str:
        words = re.findall(r"\w+", m.group(2))
        if len(words) <= 3 and "###" not in m.group(2):
            removed["stage_direction"] += 1
            return cut
        return m.group(0)  # a whole emphasised sentence is speech: keep it

    out = _ASIDE_LINE.sub(aside, out)
    lines = []
    for line in out.splitlines():
        if cut in line and not line.replace(cut, "").strip():
            continue  # the line was nothing but removed text: drop it, blank line and all
        lines.append(re.sub(r"[ \t]{2,}", " ", line.replace(cut, " ")).strip())
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    out = re.sub(r" +([,.?!…])", r"\1", out)  # "Hi (sighs) , okay" -> "Hi, okay"
    removed = +removed  # drop zero counts
    if not removed:
        return text, removed
    return out or CALLER_SILENCE, removed


# ------------------------------------------------------------------ premature stops
# Pilot2 (runs/text/pilot2-nova-lite): the agent read the change back and asked for a "yes";
# the caller answered "Yes ... ###STOP###" in one message, so the call ended before the agent
# could make the change (reschedule-001-en, routing-004-en; both scored 0). The caller's
# script now forbids that (CALLER_RULES in domains/hospital_in/environment.py); this is the
# harness safety net for when the caller LLM does it anyway.
STOP_TOKEN = "###STOP###"
TRANSFER_TOKEN = "###TRANSFER###"
_TRAILING_DECORATION = re.compile(r"[\s*_\"'”’)\]]+\Z")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\n+")
# The agent asks for a yes or a confirmation without ending on "?": 'Please say "yes" to
# proceed.' (reschedule-001-en), 'If this works, say "yes" and I'll book it' (routing-004-en).
_CONFIRM_REQUEST = re.compile(
    r"""\bsay\s+["“'‘]?\s*(?:yes|haan|ha|ok|okay)\b"""
    r"""|["“'‘](?:yes|haan|ha)["”'’]?\s+(?:bol|kah|keh)"""
    r"""|\b(?:please|kindly|can\s+you|could\s+you)\s+confirm\b"""
    r"""|(?:\A|[.!?:\n]\s*)to\s+confirm\b"""  # "To confirm, you'd like..." (not "an SMS to confirm")
    r"""|\bconfirm\s+(?:kar|kijiye|karein|karen|karo)"""
    r"""|\b(?:shall\s+i|do\s+you\s+want\s+me\s+to|would\s+you\s+like\s+me\s+to)\b"""
    r"""|\b(?:let\s+me\s+know|tell\s+me)\s+if\s+(?:this|that|it)(?:'s|’s|\s+is)?\s+"""
    r"""(?:works?|suits?|correct|fine|okay|ok|right)\b""",
    re.IGNORECASE,
)
# "Is there anything else I can help you with?" is the agent wrapping up, not waiting on an
# answer: hanging up after it is fine.
_CLOSING_QUESTION = re.compile(
    r"\b(?:anything|something)\s+else\b|\bany\s+other\s+(?:question|query|help)"
    r"|\bkuch\s+aur\b|\baur\s+kuch\b"
    r"|\b(?:aur\s+koi|koi\s+aur)\s+(?:madad|sahayata|help|sawal|cheez)",  # routing-004-hi
    re.IGNORECASE,
)


def agent_awaits_answer(text: str | None) -> bool:
    """Whether the agent's turn asks the caller something it must answer before hanging up:
    it ends with a question (other than a wrap-up "anything else?"), or it asks for a
    yes or a confirmation."""
    if not text:
        return False
    if _CONFIRM_REQUEST.search(text):
        return True
    body = _TRAILING_DECORATION.sub("", text)
    if not body.endswith(("?", "？")):
        return False
    last = [s for s in _SENTENCE_END.split(body) if s.strip()][-1]
    return not _CLOSING_QUESTION.search(last)


# gen1b-sonnet46 report-g002-en: after the agent's wrap-up "Is there anything else I can help
# you with?" the caller asked a new question and asked for a follow-up booking, then sent
# ###STOP### in the same message; the call ended unanswered and scored 0. A caller turn that
# asks something itself ("?" or a request) is not a goodbye.
_CALLER_REQUEST = re.compile(
    r"\bI(?:['’]d|\s+would)\s+like\b(?!\s+to\s+thank)"
    r"|\bplease\s+book\b"
    r"|\b(?:can|could)\s+you\b"
    # Not "bas itna hi chahiye tha" (that's all I needed), "aur koi madad nahi chahiye"
    # (no more help needed) or "kuch aur chahiye toh..." (if you need anything...).
    r"|(?<!nahi\s)(?<!nahin\s)\bchahiye\b(?!\s+(?:tha|thi|the|toh|to)\b)"
    r"|\bkar\s+(?:dijiye|dijie|dijiyega|dein|de)\b",
    re.IGNORECASE,
)


def caller_asks(text: str | None) -> bool:
    """Whether the caller's turn asks something itself: a "?" or a request such as
    "I'd like", "please book", "can you", "chahiye" or "kar dijiye"."""
    if not text:
        return False
    return "?" in text or "？" in text or bool(_CALLER_REQUEST.search(text))


def block_premature_stop(
    caller_text: str | None, agent_text: str | None
) -> tuple[str | None, bool]:
    """The caller's turn with ###STOP### removed if it would hang up on an open question:
    the agent's (see agent_awaits_answer) or one the caller's own turn asks (caller_asks).

    Returns (text, blocked). Left alone: no stop token; a ###TRANSFER###; neither side
    asks anything; or a turn that is only the stop token, which is a hang-up rather than
    an answer, and leaves nothing to say if removed.
    """
    if not caller_text or STOP_TOKEN not in caller_text or TRANSFER_TOKEN in caller_text:
        return caller_text, False
    rest = _without_token(caller_text, STOP_TOKEN)
    if not re.search(r"[^\W_]", rest):
        return caller_text, False
    if not (agent_awaits_answer(agent_text) or caller_asks(rest)):
        return caller_text, False
    return rest, True


def _without_token(text: str, token: str) -> str:
    """`text` with `token` removed and the spacing and emphasis it leaves behind tidied."""
    rest = text.replace(token, " ")
    rest = re.sub(r"[*_]{1,3}\s*[*_]{1,3}", " ", rest)  # "*###STOP###*" leaves "**"
    rest = re.sub(r"[ \t]{2,}", " ", rest)
    rest = re.sub(r" +([,.?!…])", r"\1", rest)
    return re.sub(r"\n{3,}", "\n\n", rest).strip()


# ------------------------------------------------------------------ false transfers
# Pilot3 (runs/text/pilot3-nova-lite): the caller sent ###TRANSFER###, which ends the call in
# tau2, although the agent had not transferred it. routing-003-en: the agent offered a
# transfer ("Would you like me to transfer you?") and the caller answered with the bare token.
# booking-002-en: the caller appended it to a question the agent had not answered yet. Both
# calls ended there and scored 0. A ###TRANSFER### only ends the call once the agent's
# transfer_to_human_agent call has succeeded in this call; until then the token is removed.
TRANSFER_TOOL = "transfer_to_human_agent"
# What the agent hears when the caller's turn was only the token: the caller asking for the
# transfer it tried to signal, so the agent can make it (no names, numbers or other details).
TRANSFER_REQUEST = "Please transfer me."


def block_false_transfer(caller_text: str | None, transferred: bool) -> tuple[str | None, bool]:
    """The caller's turn with ###TRANSFER### removed if the agent has not transferred the call.

    Returns (text, blocked). Left alone: no transfer token, or `transferred` (the agent's
    transfer_to_human_agent call succeeded in this call). A turn that was only the token
    becomes TRANSFER_REQUEST.
    """
    if not caller_text or TRANSFER_TOKEN not in caller_text or transferred:
        return caller_text, False
    rest = _without_token(caller_text, TRANSFER_TOKEN)
    if not re.search(r"[^\W_]", rest):
        rest = TRANSFER_REQUEST
    return rest, True


def _content_of_last(messages: Sequence | None, role: str) -> str | None:
    for m in reversed(messages or []):
        if getattr(m, "role", None) == role and getattr(m, "content", None):
            return m.content
    return None


class CallerCleanupStats:
    """How often the caller's turns needed cleaning or a premature stop blocked, for
    summary.json.

    Blocked stops are also counted per run ("task#trial"); `begin_run` names the run the
    current thread is simulating. A thread also remembers the last turn it unblocked, so a
    caller that ends the call again on its very next turn is let go (`repeat_stops_allowed`)
    instead of looping to --max-steps over a question it won't answer.

    A thread also remembers whether the agent has transferred its current call
    (`mark_transferred`); until it has, a caller ###TRANSFER### is removed and counted in
    `false_transfers_blocked`. Both per-call memories are reset by `begin_run` and by
    `new_call` (the caller's first turn), so a run tau2 retries starts clean.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self.turns = 0
        self.turns_cleaned = 0
        self.removed: Counter = Counter()
        self.stops_blocked = 0
        self.stops_blocked_by_run: Counter = Counter()
        self.repeat_stops_allowed = 0
        self.transfers_blocked = 0
        self.transfers_blocked_by_run: Counter = Counter()

    def begin_run(self, run_key: str) -> None:
        self._local.run_key = run_key
        self.new_call()

    def new_call(self) -> None:
        self._local.last_unblocked = None
        self._local.transferred = False

    def mark_transferred(self) -> None:
        """The agent's transfer_to_human_agent call succeeded in this thread's call."""
        self._local.transferred = True

    def guard_transfer(self, caller_text: str | None) -> str | None:
        """Apply block_false_transfer to a caller turn, given this thread's call."""
        transferred = getattr(self._local, "transferred", False)
        text, blocked = block_false_transfer(caller_text, transferred)
        if blocked:
            with self._lock:
                self.transfers_blocked += 1
                self.transfers_blocked_by_run[getattr(self._local, "run_key", "?")] += 1
        return text

    def add(self, removed: Counter) -> None:
        with self._lock:
            self.turns += 1
            self.turns_cleaned += bool(removed)
            self.removed.update(removed)

    def guard_stop(self, caller_text: str | None, messages: Sequence | None) -> str | None:
        """Apply block_premature_stop to a caller turn, given the caller's flipped history
        (tau2 shows the caller the agent's turns as role "user", its own as "assistant")."""
        text, blocked = block_premature_stop(caller_text, _content_of_last(messages, "user"))
        last_unblocked = getattr(self._local, "last_unblocked", None)
        repeat = last_unblocked is not None and (
            _content_of_last(messages, "assistant") == last_unblocked
        )
        if blocked and repeat:
            with self._lock:
                self.repeat_stops_allowed += 1
            self._local.last_unblocked = None
            return caller_text
        if not blocked:
            return text
        self._local.last_unblocked = text
        with self._lock:
            self.stops_blocked += 1
            self.stops_blocked_by_run[getattr(self._local, "run_key", "?")] += 1
        return text

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "caller_turns": self.turns,
                "caller_turns_cleaned": self.turns_cleaned,
                "caller_removed": dict(self.removed),
                "premature_stops_blocked": self.stops_blocked,
                "premature_stops_blocked_by_run": dict(self.stops_blocked_by_run),
                "repeat_stops_allowed": self.repeat_stops_allowed,
                "false_transfers_blocked": self.transfers_blocked,
                "false_transfers_blocked_by_run": dict(self.transfers_blocked_by_run),
            }


def is_hinglish_caller(messages) -> bool:
    """The caller speaks Hinglish if its system prompt (persona) says so."""
    for m in messages or []:
        if getattr(m, "role", None) == "system" and "You speak Hinglish" in (m.content or ""):
            return True
    return False


def clean_caller_output(stats: CallerCleanupStats, speech=None) -> None:
    """Clean every caller turn before tau2 hands it to the agent or checks it for ###STOP###.

    tau2 v2.0.0's UserSimulator._generate_next_message calls `generate(...)`, looked up in
    tau2.user.user_simulator at call time, and builds the UserMessage from the reply's
    content; replacing that `generate` cleans the text everywhere it is used (the agent,
    the stop check, the caller's own history, results.json). The model's untouched reply
    stays in the message's raw_data.

    Cleaning also blocks a false transfer: a ###TRANSFER### sent before the agent's
    transfer_to_human_agent call has succeeded in this call is removed (`guard_transfer`;
    tau2's Environment.get_response is wrapped to see that call). And it blocks a premature
    stop: a ###STOP### sent while the agent's last turn asked a question or for a yes is
    removed, so the call goes on (`guard_stop`).
    """
    from tau2.environment.environment import Environment
    from tau2.user import user_simulator as us

    original = us.generate

    def caller_generate(*args, **kwargs):
        reply = original(*args, **kwargs)
        cleaned, removed = clean_caller_text(reply.content)
        stats.add(removed)
        messages = kwargs.get("messages", args[1] if len(args) > 1 else None)
        if not any(getattr(m, "role", None) == "assistant" for m in messages or []):
            stats.new_call()  # the caller's first turn (its own turns are "assistant" here)
        guarded = stats.guard_stop(stats.guard_transfer(cleaned), messages)
        if speech is not None and guarded:
            # The agent hears this turn only through TTS -> phone channel -> ASR.
            run_key = getattr(stats._local, "run_key", "?")
            guarded = speech.hear(guarded, is_hinglish_caller(messages), run_key)
        if guarded != reply.content:
            return reply.model_copy(update={"content": guarded})
        return reply

    us.generate = caller_generate

    original_get_response = Environment.get_response

    def get_response(self, message):
        response = original_get_response(self, message)
        if (
            message.name == TRANSFER_TOOL
            and message.requestor == "assistant"
            and not response.error
        ):
            stats.mark_transferred()
        return response

    Environment.get_response = get_response


# ------------------------------------------------------------------ NL judge
JUDGE_ATTEMPTS = 2  # first try + one retry
JUDGE_JSON_INSTRUCTION = (
    "\n\nReply with the JSON object only: start with { and end with }. No markdown, no "
    "code fences and no text before or after it. Include one entry in `results` for "
    "every expected outcome."
)
JUDGE_RETRY_PROMPT = (
    "Your previous reply could not be used ({reason}). Reply again with only the JSON "
    "object described above, starting with {{ and ending with }}."
)
_FENCE = re.compile(r"^\s*```[\w-]*\s*$", re.MULTILINE)


class JudgeReplyError(ValueError):
    """One judge reply could not be parsed into tau2's expected shape."""


class JudgeError(RuntimeError):
    """The judge gave no usable reply after all attempts; the run is marked judge_error."""


def parse_judge_reply(text: str | None) -> dict:
    """The judge's reply as {"results": [...]} with the keys tau2 reads, or JudgeReplyError.

    Strips Markdown code fences, then takes the first JSON object in the text that has a
    `results` list, so prose before or after the object doesn't matter.
    """
    if not text or not text.strip():
        raise JudgeReplyError("empty reply")
    body = _FENCE.sub("", text)
    decoder = json.JSONDecoder()
    found_object = False
    data = None
    for i, ch in enumerate(body):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(body, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found_object = True
            if isinstance(obj.get("results"), list):
                data = obj
                break
    if data is None:
        raise JudgeReplyError(
            "JSON object has no `results` list" if found_object else "no JSON object found"
        )
    if not data["results"]:
        raise JudgeReplyError("`results` is empty")
    results = []
    for n, r in enumerate(data["results"]):
        if not isinstance(r, dict) or "expectedOutcome" not in r or "metExpectation" not in r:
            raise JudgeReplyError(f"result {n} lacks expectedOutcome/metExpectation")
        met = r["metExpectation"]
        if isinstance(met, str) and met.strip().lower() in ("true", "false"):
            met = met.strip().lower() == "true"
        if not isinstance(met, bool):
            raise JudgeReplyError(f"result {n} metExpectation is {met!r}, not true/false")
        results.append(
            {
                "expectedOutcome": str(r["expectedOutcome"]),
                "metExpectation": met,
                "reasoning": str(r.get("reasoning", "")),
            }
        )
    return {"results": results}


def _ask_for_json(messages: list) -> list:
    """Append JUDGE_JSON_INSTRUCTION to the judge's system prompt."""
    out = list(messages)
    for i, m in enumerate(out):
        if getattr(m, "role", None) == "system":
            out[i] = m.model_copy(update={"content": (m.content or "") + JUDGE_JSON_INSTRUCTION})
            break
    return out


def route_nl_judge(judge_llm: str, judge_price: Price, meter: SpendMeter) -> None:
    """Send tau2's NL-assertion judge to `judge_llm`, make it robust, and meter its spend.

    tau2 v2.0.0 calls `generate(model=DEFAULT_LLM_NL_ASSERTIONS, ...)` (an OpenAI model,
    with OpenAI-only `reasoning_effort`) from evaluator_nl_assertions and then runs
    `json.loads` on the raw reply, which crashed the whole simulation when Nova 2 Lite
    answered with non-JSON (and tau2 then reran the conversation 3 more times).
    Here the judge is asked for bare JSON (in the prompt: LiteLLM's Bedrock
    `response_format` is a forced tool call that not every model accepts), each reply is
    cleaned by `parse_judge_reply` and handed to tau2 as plain JSON, and a bad reply gets
    one retry. If that fails too, `NLAssertionsEvaluator.calculate_reward` returns a
    zero-reward result marked `judge_error` instead of raising, so the run is kept,
    graded on everything else, and reported (see `run_outcome`).
    """
    from tau2.data_model.message import UserMessage
    from tau2.data_model.simulation import RewardInfo
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator import evaluator_nl_assertions as nl

    original_generate = nl.generate
    original_reward = nl.NLAssertionsEvaluator.calculate_reward  # bound classmethod

    def metered(messages, kwargs):
        msg = original_generate(model=judge_llm, messages=messages, **kwargs)
        usage = getattr(msg, "usage", None) or {}
        meter.add_judge(
            max(
                getattr(msg, "cost", 0.0) or 0.0,
                judge_price.cost(
                    usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
                ),
            )
        )
        return msg

    def judge_generate(*args, model=None, messages=None, **kwargs):
        kwargs.pop("reasoning_effort", None)
        base = _ask_for_json(messages or [])
        attempt_messages = base
        reason, last_text = "", ""
        for _ in range(JUDGE_ATTEMPTS):
            try:
                msg = metered(attempt_messages, kwargs)
            except Exception as e:  # noqa: BLE001 -- any judge failure must not rerun the call
                reason, last_text = f"{type(e).__name__}: {e}", ""
                attempt_messages = base
                continue
            last_text = msg.content or ""
            try:
                parsed = parse_judge_reply(last_text)
            except JudgeReplyError as e:
                reason = str(e)
                attempt_messages = base
                if last_text.strip():  # show it its reply; Bedrock rejects empty turns
                    attempt_messages = [
                        *base,
                        msg,
                        UserMessage(role="user", content=JUDGE_RETRY_PROMPT.format(reason=reason)),
                    ]
                continue
            return msg.model_copy(update={"content": json.dumps(parsed, ensure_ascii=False)})
        raise JudgeError(
            f"NL judge {judge_llm} gave no usable JSON after {JUDGE_ATTEMPTS} attempts: "
            f"{reason}. Last reply: {last_text[:200]!r}"
        )

    def safe_reward(cls, task, full_trajectory):
        expected = len((task.evaluation_criteria and task.evaluation_criteria.nl_assertions) or [])
        try:
            info = original_reward(task=task, full_trajectory=full_trajectory)
        except JudgeError as e:
            error = str(e)
        else:
            graded = len(info.nl_assertions or [])
            if graded == expected:
                return info
            # tau2 counts a missing result as met (all([]) is True), so this must not pass.
            error = f"NL judge {judge_llm} graded {graded} of {expected} expected outcomes"
        return RewardInfo(
            reward=0.0,
            nl_assertions=[],
            info={"judge_error": error},
            reward_breakdown={RewardType.NL_ASSERTION: 0.0},
        )

    nl.generate = judge_generate
    nl.NLAssertionsEvaluator.calculate_reward = classmethod(safe_reward)


def judge_error_of(sim) -> str | None:
    """The judge_error text that `route_nl_judge` left on a graded run, if any."""
    info = getattr(getattr(sim, "reward_info", None), "info", None) or {}
    nl_info = info.get("nl") if isinstance(info, dict) else None
    return nl_info.get("judge_error") if isinstance(nl_info, dict) else None


# ------------------------------------------------------------------ outcomes
OK_REASONS = {"user_stop", "agent_stop"}


def run_outcome(sim, max_steps: int) -> dict:
    """One run's termination reason, reward and status for printing and summary.json.

    status is "ok", "infrastructure_error", "max_steps", "judge_error", or the
    termination reason for any other early stop (agent_error, too_many_errors, ...).
    Runs whose grade means nothing (infrastructure or judge errors) get reward None.
    """
    reason = getattr(sim.termination_reason, "value", sim.termination_reason)
    reward = sim.reward_info.reward if sim.reward_info else None
    out = {
        "task_id": sim.task_id,
        "trial": sim.trial,
        "termination_reason": reason,
        "reward": reward,
        "status": "ok",
    }
    judge_error = judge_error_of(sim)
    if reason == "infrastructure_error":
        info = sim.info or {}
        attempts = info.get("failed_after_attempts")
        out.update(
            status="infrastructure_error",
            reward=None,
            error=f"{info.get('error_type', 'Error')}: {info.get('error', '')}".strip(),
            attempts=attempts,
        )
    elif reason == "max_steps":
        out.update(
            status="max_steps",
            error=(
                f"stopped after {len(sim.messages or [])} messages (--max-steps {max_steps}); "
                "graded 0"
            ),
        )
    elif judge_error:
        out.update(status="judge_error", reward=None, error=judge_error)
    elif reason not in OK_REASONS:
        out["status"] = reason
    return out


def build_summary(
    plan: Plan,
    sims: Sequence,
    meter: SpendMeter,
    stopped: bool,
    calls: CallMeter | None = None,
    prior_metered_usd: float = 0.0,
    caller: CallerCleanupStats | None = None,
    a: Assumptions = DEFAULT_ASSUMPTIONS,
) -> dict:
    """summary.json. spent_usd: the runs kept in results.json, as tau2 logged them (plus
    the judge). metered_usd: every LLM call made under this run name, discarded attempts
    included; this is the figure the --max-usd guard uses and the one to check the AWS
    bill against. caller_*: how many caller turns (this session) had reasoning or stage
    directions removed. premature_stops_blocked (this session): caller ###STOP### tokens
    removed because the agent had just asked a question or the caller's own turn asked
    something, with a per-run breakdown.
    false_transfers_blocked (this session): caller ###TRANSFER### tokens removed because
    the agent had not transferred the call, with a per-run breakdown."""
    calls = calls or CallMeter()
    caller = caller or CallerCleanupStats()
    outcomes = [run_outcome(s, plan.max_steps) for s in sims]

    def pick(status: str) -> list[dict]:
        keys = ("task_id", "trial", "termination_reason", "error", "attempts")
        return [{k: o[k] for k in keys if k in o} for o in outcomes if o["status"] == status]

    errors = pick("infrastructure_error")
    return {
        "agent_llm": plan.agent_llm,
        "user_llm": plan.user_llm,
        "judge_llm": plan.judge_llm,
        "max_usd": plan.max_usd,
        "max_usd_per_run": plan.max_usd_per_run,
        "max_steps_limit": plan.max_steps,
        "estimated_usd": round(plan.total_usd, 6),
        "worst_case_usd_per_run": round(plan.worst_task.worst.total_usd, 6),
        "metered_usd": round(prior_metered_usd + calls.total, 6),
        "metered_usd_this_session": round(calls.total, 6),
        "spent_usd": round(meter.total, 6),
        "spent_judge_usd": round(meter.judge_usd, 6),
        **calls.as_dict(),
        "stopped_by_guard": stopped,
        "runs_planned": plan.n_runs,
        "runs_done": len(outcomes),
        "outcomes": dict(Counter(o["status"] for o in outcomes)),
        "errors": errors,
        "max_steps": pick("max_steps"),
        "judge_errors": pick("judge_error"),
        # tau2 reruns a run that raises, up to 4 attempts in all. The attempts it throws
        # away are not in results.json (so not in spent_usd) but are in metered_usd.
        "discarded_attempts": sum(e.get("attempts") or 0 for e in errors),
        "rewards": {f"{o['task_id']}#{o['trial']}": o["reward"] for o in outcomes},
        "runs": outcomes,
        **caller.as_dict(),
        "assumptions": asdict(a),
    }


def prior_metered_usd(summary_path: Path, fallback: float) -> float:
    """Metered spend from an earlier session of this run name (for --resume).

    Uses the earlier summary.json's metered_usd; if there is none (no earlier session, or
    one that died before writing it), falls back to the cost logged in results.json.
    """
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8")).get("metered_usd")
    except (OSError, ValueError, AttributeError):
        value = None
    return max(float(value), fallback) if isinstance(value, int | float) else fallback


def format_outcome(o: dict, metered: float | None = None) -> str:
    line = (
        f"  {o['task_id']} trial {o['trial'] + 1}: {o['termination_reason']}  reward {o['reward']}"
    )
    if metered is not None:
        line += f"  metered ${metered:.4f}"
    if o.get("error"):
        line += f"\n      {o['status']}: {o['error'][:300]}"
    return line


# ------------------------------------------------------------------ real run
def make_config(plan: Plan, args: argparse.Namespace):
    """tau2's TextRunConfig for the plan. Builds an object only; calls nothing."""
    from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring
    from tau2.data_model.simulation import TextRunConfig

    config = TextRunConfig(
        domain=DOMAIN,
        agent="llm_agent",
        llm_agent=plan.agent_llm,
        user="user_simulator",
        llm_user=plan.user_llm,
        num_trials=plan.trials,
        max_steps=plan.max_steps,
        max_concurrency=plan.concurrency,
        seed=args.seed,
        save_to=plan.save_path.parent.name,
        auto_resume=args.resume,
        # No task uses communicate_info (build_plan refuses them), so no judge is needed.
        communicate_scoring=CommunicateScoring(scorer=CommunicateScorer.SUBSTRING),
    )
    config.validate()
    return config


def run(plan: Plan, tasks: list, args: argparse.Namespace) -> dict:
    """Run the plan on Bedrock. This is the only function that can spend money."""
    from loguru import logger
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import _BatchContext, prepare_batch, run_unit

    os.environ.setdefault("AWS_REGION_NAME", AWS_REGION)
    save_path = plan.save_path
    if save_path.exists() and not args.resume:
        raise PlanError(f"{save_path} exists. Pass --resume to continue it, or pick --save-to.")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config = make_config(plan, args)
    logger.remove()
    logger.add(sys.stderr, level=config.log_level)

    agent_p, user_p = price_for(plan.agent_llm), price_for(plan.user_llm)
    prep = prepare_batch(
        config,
        tasks,
        save_path=save_path,
        save_dir=save_path.parent,
        evaluation_type=EvaluationType.ALL,
    )
    results = prep.simulation_results
    already = sum(sim_cost_usd(s, agent_p, user_p) for s in results.simulations)
    meter = SpendMeter(start_usd=already)
    prior_metered = prior_metered_usd(save_path.parent / "summary.json", fallback=already)
    calls = CallMeter()
    caller = CallerCleanupStats()
    speech = speech_meter = None
    if getattr(args, "speech_channel", "none") != "none":
        import boto3

        from simulator.speech_channel import SpeechChannel, SpeechMeter, bucket_for

        speech_meter = SpeechMeter()
        bucket = bucket_for(boto3.Session(region_name=AWS_REGION))
        speech = SpeechChannel(args.speech_channel, args.speech_snr, bucket, speech_meter)
    clean_caller_output(caller, speech)
    if plan.judge_llm:
        route_nl_judge(plan.judge_llm, price_for(plan.judge_llm), meter)

    ctx = _BatchContext(
        config=config,
        evaluation_type=EvaluationType.ALL,
        save_dir=save_path.parent,
        user_voice_settings=prep.user_voice_settings,
        user_persona_config=prep.user_persona_config,
        info=results.info,
        save_fn=prep.save_fn,
    )
    tasks_by_id = {t.id: t for t in prep.tasks}

    def run_one(unit):
        caller.begin_run(f"{unit.task_id}#{unit.trial}")
        return run_unit(ctx, tasks_by_id[unit.task_id], unit.trial, unit.seed, unit.progress_str)

    def on_result(unit, sim, metered):
        meter.add_sim(sim_cost_usd(sim, agent_p, user_p))
        print(format_outcome(run_outcome(sim, plan.max_steps), metered), flush=True)

    print(f"Starting {len(prep.units)} run(s); ${prior_metered:.4f} already metered for this name.")
    with metering(calls):
        new, stopped = run_guarded(
            prep.units,
            run_one,
            lambda: prior_metered + calls.total + (speech_meter.total if speech_meter else 0.0),
            plan.max_usd,
            plan.concurrency,
            on_result,
        )
    results.simulations.extend(new)

    summary = build_summary(plan, results.simulations, meter, stopped, calls, prior_metered, caller)
    if speech_meter is not None:
        summary["speech_channel"] = {
            "condition": args.speech_channel,
            "snr_db": args.speech_snr if args.speech_channel == "phone_noisy" else None,
            "asr": "Amazon Transcribe batch, IdentifyMultipleLanguages en-IN+hi-IN",
            "tts": "Amazon Polly Kajal (neural) / Arjun (generative), en-IN or hi-IN",
            **speech_meter.summary(),
        }
        summary["metered_usd"] = round(summary.get("metered_usd", 0) + speech_meter.total, 6)
        log_path = save_path.parent / "speech_log.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            for e in speech_meter.log:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (save_path.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ------------------------------------------------------------------ CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--agent-llm", required=True, help="bedrock/<model-id> for the front-desk agent")
    p.add_argument("--user-llm", required=True, help="bedrock/<model-id> for the simulated caller")
    p.add_argument("--judge-llm", help="bedrock/<model-id> for NL-assertion grading (if needed)")
    p.add_argument(
        "--split",
        default="base",
        help="task split: 'base' (hand-written, default) or 'generated' (taskgen.py)",
    )
    p.add_argument("--tasks", nargs="+", help="task ids (default: the whole split)")
    p.add_argument("--trials", type=int, default=1, help="runs per task (pass^k); default 1")
    p.add_argument(
        "--max-usd", type=float, required=True, help="stop once logged spend passes this"
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="tau2 ends a run after this many messages (tool results count); default 40",
    )
    p.add_argument(
        "--max-usd-per-run",
        type=float,
        default=0.10,
        help="refuse the plan if a run that uses every step could cost more; default 0.10",
    )
    p.add_argument("--concurrency", type=int, default=1, help="simulations in parallel; default 1")
    p.add_argument("--seed", type=int, default=300, help="tau2 base seed (tau2 default 300)")
    p.add_argument("--save-to", help="run name under runs/text/ (default: timestamp + models)")
    p.add_argument("--resume", action="store_true", help="continue an existing --save-to run")
    p.add_argument("--dry-run", action="store_true", help="print the plan and cost; call nothing")
    p.add_argument(
        "--speech-channel",
        choices=["none", "phone", "phone_noisy"],
        default="none",
        help="agent hears each caller turn through Polly -> phone channel -> Transcribe "
        "(simulator/speech_channel.py); adds Polly/Transcribe cost to the guard",
    )
    p.add_argument(
        "--speech-snr", type=float, default=15.0, help="noise SNR in dB for phone_noisy (default 15)"
    )
    return p.parse_args(argv)


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    data_dir = os.environ.get("TAU2_DATA_DIR")
    if data_dir and not Path(data_dir).is_absolute():  # .env gives it relative to the repo
        os.environ["TAU2_DATA_DIR"] = str(ROOT / data_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()  # before tau2 is imported: it reads TAU2_DATA_DIR at import time

    from scripts.pandas_compat import install_if_blocked, stub_installed

    install_if_blocked()
    from tau2.registry import registry

    from domains.hospital_in.environment import register

    register()
    try:
        all_tasks = registry.get_tasks_loader(DOMAIN)(args.split)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        plan, tasks = build_plan(args, all_tasks)
    except PlanError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(format_plan(plan))
    if args.dry_run:
        print("\nDry run: no AWS calls made, nothing written.")
        return 0
    if stub_installed():
        print(
            "note: pandas is blocked here; tau2's own metrics are skipped, summary.json has ours."
        )
    # Bedrock's Converse API rejects a conversation that doesn't start with a user turn
    # (the simulated caller's first call has only a system prompt). Let litellm insert a
    # placeholder user turn instead of failing the run as an infrastructure_error.
    import litellm

    litellm.modify_params = True
    try:
        summary = run(plan, tasks, args)
    except PlanError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return report(summary)


def report(summary: dict) -> int:
    """Print the summary (without the per-run lists) and pick the exit code."""
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rewards", "runs")}, indent=2))
    for key, what in (("max_steps", "hit --max-steps"), ("judge_errors", "had a judge_error")):
        if summary[key]:
            print(f"warning: {len(summary[key])} run(s) {what}; see summary.json", file=sys.stderr)
    if summary["errors"]:
        print(
            f"error: {len(summary['errors'])} run(s) ended in infrastructure_error: "
            + ", ".join(f"{e['task_id']}#{e['trial']}" for e in summary["errors"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
