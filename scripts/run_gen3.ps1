# Run (or resume) the gen3 benchmark: 5 agents x generated batches 2 and 3 (80 tasks each).
# Safe to re-run: a run folder that already exists is continued with --resume, and finished
# runs are skipped by run_text.py. Logs go to runs\text\<name>.log (appended).
#
# Usage (repo root, after `aws sso login --profile carecall`):
#   powershell -ExecutionPolicy Bypass -File scripts\run_gen3.ps1
#
# Paused on Nov 11, 2026: batch 2 done for DeepSeek, Nova 2 Lite, Sonnet 4.6, Mistral;
# Qwen3-Next at 7/80; batch 3 not started.

Set-Location (Split-Path -Parent $PSScriptRoot)
$agents = @(
    'deepseek.v3.2',
    'global.anthropic.claude-sonnet-4-6',
    'mistral.mistral-large-3-675b-instruct',
    'global.amazon.nova-2-lite-v1:0',
    'qwen.qwen3-next-80b-a3b'
)
foreach ($b in @('b2', 'b3')) {
    foreach ($a in $agents) {
        $name = "gen3-$b-" + ($a -replace '[^a-z0-9]+', '-')
        $extra = @()
        if (Test-Path "runs\text\$name\results.json") { $extra = @('--resume') }
        .\.venv\Scripts\python.exe scripts\run_text.py --split "generated_$b" `
            --agent-llm "bedrock/$a" --user-llm bedrock/openai.gpt-oss-120b-1:0 `
            --judge-llm bedrock/qwen.qwen3-next-80b-a3b --trials 1 --concurrency 3 `
            --max-usd 18 --max-usd-per-run 0.60 --save-to $name @extra *>> "runs\text\$name.log"
        "$name exit=$LASTEXITCODE"
    }
}
