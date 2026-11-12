# Remaining Claude Sonnet 4.6 work, run when the account's daily Claude token cap has reset.
# ("Too many tokens per day" from Bedrock: an account-level daily limit on Anthropic models,
#  not an adjustable Service Quota; it hit on Nov 11 after ~200 Sonnet conversations.)
#
# Order = importance. Each step clears rate-limited runs first (scripts/drop_errors.py) and
# resumes, so the script is safe to re-run until everything is done.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_sonnet_queue.ps1

Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = '1'
$m = 'global.anthropic.claude-sonnet-4-6'
$common = @('--agent-llm', "bedrock/$m", '--user-llm', 'bedrock/openai.gpt-oss-120b-1:0',
            '--judge-llm', 'bedrock/qwen.qwen3-next-80b-a3b', '--concurrency', '2',
            '--max-usd-per-run', '0.60')

function Run-Step($split, $name, $trials, $cap, $extra) {
    if (Test-Path "runs\text\$name\results.json") {
        .\.venv\Scripts\python.exe scripts\drop_errors.py "runs\text\$name"
        $resume = @('--resume')
    } else { $resume = @() }
    .\.venv\Scripts\python.exe -W ignore scripts\run_text.py --split $split --trials $trials `
        --max-usd $cap --save-to $name @common @extra @resume *>> "runs\text\$name.log"
    "$name exit=$LASTEXITCODE"
}

# 1. Main benchmark: the rate-limited runs of batches 3 and 2.
Run-Step 'generated_b3' 'gen3-b3-global-anthropic-claude-sonnet-4-6' 1 18 @()
Run-Step 'generated_b2' 'gen3-b2-global-anthropic-claude-sonnet-4-6' 1 18 @()
# 2. Speech channel on batch 1 (text vs speech comparison).
Run-Step 'generated_b1' 'voice-b1-global-anthropic-claude-sonnet-4-6' 1 25 @('--speech-channel', 'phone_noisy', '--speech-snr', '15')
# 3. Optional: pass^4 reliability (320 conversations; the biggest Claude token user).
Run-Step 'generated_b1' 'pk4-b1-global-anthropic-claude-sonnet-4-6' 4 60 @()
