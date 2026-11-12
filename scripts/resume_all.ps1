# Resume every unfinished CareCall-Bench run (safe to re-run; finished runs are skipped).
#   powershell -ExecutionPolicy Bypass -File scripts\resume_all.ps1
# Starts three background lanes and returns; logs are runs\text\<name>.log.
# Needs a valid login first: aws sso login --profile carecall

Set-Location (Split-Path -Parent $PSScriptRoot)
$root = (Get-Location).Path
$common = "--user-llm bedrock/openai.gpt-oss-120b-1:0 --judge-llm bedrock/qwen.qwen3-next-80b-a3b --max-usd-per-run 0.60"

$lanes = @(
  # Lane 1: Claude Sonnet queue (daily-cap sensitive, concurrency 2)
  "powershell -ExecutionPolicy Bypass -File scripts\run_sonnet_queue.ps1",
  # Lane 2: pass^4 reliability for the four non-Claude models
  ("foreach (`$a in @('deepseek.v3.2','mistral.mistral-large-3-675b-instruct','global.amazon.nova-2-lite-v1:0','qwen.qwen3-next-80b-a3b')) { " +
   "`$n = 'pk4-b1-' + (`$a -replace '[^a-z0-9]+','-'); `$r=@(); if (Test-Path ""runs\text\`$n\results.json"") { .\.venv\Scripts\python.exe scripts\drop_errors.py ""runs\text\`$n""; `$r=@('--resume') }; " +
   ".\.venv\Scripts\python.exe -W ignore scripts\run_text.py --split generated_b1 --agent-llm ""bedrock/`$a"" $common --trials 4 --concurrency 3 --max-usd 15 --save-to `$n @r *>> ""runs\text\`$n.log"" }"),
  # Lane 3: speech channel for the four non-Claude models
  ("foreach (`$a in @('deepseek.v3.2','qwen.qwen3-next-80b-a3b','global.amazon.nova-2-lite-v1:0','mistral.mistral-large-3-675b-instruct')) { " +
   "`$n = 'voice-b1-' + (`$a -replace '[^a-z0-9]+','-'); `$r=@(); if (Test-Path ""runs\text\`$n\results.json"") { .\.venv\Scripts\python.exe scripts\drop_errors.py ""runs\text\`$n""; `$r=@('--resume') }; " +
   ".\.venv\Scripts\python.exe -W ignore scripts\run_text.py --split generated_b1 --agent-llm ""bedrock/`$a"" $common --trials 1 --concurrency 4 --max-usd 25 --speech-channel phone_noisy --speech-snr 15 --save-to `$n @r *>> ""runs\text\`$n.log"" }")
)
foreach ($cmd in $lanes) {
  Start-Process powershell -WindowStyle Minimized -ArgumentList @(
    '-NoProfile', '-Command', "Set-Location '$root'; `$env:PYTHONUTF8='1'; $cmd"
  )
}
"Started $($lanes.Count) lanes. Check progress with: python -m eval.score runs/text/<run>"
