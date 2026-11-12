# leaderboard

The v0.1 results are in the main [README](../README.md#results-v01-text-mode-pass1-on-240-generated-tasks)
and, with every number used in the paper (confidence intervals, matched-pair tests, speech
channel, pass^4, hearing test, jury agreement), in
[`paper/generated/results.json`](../paper/generated/results.json).

Regenerate both from run outputs with:

```bash
python -m paper.make_results runs/text/gen2-* runs/text/gen3-* --jury \
  --hearing runs/hearing/score.json --voice runs/text/voice-b1-* \
  --paired-text runs/text/gen2-* --passk runs/text/pk4-b1-*
```

A static leaderboard site is planned for a later release.
