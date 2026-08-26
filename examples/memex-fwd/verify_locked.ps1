# Verify one architecture against the fork's own llama_decode, under the machine lock.
#
# One model per invocation, on purpose. Gemma 4 is 17 GB and Qwen 3.6 is 29.3 GB on a 32 GB
# machine; two at once does not fit, and a run that swaps is a run whose numbers mean nothing.
#
# Two checks, in this order:
#
#   1. Prefill. Our graph over the whole prompt against llama_decode over the same tokens,
#      compared at the logits AND, with --probe all, layer by layer under the reference's own
#      node names. This is the cheap check and it validates one graph over a fixed input.
#
#   2. Decode, step by step. This is the check that matters. A prefill carries nothing between
#      steps, so it cannot see a fault in state that IS carried - which is exactly what
#      qwen35moe's thirty delta-net layers do. A slightly wrong carry degrades smoothly and
#      reads as "the model is a bit worse", indistinguishable from quantisation noise.
#
# --no-repack throughout: repacking rewrites weights in place and forces mmap off, which turns
# a 29 GB file into 29 GB of private resident memory. Correctness does not need it and this
# machine cannot afford it.

param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Prompt = 'The capital of France is Paris. The capital of Japan is',
    [int]$Tokens = 24,
    [int]$DecodeCheck = 6,
    [int]$Threads = 8,
    [int]$TimeoutMin = 180,
    [switch]$SkipProbe,
    # A third check, for gemma4 only and only worth the minutes it costs there: a prompt long
    # enough that the sliding window actually bites. With a 24-token prompt every query can
    # still see position zero, so 1024-wide windowing is exercised structurally - two masks,
    # two head geometries, two rope bases - without its BOUNDARY ever being crossed. An
    # off-by-one between `pos - j < n_swa` and `<=` would pass that test silently.
    # 0 disables it.
    [int]$LongTokens = 0,
    [string]$LongPrompt = 'D:/MemeX/results/long_prompt.txt'
)

. C:/Users/User11/Desktop/MemeX/bench/lock.ps1

$exe = 'D:/MemeX/src/ik_llama.cpp/build/bin/Release/llama-memex-fwd.exe'
if (-not (Test-Path -LiteralPath $exe)) { Write-Host "net exe: $exe"; exit 4 }
if (-not (Test-Path -LiteralPath $Model)) { Write-Host "net modeli: $Model"; exit 4 }

$tag = [IO.Path]::GetFileNameWithoutExtension($Model)
$outdir = 'D:/MemeX/results'
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

Write-Host ("lock holder before: " + (Get-LockHolder))
Write-Host "waiting for the machine..."
$t0 = Get-Date
if (-not (Take-Machine -Who 'arch-verify' -TimeoutMin $TimeoutMin)) {
    Write-Host "NE POLUCHIL MASHINU za $TimeoutMin min"
    exit 3
}
Write-Host ("got the machine after {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)

$code = 0
try {
    $o = Get-CimInstance Win32_OperatingSystem
    Write-Host ("free memory at start: {0:N1} GB of {1:N1} GB" -f ($o.FreePhysicalMemory / 1MB),
                ($o.TotalVisibleMemorySize / 1MB))

    # ---- 1. prefill, with the per-layer comparison
    $log1 = Join-Path $outdir "verify_${tag}_prefill.log"
    Write-Host "=== prefill comparison -> $log1 ==="
    $args1 = @('-m', $Model, '-p', $Prompt, '--tokens', $Tokens, '-t', $Threads, '--no-repack')
    if (-not $SkipProbe) { $args1 += @('--probe', 'all') }
    & $exe @args1 2>&1 | Tee-Object -FilePath $log1
    $c1 = $LASTEXITCODE
    Write-Host "prefill exit $c1"

    # ---- 2. decode, step by step. Only worth running if the prefill agreed: a per-step
    # comparison against a prefill that already diverged measures two faults at once.
    if ($c1 -eq 0 -or $c1 -eq 2) {
        $log2 = Join-Path $outdir "verify_${tag}_decode.log"
        Write-Host "=== decode-check ($DecodeCheck steps) -> $log2 ==="
        $args2 = @('-m', $Model, '-p', $Prompt, '--tokens', $Tokens, '-t', $Threads,
                   '--no-repack', '--decode-check', $DecodeCheck)
        & $exe @args2 2>&1 | Tee-Object -FilePath $log2
        $c2 = $LASTEXITCODE
        Write-Host "decode-check exit $c2"
        if ($c2 -ne 0) { $code = $c2 }
    } else {
        Write-Host "prefill failed; not running the decode check"
        $code = $c1
    }

    # ---- 3. the sliding window, if asked for. Separate from the short run rather than
    # replacing it: the short one is the fast signal and this one costs minutes.
    if ($LongTokens -gt 0 -and $c1 -eq 0) {
        if (-not (Test-Path -LiteralPath $LongPrompt)) {
            Write-Host "net dlinnogo prompta: $LongPrompt"
        } else {
            $log3 = Join-Path $outdir "verify_${tag}_window.log"
            Write-Host "=== window crossing ($LongTokens tokens) -> $log3 ==="
            & $exe -m $Model -f $LongPrompt --tokens $LongTokens -t $Threads --no-repack `
                2>&1 | Tee-Object -FilePath $log3
            Write-Host "window test exit $LASTEXITCODE"
            if ($LASTEXITCODE -ne 0 -and $code -eq 0) { $code = $LASTEXITCODE }
        }
    }
} finally {
    Free-Machine
    Write-Host ("lock holder after: " + (Get-LockHolder))
}
exit $code
