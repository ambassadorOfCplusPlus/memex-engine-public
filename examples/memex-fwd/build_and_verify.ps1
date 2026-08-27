# Build, then verify ONE architecture - all inside a single lock acquisition.
#
# Chained rather than run as two scripts because the queue is busy: releasing the lock between
# the build and the run means queueing again behind whatever arrived meanwhile, and the build
# is worthless until something has been run against it. One model per invocation still holds -
# Gemma 4 is 17 GB and Qwen 3.6 is 29.3 GB on a 32 GB machine.
#
# The lock is taken by THIS process and the work happens in THIS process, because the lock
# file records a PID and a script that takes the lock and exits leaves a lock nobody holds.

param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Prompt = 'The capital of France is Paris. The capital of Japan is',
    [int]$Tokens = 24,
    [int]$DecodeCheck = 6,
    [int]$Threads = 8,
    [int]$TimeoutMin = 240,
    [int]$LongTokens = 0,
    [string]$LongPrompt = 'D:/MemeX/results/long_prompt.txt',
    [switch]$SkipBuild
)

. C:/Users/User11/Desktop/MemeX/bench/lock.ps1

$exe    = 'D:/MemeX/src/ik_llama.cpp/build/bin/Release/llama-memex-fwd.exe'
$outdir = 'D:/MemeX/results'
$tag    = [IO.Path]::GetFileNameWithoutExtension($Model)
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

if (-not (Test-Path -LiteralPath $Model)) { Write-Host "net modeli: $Model"; exit 4 }

# Source integrity BEFORE taking the machine: it costs a second, needs nothing exclusive, and
# there is no point queueing for a compile that a scripted edit has already broken.
Write-Host "--- check_source.py ---"
& python D:/MemeX/src/ik_llama.cpp/examples/memex-fwd/check_source.py
if ($LASTEXITCODE -ne 0) { Write-Host "check_source failed; not queueing"; exit 2 }

Write-Host ("lock holder before: " + (Get-LockHolder))
Write-Host "waiting for the machine..."
$t0 = Get-Date
if (-not (Take-Machine -Who 'arch' -TimeoutMin $TimeoutMin)) {
    Write-Host "NE POLUCHIL MASHINU za $TimeoutMin min"
    exit 3
}
Write-Host ("got the machine after {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)

$code = 0
try {
    if (-not $SkipBuild) {
        Write-Host "=== build ==="
        $b0 = Get-Date
        Push-Location D:/MemeX/src/ik_llama.cpp/build
        & cmake --build . --config Release --target llama-memex-fwd -- /nologo /v:m 2>&1 |
            Tee-Object -FilePath (Join-Path $outdir 'memex_fwd_build.log')
        $bc = $LASTEXITCODE
        Pop-Location
        Write-Host ("build exit {0} after {1:N1} min" -f $bc, ((Get-Date) - $b0).TotalMinutes)
        if ($bc -ne 0) { throw "build failed" }
    }

    $o = Get-CimInstance Win32_OperatingSystem
    Write-Host ("free memory: {0:N1} GB of {1:N1} GB" -f ($o.FreePhysicalMemory / 1MB),
                ($o.TotalVisibleMemorySize / 1MB))

    # ---- prefill, with the per-layer comparison against the reference's own node names
    $log1 = Join-Path $outdir "verify_${tag}_prefill.log"
    Write-Host "=== prefill comparison -> $log1 ==="
    & $exe -m $Model -p $Prompt --tokens $Tokens -t $Threads --no-repack --probe all 2>&1 |
        Tee-Object -FilePath $log1
    $c1 = $LASTEXITCODE
    Write-Host "prefill exit $c1"

    # ---- decode, step by step. The check that matters: a prefill carries nothing between
    # steps, so it cannot see a fault in state that is carried.
    if ($c1 -eq 0 -or $c1 -eq 2) {
        $log2 = Join-Path $outdir "verify_${tag}_decode.log"
        Write-Host "=== decode-check ($DecodeCheck steps) -> $log2 ==="
        & $exe -m $Model -p $Prompt --tokens $Tokens -t $Threads --no-repack `
               --decode-check $DecodeCheck 2>&1 | Tee-Object -FilePath $log2
        Write-Host "decode-check exit $LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE }
    } else {
        Write-Host "prefill failed; not running the decode check"
        $code = $c1
    }

    # ---- the sliding window, if asked for. A 24-token prompt never crosses a 1024 window,
    # so without this the windowed path is exercised structurally but its boundary never is.
    if ($LongTokens -gt 0 -and ($c1 -eq 0 -or $c1 -eq 2)) {
        if (Test-Path -LiteralPath $LongPrompt) {
            $log3 = Join-Path $outdir "verify_${tag}_window.log"
            Write-Host "=== window crossing ($LongTokens tokens) -> $log3 ==="
            & $exe -m $Model -f $LongPrompt --tokens $LongTokens -t $Threads --no-repack 2>&1 |
                Tee-Object -FilePath $log3
            Write-Host "window test exit $LASTEXITCODE"
        } else {
            Write-Host "net dlinnogo prompta: $LongPrompt"
        }
    }
} catch {
    Write-Host ("ABORT: " + $_.Exception.Message)
    if ($code -eq 0) { $code = 1 }
} finally {
    Free-Machine
    Write-Host ("lock holder after: " + (Get-LockHolder))
}
exit $code
