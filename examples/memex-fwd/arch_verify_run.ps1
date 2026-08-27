# Build once, then verify both architectures - all inside ONE lock acquisition.
#
# One acquisition, but never two models resident at once: each check is a separate process that
# is waited on to exit before the next one starts. Gemma 4 is 17 GB and Qwen 3.6 is 29.3 GB on a
# 32 GB machine, so overlap would swap; sequential processes under one lock do not.
#
# Every step has a hard timeout and the whole script has a deadline. A holder that outlives the
# work it was holding for keeps the machine locked with nothing left to use it - that has already
# happened here once. On timeout the child is killed, on any exit the lock is freed, and if this
# process is killed outright the lock file records a dead PID and the next claimant breaks it.

param(
    [int]$Threads      = 8,
    [int]$TimeoutMin   = 180,   # how long to WAIT for the machine
    [int]$BudgetMin    = 200,   # how long we may HOLD it once we have it
    [int]$BuildMin     = 60,
    [int]$StepMin      = 45,
    [int]$QwenStepMin  = 75,
    [int]$LongTokens   = 1100,
    [ValidateSet('all','short','window')][string]$Steps = 'all',
    # A length sweep over the SAME prompt, logits only. Its whole point is to separate two
    # explanations of a divergence at 1100 tokens that look identical at 1100 tokens:
    #   500  - one reference ubatch (n_ubatch is 512), no window crossing
    #   900  - two reference ubatches, still no window crossing
    #  1100  - three ubatches AND the 1024 window boundary crossed
    # Clean at 500 and 900 and broken at 1100 means the window. Broken at 900 means the
    # length or the reference's own batching, and the window is not implicated at all.
    # A comma-separated string rather than [int[]]: pwsh -File binds every parameter as a
    # string, so an array parameter takes only the first element and drops the rest silently.
    [string]$Sweep = '',
    # Gemma and Qwen in one acquisition is 17 GB then 29.3 GB on a 32 GB machine, and
    # the machine is shared. Splitting them lets the lock be given back between the two.
    [switch]$NoQwen,
    [switch]$SkipBuild
)

. C:/Users/User11/Desktop/MemeX/bench/lock.ps1

$exe    = 'D:/MemeX/src/ik_llama.cpp/build/bin/Release/llama-memex-fwd.exe'
$outdir = 'D:/MemeX/results'
$gemma  = 'D:/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf'
$qwen   = 'D:/Qwen3.6-35B-A3B-UD-Q6_K.gguf'
$long   = 'D:/MemeX/results/long_prompt_varied.txt'
$prompt = 'The capital of France is Paris. The capital of Japan is'

New-Item -ItemType Directory -Force -Path $outdir | Out-Null
foreach ($needed in @($gemma, $qwen, $long)) {
    if (-not (Test-Path -LiteralPath $needed)) { Write-Host "NET FAJLA: $needed"; exit 4 }
}

$script:child = $null

# One step: run the engine with a hard timeout, log it, return the exit code.
# -1 means it was killed on timeout, which is a failure of the step and not of the check.
function Invoke-Step {
    param([string]$Name, [string[]]$StepArgs, [int]$LimitMin)
    $log = Join-Path $outdir "verify_$Name.log"
    $err = "$log.err"
    Write-Host ""
    Write-Host "=== $Name (limit $LimitMin min) -> $log ==="
    Write-Host ("    " + ($StepArgs -join ' '))
    $t0 = Get-Date
    # Quote by hand. Start-Process -ArgumentList joins an array with spaces and quotes
    # nothing, so a prompt with spaces in it arrives at the engine as one word per space and
    # every word after the first is rejected as an unknown flag - which is exactly what
    # happened, in six seconds, before the model was even opened.
    $cmdline = ($StepArgs | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { "$_" }
    }) -join ' '
    $proc = Start-Process -FilePath $exe -ArgumentList $cmdline -NoNewWindow -PassThru `
                          -RedirectStandardOutput $log -RedirectStandardError $err
    # Touch .Handle before waiting, or .ExitCode comes back EMPTY - not zero, not an error,
    # empty. Start-Process -PassThru hands back a Process object with no cached OS handle,
    # and once the process exits there is nothing left to read the code from; reading .Handle
    # first is what makes the object keep it. Measured here rather than assumed: the same
    # cmake --version returns [] without this line and [0] with it, and a deliberately failing
    # command returns [1].
    #
    # This is not cosmetic. $null -ne 0 is TRUE, so every caller comparing the result against
    # zero declares failure on a step that succeeded - which is exactly what happened: a build
    # that produced the exe reported "sborka upala" and threw away the lock it had waited
    # 5.5 minutes for.
    #
    # $proc and not $p: METHODS 45. A single-letter local shadowing a script variable of the
    # same letter (case-insensitively) has cost this project two sweeps.
    $null = $proc.Handle
    $script:child = $proc
    if (-not $proc.WaitForExit($LimitMin * 60 * 1000)) {
        Write-Host "TAJM-AUT $LimitMin min: ubivaju"
        try { $proc.Kill($true) } catch { }
        try { $proc.WaitForExit(30000) | Out-Null } catch { }
        $script:child = $null
        return -1
    }
    $script:child = $null
    $code = $proc.ExitCode
    if ($null -eq $code) { Write-Host "    (ExitCode pust - schitaem otkazom)"; $code = -2 }
    Write-Host ("    exit {0} za {1:N1} min" -f $code, ((Get-Date) - $t0).TotalMinutes)
    return $code
}

Write-Host "--- check_source.py ---"
& python D:/MemeX/src/ik_llama.cpp/examples/memex-fwd/check_source.py
if ($LASTEXITCODE -ne 0) { Write-Host "check_source upal; v ochered ne vstajom"; exit 2 }

Write-Host ("lock holder before: " + (Get-LockHolder))
Write-Host "zhdjom mashinu..."
$tw = Get-Date
if (-not (Take-Machine -Who 'arch' -TimeoutMin $TimeoutMin)) {
    Write-Host "NE POLUCHIL MASHINU za $TimeoutMin min"
    exit 3
}
Write-Host ("vzjali mashinu cherez {0:N1} min" -f ((Get-Date) - $tw).TotalMinutes)
$deadline = (Get-Date).AddMinutes($BudgetMin)

$code = 0
try {
    if (-not $SkipBuild) {
        # Force the two libraries to RE-LINK, every time, by removing their link tlogs.
        #
        # ggml.dll and llama.dll have now vanished from build/bin/Release three times, and the
        # timestamps finally say what is going on. After the 08:30 build the log contained
        # `ggml.vcxproj -> ...\bin\Release\ggml.dll` and the dll was not there - while
        # ggml.exp, llama.exp and all four link.*.tlog still carried 08:16, the time of the
        # previous forced relink. Only ggml.lastbuildstate and Cl.items.tlog were rewritten.
        # So MSBuild visited the target, judged the link up to date from its tlogs, printed the
        # arrow anyway, and linked nothing. The arrow in a build log is not evidence of a link.
        #
        # What removes them is still unnamed, but the correlation is: a build in ANOTHER
        # ik_llama tree. night.ps1 already carries the note in the other direction ("building
        # targets in `build` has been observed to remove ggml.dll from build-vk's output
        # directory"); this time a full build of ik_upstream emptied ours. Two trees, both
        # producing a file called ggml.dll, and the loser is whichever one is not building.
        #
        # Removing link.* costs one relink from object files that are already there - measured
        # at twelve seconds - and needs no --clean-first, which rebuilds the iqk kernels for
        # half an hour. Doing it unconditionally is cheaper than detecting when it is needed.
        foreach ($tl in @('D:/MemeX/src/ik_llama.cpp/build/ggml/src/ggml.dir/Release/ggml.tlog',
                          'D:/MemeX/src/ik_llama.cpp/build/src/llama.dir/Release/llama.tlog')) {
            Get-ChildItem -Path (Join-Path $tl 'link.*') -ErrorAction SilentlyContinue |
                Remove-Item -Force -ErrorAction SilentlyContinue
        }
        # ONE target. Naming three (ggml, llama, llama-memex-fwd) made the log create
        # ggml.lib and ggml.exp three times in a single build - once as the explicit target
        # and twice as a dependency - and twice out of three runs bin/Release ended up with
        # no ggml.dll while the .lib and .exp carried the fresh timestamp and MSBuild exited
        # zero. Two links racing for one output can leave no output. llama-memex-fwd pulls
        # ggml and llama exactly once each, and the tlog wipe above still forces both to
        # relink, which is the part that was actually needed.
        Write-Host "=== build (link tlogs cleared, odna cel) ==="
        $b0 = Get-Date
        $blog = Join-Path $outdir 'memex_fwd_build.log'
        $bp = Start-Process -FilePath 'cmake' `
              -ArgumentList @('--build','D:/MemeX/src/ik_llama.cpp/build','--config','Release',
                              '--target','llama-memex-fwd','--','/nologo','/v:m') `
              -NoNewWindow -PassThru -RedirectStandardOutput $blog -RedirectStandardError "$blog.err"
        $null = $bp.Handle          # see the note in Invoke-Step; without it ExitCode is empty
        $script:child = $bp
        if (-not $bp.WaitForExit($BuildMin * 60 * 1000)) {
            try { $bp.Kill($true) } catch { }
            throw "sborka ne uspela za $BuildMin min"
        }
        $script:child = $null
        $bcode = $bp.ExitCode
        if ($null -eq $bcode) { $bcode = -2 }
        Write-Host ("build exit {0} za {1:N1} min" -f $bcode, ((Get-Date) - $b0).TotalMinutes)
        if ($bcode -ne 0) {
            Get-Content -LiteralPath "$blog.err" -Tail 40 -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Host ("    ! " + $_) }
            Get-Content -LiteralPath $blog -Tail 60 -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Host ("    | " + $_) }
            throw "sborka upala"
        }
    }
    if (-not (Test-Path -LiteralPath $exe)) { throw "net exe posle sborki" }
    Write-Host ("exe: " + (Get-Item -LiteralPath $exe).LastWriteTime)

    # Runnability, not existence. Test-Path answers a question nobody asked: ggml.dll has now
    # disappeared from bin/Release three separate times in this project while cmake reported
    # the target up to date, and the only symptom is every run dying with -1073741515 before
    # it prints anything - which reads like "the model failed" rather than "the build is
    # broken". --help touches no model and costs nothing.
    #   -1073741515 = STATUS_DLL_NOT_FOUND, a dll is missing next to the exe
    #   -1073741511 = STATUS_ENTRYPOINT_NOT_FOUND, dll and exe are from different builds
    $smoke = Start-Process -FilePath $exe -ArgumentList '--help' -NoNewWindow -PassThru `
                           -RedirectStandardOutput (Join-Path $outdir 'smoke.log') `
                           -RedirectStandardError  (Join-Path $outdir 'smoke.log.err')
    $null = $smoke.Handle
    $null = $smoke.WaitForExit(60000)
    $sc = $smoke.ExitCode
    Write-Host ("smoke --help exit: " + $sc)
    if ($sc -ne 0) {
        $why = switch ($sc) {
            -1073741515 { 'STATUS_DLL_NOT_FOUND - ryadom s exe net ggml.dll ili llama.dll' }
            -1073741511 { 'STATUS_ENTRYPOINT_NOT_FOUND - dll i exe iz raznyh sborok' }
            default     { 'neizvestnaja prichina' }
        }
        Write-Host "binarnik ne zapuskaetsja (exit $sc): $why - probuju perelinkovat odin raz"
        foreach ($tl in @('D:/MemeX/src/ik_llama.cpp/build/ggml/src/ggml.dir/Release/ggml.tlog',
                          'D:/MemeX/src/ik_llama.cpp/build/src/llama.dir/Release/llama.tlog')) {
            Get-ChildItem -Path (Join-Path $tl 'link.*') -ErrorAction SilentlyContinue |
                Remove-Item -Force -ErrorAction SilentlyContinue
        }
        $rp = Start-Process -FilePath 'cmake' `
              -ArgumentList @('--build','D:/MemeX/src/ik_llama.cpp/build','--config','Release',
                              '--target','llama-memex-fwd','--','/nologo','/v:m') `
              -NoNewWindow -PassThru `
              -RedirectStandardOutput (Join-Path $outdir 'repair_build.log') `
              -RedirectStandardError  (Join-Path $outdir 'repair_build.log.err')
        $null = $rp.Handle
        $null = $rp.WaitForExit(20 * 60 * 1000)
        Write-Host ("repair build exit: " + $rp.ExitCode)
        $smoke2 = Start-Process -FilePath $exe -ArgumentList '--help' -NoNewWindow -PassThru `
                               -RedirectStandardOutput (Join-Path $outdir 'smoke.log') `
                               -RedirectStandardError  (Join-Path $outdir 'smoke.log.err')
        $null = $smoke2.Handle
        $null = $smoke2.WaitForExit(60000)
        Write-Host ("smoke posle remonta: " + $smoke2.ExitCode)
        if ($smoke2.ExitCode -ne 0) { throw "binarnik ne zapuskaetsja i posle remonta: $why" }
    }

    $o = Get-CimInstance Win32_OperatingSystem
    Write-Host ("svobodno pamjati: {0:N1} GB iz {1:N1} GB" -f ($o.FreePhysicalMemory / 1MB),
                ($o.TotalVisibleMemorySize / 1MB))

    # ---------------- Gemma 4 ----------------
    #
    # --probe all and --decode-check in ONE process, because the engine already does both in
    # that order out of a single load: the prefill comparison layer by layer under the
    # reference's own node names, then the step-by-step decode with the same per-layer
    # comparison at every step. Splitting them into two invocations would buy nothing and
    # cost a second two-minute load of a 17 GB file - and, on qwen, of a 29 GB one.
    if ($Steps -ne 'window') {
        $g1 = Invoke-Step -Name 'gemma4_short' -LimitMin $StepMin -StepArgs @(
            '-m', $gemma, '-p', $prompt, '--tokens', '24', '-t', "$Threads", '--no-repack',
            '--probe', 'all', '--decode-check', '6')
        # exit 2 is the engine saying "it ran and the numbers disagree" - a result, not a
        # broken step. Only anything else means the step itself did not happen.
        if ($g1 -eq 2) { Write-Host "gemma korotkij progon: RASHOZHDENIE (exit 2)"; $code = 10 }
        elseif ($g1 -ne 0) { Write-Host "gemma korotkij progon ne proshjol (exit $g1)"; $code = 11 }

        # The same prompt and the same binary, with ONE thing changed: whether the reference
        # runs flash attention. Both arms in one acquisition, because the whole value of the
        # pair is that nothing else can differ between them.
        #
        # Why it is a pair and not a replacement. With flash_attn off the reference's gemma4 V
        # cache is written through ggml_transpose + a flat ggml_cpy, and gemma4 is the one
        # architecture that hands that code a 3-D V - so store and read disagree and the
        # reference is wrong. With it on, the store is flat and the read matches. If our
        # numbers collapse in the second arm and not the first, the fault was never ours.
        $g2 = Invoke-Step -Name 'gemma4_short_reffa' -LimitMin $StepMin -StepArgs @(
            '-m', $gemma, '-p', $prompt, '--tokens', '24', '-t', "$Threads", '--no-repack',
            '--ref-fa', '--probe', 'all', '--decode-check', '6')
        if ($g2 -eq 2) { Write-Host "gemma --ref-fa: RASHOZHDENIE (exit 2)" }
        elseif ($g2 -ne 0) { Write-Host "gemma --ref-fa ne proshjol (exit $g2)" }
        else { Write-Host "gemma --ref-fa: SOSHLOS (exit 0)"; $code = 0 }
    }

    $sweepLens = @()
    if ($Sweep -ne '') { $sweepLens = @($Sweep -split ',' | ForEach-Object { $_.Trim() }) }
    foreach ($entry in $sweepLens) {
        if ((Get-Date) -ge $deadline) { Write-Host "bjudzhet ischerpan: sweep $entry propushchen"; break }
        # "N" or "N:U" - U forces the reference's micro-batch, so the same length can be run
        # with the reference split into several micro-batches and with it not split at all.
        $parts = $entry -split ':'
        $len   = [int]$parts[0]
        $name  = "gemma4_len$len"
        # Logits only: no --probe (with several micro-batches the reference captures only its
        # first, so per-layer tensors are 512 rows against our N and compare against nothing)
        # and no decode check.
        $a = @('-m', $gemma, '-f', $long, '--tokens', "$len", '-t', "$Threads", '--no-repack')
        if ($parts.Count -gt 1) { $a += @('--ref-ubatch', $parts[1]); $name = "${name}_ub$($parts[1])" }
        $s = Invoke-Step -Name $name -LimitMin $StepMin -StepArgs $a
        if ($s -ne 0) { Write-Host "sweep $entry ne proshjol" }
    }

    if ($Steps -ne 'short' -and (Get-Date) -lt $deadline) {
        # The window boundary. With a 24-token prompt every query still sees position zero,
        # so 1024-wide windowing is exercised structurally and its boundary never is: an
        # off-by-one between `pos - j < n_swa` and `<=` passes that test silently.
        # The decode steps on top cross the boundary on the CACHED path too, which is the
        # path generation actually runs.
        # --ref-ubatch matters here and only here. n_ubatch is 512 by default, so an 1100
        # token prompt makes llama_decode run three micro-batches and capture the layer
        # tensors of one of them: 512 rows against our 1100, which the comparison then
        # refuses on size and prints as "ne sravnivaju" for all 120 nodes. Forcing one
        # micro-batch is what makes --probe all mean anything at this length.
        $g3 = Invoke-Step -Name 'gemma4_window' -LimitMin $StepMin -StepArgs @(
            '-m', $gemma, '-f', $long, '--tokens', "$LongTokens", '-t', "$Threads",
            '--no-repack', '--ref-ubatch', "$LongTokens",
            '--probe', 'all', '--decode-check', '4')
        if ($g3 -eq 2) { Write-Host "okonnyj progon: RASHOZHDENIE (exit 2), povtor ne nuzhen"; $code = 12 }
        elseif ($g3 -ne 0) {
            # Capturing every layer of an 1100-token prefill on both sides is a couple of
            # gigabytes. If that is what failed, the boundary check itself is still worth
            # having, so retry it without the per-layer capture.
            Write-Host "okonnyj progon s --probe all ne proshjol; povtorjaem bez probe"
            $g3b = Invoke-Step -Name 'gemma4_window_noprobe' -LimitMin $StepMin -StepArgs @(
                '-m', $gemma, '-f', $long, '--tokens', "$LongTokens", '-t', "$Threads",
                '--no-repack', '--decode-check', '4')
            if ($g3b -ne 0) { $code = 12 }
        }
    } else { Write-Host "bjudzhet vremeni ischerpan: okonnyj progon propushchen" }

    # ---------------- Qwen 3.6 ----------------
    # Separate process, started only after every gemma process has exited.
    if ($NoQwen) { Write-Host 'qwen propushchen: -NoQwen' }
    elseif ($Steps -ne 'window' -and (Get-Date).AddMinutes($QwenStepMin) -lt $deadline) {
        [GC]::Collect()
        $o = Get-CimInstance Win32_OperatingSystem
        Write-Host ("svobodno pamjati pered qwen: {0:N1} GB" -f ($o.FreePhysicalMemory / 1MB))
        # The decode half is the one that matters here: a prefill carries nothing between
        # steps, and thirty delta-net layers carry a 128x128 matrix per head.
        $q1 = Invoke-Step -Name 'qwen35_short' -LimitMin $QwenStepMin -StepArgs @(
            '-m', $qwen, '-p', $prompt, '--tokens', '24', '-t', "$Threads", '--no-repack',
            '--probe', 'all', '--decode-check', '6')
        if ($q1 -eq 2) { Write-Host "qwen korotkij progon: RASHOZHDENIE (exit 2)"; if ($code -eq 0) { $code = 20 } }
        elseif ($q1 -ne 0) { Write-Host "qwen korotkij progon ne proshjol (exit $q1)"; if ($code -eq 0) { $code = 21 } }
    } else { Write-Host "bjudzhet vremeni ischerpan: qwen propushchen celikom" }
} catch {
    Write-Host ("ABORT: " + $_.Exception.Message)
    if ($code -eq 0) { $code = 1 }
} finally {
    if ($null -ne $script:child) { try { $script:child.Kill($true) } catch { } }
    Free-Machine
    Write-Host ("lock holder after: " + (Get-LockHolder))
}
Write-Host "ITOG exit $code"
exit $code
