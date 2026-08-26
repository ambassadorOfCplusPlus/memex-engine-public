# Take the machine lock, build llama-memex-fwd, release the lock. One process from start to
# finish, deliberately: the lock file records a PID and Test-LockAlive checks that the PID is
# still running, so a script that takes the lock and then exits leaves a lock nobody holds.
# Taking it in one process and doing the work in the same process is the only correct shape.
#
# A compile is exactly the work this lock exists to serialise - a reference measurement was
# once taken at 9.95 tok/s with a 34% spread while cl.exe burned cores for somebody's build.
#
# Frees the lock on every path, including a failed build, so that a fix-and-retry cycle
# queues behind other work politely instead of holding the machine across an edit.

param(
    [int]$TimeoutMin = 180,
    [string]$Who = 'arch'
)

. C:/Users/User11/Desktop/MemeX/bench/lock.ps1

Write-Host ("lock holder before: " + (Get-LockHolder))
Write-Host "waiting for the machine..."
$t0 = Get-Date
if (-not (Take-Machine -Who $Who -TimeoutMin $TimeoutMin)) {
    Write-Host "NE POLUCHIL MASHINU za $TimeoutMin min"
    exit 3
}
Write-Host ("got the machine after {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)

$code = 0
try {
    # The source-integrity check first. It costs a second and it catches the two things a
    # scripted edit does that a diff review does not show - see check_source.py.
    Write-Host "--- check_source.py ---"
    & python D:/MemeX/src/ik_llama.cpp/examples/memex-fwd/check_source.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "check_source failed; not building"
        $code = 2
    } else {
        Write-Host "--- build ---"
        $b0 = Get-Date
        Push-Location D:/MemeX/src/ik_llama.cpp/build
        # /nologo and minimal verbosity: the interesting output is the error list, and a full
        # MSBuild log buries it in a thousand lines of "up to date".
        & cmake --build . --config Release --target llama-memex-fwd -- /nologo /v:m 2>&1 |
            Tee-Object -FilePath D:/MemeX/results/memex_fwd_build.log
        $bc = $LASTEXITCODE
        Pop-Location
        Write-Host ("build exit {0} after {1:N1} min" -f $bc, ((Get-Date) - $b0).TotalMinutes)
        if ($bc -ne 0) { $code = 1 }
    }
} finally {
    Free-Machine
    Write-Host ("lock holder after: " + (Get-LockHolder))
}
exit $code
