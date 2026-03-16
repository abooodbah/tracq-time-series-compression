Param(
    [string]$InputCsv = "uci_electricity_sample.cleaned.csv",
    [string]$OutDir = "bench_results",
    [switch]$Force,
    [int]$GridtsBits = 8,
    [int]$GridtsPngLevel = 6
)

# PowerShell benchmark orchestrator for tracq vs gzip/parquet
# Usage from repo root:
#   .\scripts\run_benchmarks.ps1 -InputCsv uci_electricity_sample.cleaned.csv -OutDir bench_results

$ErrorActionPreference = 'Stop'
$cwd = Get-Location
$inputPath = Join-Path $cwd.Path $InputCsv
if (-not (Test-Path $inputPath)) {
    Write-Host "Input CSV not found: $inputPath" -ForegroundColor Red
    exit 2
}

# ensure output folder
$outDirPath = Join-Path $cwd.Path $OutDir
if (-not (Test-Path $outDirPath)) { New-Item -ItemType Directory -Path $outDirPath | Out-Null }

function Run-Measure {
    param($ScriptBlock)
    $result = Measure-Command { & $ScriptBlock }
    return $result.TotalSeconds
}

# 1) Run tracq compress (auto-tune) via CLI (kept for reference; labeled tracq_cli)
Write-Host "Running tracq compress (auto-tune) on $inputPath ..."
$forceFlag = if ($Force) { '--force' } else { '--force' }
$tracqCmd = "conda run -n gtc-env python -m tracq compress `"$PWD\\$InputCsv`" --auto-tune --max-rmse 0.05 $forceFlag"
Write-Host "Command: $tracqCmd"
$tracqTime = Measure-Command { iex $tracqCmd }
$tracqSec = $tracqTime.TotalSeconds
# output PNG path inferred by CLI: input.tracq.png
$pngPath = Join-Path $cwd.Path ([IO.Path]::GetFileNameWithoutExtension($InputCsv) + ".tracq.png")
if (-not (Test-Path $pngPath)) { Write-Host "tracq output not found: $pngPath" -ForegroundColor Yellow }
$pngSize = if (Test-Path $pngPath) { (Get-Item $pngPath).Length } else { 0 }
$origSize = (Get-Item $inputPath).Length
$tracqRatio = if ($origSize -ne 0) { [math]::Round($pngSize / $origSize, 4) } else { 0 }

# 2) Run Python benchmark helpers (gzip/parquet/zstd/brotli/tracq in-process)
Write-Host "Running gzip/parquet/zstd/brotli/tracq-inproc benchmarks (via Python helper)..."
$pyCmd = "conda run -n gtc-env python `"scripts/benchmark_timeseries.py`" --input `"$PWD\\$InputCsv`" --outdir `"$PWD\\$OutDir`" --tracq-bits $GridtsBits --tracq-png-level $GridtsPngLevel"
Write-Host "Command: $pyCmd"
$pyTime = Measure-Command { iex $pyCmd }
$pySec = $pyTime.TotalSeconds

# 3) Read results JSON and print consolidated table
$resultsJson = Join-Path $outDirPath ([IO.Path]::GetFileName($InputCsv) + '.benchmark.json')
$benchData = $null
if (Test-Path $resultsJson) {
    $benchData = Get-Content $resultsJson -Raw | ConvertFrom-Json
}

Write-Host "\n--- Summary ---"
Write-Host "Input CSV: $inputPath" 
Write-Host "Original size: $([math]::Round($origSize/1024,2)) KB ($origSize bytes)"
if (Test-Path $pngPath) { Write-Host "tracq_cli: $pngPath ($([math]::Round($pngSize/1024,2)) KB) time=${tracqSec}s ratio=${tracqRatio}" } else { Write-Host "tracq_cli: not found" }

if ($benchData -ne $null) {
    foreach ($k in $benchData.runs.PSObject.Properties.Name) {
        $r = $benchData.runs.$k
        $bytes = $r.bytes
        $ratio = [math]::Round($r.ratio,4)
        $time = [math]::Round($r.time_s,3)
        Write-Host "$k`t: $([math]::Round($bytes/1024,2)) KB ($bytes bytes) time=${time}s ratio=${ratio}"
    }
} else {
    Write-Host "No python benchmark results found at $resultsJson" -ForegroundColor Yellow
}

Write-Host "\nDetailed results written to: $outDirPath"

# Save a small CSV summary
$summaryPath = Join-Path $outDirPath 'summary.csv'
@(
    'method,bytes,time_s,ratio',
    ("tracq_cli,$pngSize,$tracqSec,$tracqRatio")
) | Out-File -FilePath $summaryPath -Encoding utf8
if ($benchData -ne $null) {
    foreach ($k in $benchData.runs.PSObject.Properties.Name) {
        $r = $benchData.runs.$k
        Add-Content -Path $summaryPath -Value ("$k,$($r.bytes),$($r.time_s),$([math]::Round($r.ratio,4))")
    }
}

Write-Host "Saved summary to $summaryPath"
