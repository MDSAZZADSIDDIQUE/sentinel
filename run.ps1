<#
.SYNOPSIS
  SENTINEL stage runner for Windows (no `make` required).

.DESCRIPTION
  Sets the runtime env (duplicate-OpenMP workaround) and dispatches to the
  `sentinel` Typer CLI. Friendly composite targets mirror the Makefile.

.EXAMPLE
  .\run.ps1 verify-data
  .\run.ps1 phase1            # to-parquet -> resolve-itemids -> cohort -> labels -> report
  .\run.ps1 to-parquet --mode dev
#>
param(
  [Parameter(Position = 0)]
  [string]$Target = "help",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Rest
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"

function Invoke-Sentinel { param([string[]]$CliArgs) & sentinel @CliArgs; if ($LASTEXITCODE -ne 0) { throw "sentinel $($CliArgs -join ' ') failed ($LASTEXITCODE)" } }

switch ($Target) {
  "setup"        { python -m pip install -e . --no-deps }
  "verify-data"  { Invoke-Sentinel (@("verify-data") + $Rest) }
  "to-parquet"   { Invoke-Sentinel (@("to-parquet") + $Rest) }
  "itemids"      { Invoke-Sentinel (@("resolve-itemids") + $Rest) }
  "cohort"       { Invoke-Sentinel (@("build-cohort") + $Rest) }
  "labels"       { Invoke-Sentinel (@("build-labels") + $Rest) }
  "report"       { Invoke-Sentinel (@("cohort-report") + $Rest) }
  "test"         { python -m pytest @Rest }
  "phase1" {
    # dependency order: itemids + cohort first, then filter big event tables,
    # then derive labels, then report.
    Invoke-Sentinel @("resolve-itemids")
    Invoke-Sentinel @("build-cohort")
    Invoke-Sentinel @("to-parquet")
    Invoke-Sentinel @("build-labels")
    Invoke-Sentinel @("cohort-report")
  }
  "dashboard"    { streamlit run dashboard/app.py @Rest }
  "help" {
    Write-Host "SENTINEL runner. Targets:" -ForegroundColor Cyan
    Write-Host "  setup verify-data to-parquet itemids cohort labels report test phase1 dashboard"
    Write-Host "Anything else is passed straight to the 'sentinel' CLI."
  }
  default        { Invoke-Sentinel (@($Target) + $Rest) }
}
