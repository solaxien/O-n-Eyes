# Phase 1: measure the current source on the phone, then score it offline.
# Run from the repository root on the laptop with the Poco F6 connected by USB.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1 `
#     -Origin https://on-eyes-device-lab-t38.brawnymoss1024.chatgpt.site -Tab <tab-id> -Adb <path-to-adb.exe>
#
# Add -SelectionStudy to also measure Balanced with the verifier and retry,
# and Eco with quality hints. Each output directory is new; nothing is overwritten.
param(
  [Parameter(Mandatory = $true)][string]$Origin,
  [Parameter(Mandatory = $true)][string]$Tab,
  [Parameter(Mandatory = $true)][string]$Adb,
  [string]$Python = "python",
  [string]$Manifest = "data/android-composition-v2.jsonl",
  [string]$Out = ".runs/phase1-" + (Get-Date -Format "yyyyMMdd-HHmm"),
  [switch]$SelectionStudy
)
$ErrorActionPreference = "Stop"
if (Test-Path $Out) { throw "Use a new output directory: $Out already exists." }
New-Item -ItemType Directory -Path $Out | Out-Null

function Invoke-Run([string]$Name, [string]$Mode, [string]$Verify, [string]$Hints) {
  Write-Host "`n=== $Name ($Mode, verify=$Verify, quality-hints=$Hints) ==="
  node scripts/benchmark_browser.cjs --origin $Origin --tab $Tab --out "$Out/$Name" `
    --manifest $Manifest --adb $Adb --mode $Mode --warmup --keep-loaded --python $Python `
    --verify $Verify --quality-hints $Hints
  if ($LASTEXITCODE -ne 0) { throw "Phone run $Name failed." }
  & $Python -m p3.import_browser_run --source "$Out/$Name" --out "$Out/scored-$Name" --kind enhanced
  if ($LASTEXITCODE -ne 0) { throw "Import of $Name failed." }
}

# Same prompts, seeds, model cache and prompt policy in every core row.
$core = @(
  @{ Name = "baseline"; Mode = "baseline" },
  @{ Name = "balanced"; Mode = "balanced" },
  @{ Name = "eco"; Mode = "eco" },
  @{ Name = "preview"; Mode = "preview" }
)
foreach ($run in $core) { Invoke-Run $run.Name $run.Mode "false" "false" }

$scored = $core | ForEach-Object { "$Out/scored-$($_.Name)" }
& $Python -m p3.benchmark score --manifest $Manifest --run @scored
if ($LASTEXITCODE -ne 0) { throw "Offline scoring failed." }
& $Python scripts/compare_browser_profiles.py `
  --run "$Out/baseline" --run "$Out/balanced" --run "$Out/eco" --run "$Out/preview" `
  --out "$Out/cost.json"
$pairs = $core | ForEach-Object { "$($_.Name)=$Out/scored-$($_.Name)" }
& $Python scripts/phase1_report.py --cost "$Out/cost.json" --scored @pairs --out "$Out/report"
# Classical CV judged by a metric it was not tuned on (needs opencv-contrib-python-headless).
foreach ($run in $core) {
  & $Python scripts/offline_iqa.py --run "$Out/$($run.Name)" --out "$Out/iqa-$($run.Name).json"
}

if ($SelectionStudy) {
  # Verifier-steered retry versus one image, and quality hints versus none.
  Invoke-Run "balanced-verify" "balanced" "true" "false"
  Invoke-Run "eco-hints" "eco" "false" "true"
  & $Python -m p3.benchmark score --manifest $Manifest --run "$Out/scored-balanced-verify" "$Out/scored-eco-hints"
  & $Python scripts/compare_browser_profiles.py --run "$Out/balanced" --run "$Out/balanced-verify" `
    --out "$Out/cost-selection.json"
  & $Python scripts/phase1_report.py --cost "$Out/cost-selection.json" `
    --scored "balanced=$Out/scored-balanced" "balanced-verify=$Out/scored-balanced-verify" `
    --out "$Out/report-selection"
  Write-Host "Quality hints: compare scored-eco and scored-eco-hints scores.json (CLIP, clauses) and view the images side by side."
}
Write-Host "`nDone. Paste $Out/report/phase1-results.md into eval_P3_T38.md and keep the run folder as evidence."
