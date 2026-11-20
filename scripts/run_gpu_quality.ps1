# Distribution quality at a valid sample size, on the laptop GPU, with the same
# weights, steps and resolution as the browser profiles. Phone timings stay separate.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_gpu_quality.ps1 -Count 1000
param(
  [int]$Count = 1000,
  [string]$Python = "python",
  [string]$Out = ".runs/gpu-quality-" + (Get-Date -Format "yyyyMMdd-HHmm")
)
$ErrorActionPreference = "Stop"
if (Test-Path $Out) { throw "Use a new output directory: $Out already exists." }
& $Python -m p3.fetch_coco --split validation --count $Count --out "$Out/data"
if ($LASTEXITCODE -ne 0) { throw "COCO download failed." }
$manifest = "$Out/data/manifest.jsonl"
$variants = @("browser-ref20", "browser-balanced12", "browser-eco6", "browser-preview4")
foreach ($v in $variants) {
  & $Python -m p3.benchmark generate --manifest $manifest --variant $v --out "$Out/runs" `
    --adapter p3/weights/t38-coco.safetensors
  if ($LASTEXITCODE -ne 0) { throw "Generation for $v failed." }
}
$runs = $variants | ForEach-Object { "$Out/runs/$_" }
& $Python -m p3.benchmark score --manifest $manifest --run @runs --fid
$pairs = $variants | ForEach-Object { "$_=$Out/runs/$_" }
Write-Host "`nAdd these to the Phase 1 report with:"
Write-Host "  python scripts/phase1_report.py --cost <phase1>/cost.json --scored ... --gpu $($pairs -join ' ') --out <new dir>"
