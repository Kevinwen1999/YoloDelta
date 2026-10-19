param(
    [switch]$Broad,
    [string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
$portRoot = Split-Path $PSScriptRoot -Parent
if (!$OutputRoot) {
    $OutputRoot = Join-Path $portRoot ('benchmark-results/closed-loop-suite-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
}
if (Test-Path -LiteralPath $OutputRoot) { throw "Suite output already exists: $OutputRoot" }
if (![System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path (Get-Location) $OutputRoot
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$quick = @(
    @{ Name='clean-1500'; Scenario='clean'; Speed=1500; Scale=1.0; Cadence=8; Frames=100 },
    @{ Name='fast-6000'; Scenario='clean'; Speed=6000; Scale=1.0; Cadence=8; Frames=150 },
    @{ Name='occlusion-1500'; Scenario='occlusion'; Speed=1500; Scale=1.0; Cadence=8; Frames=100 },
    @{ Name='crossing-1500'; Scenario='crossing'; Speed=1500; Scale=1.0; Cadence=8; Frames=100 }
)
$extra = @(
    @{ Name='small-3000'; Scenario='clean'; Speed=3000; Scale=0.5; Cadence=8; Frames=150 },
    @{ Name='large-3000'; Scenario='clean'; Speed=3000; Scale=1.5; Cadence=8; Frames=150 },
    @{ Name='slow-cadence'; Scenario='clean'; Speed=1500; Scale=1.0; Cadence=16; Frames=100 },
    @{ Name='fast-cadence'; Scenario='clean'; Speed=1500; Scale=1.0; Cadence=4; Frames=150 },
    @{ Name='dropout-1500'; Scenario='dropout'; Speed=1500; Scale=1.0; Cadence=8; Frames=100 },
    @{ Name='shake-1500'; Scenario='shake'; Speed=1500; Scale=1.0; Cadence=8; Frames=100 }
)
$cases = if ($Broad) { $quick + $extra } else { $quick }
$candidate = Join-Path $PSScriptRoot 'motion-identity-config.json'
$rows = [System.Collections.Generic.List[object]]::new()
$firstRun = $true
foreach ($case in $cases) {
    foreach ($variant in @('baseline','motion-identity')) {
        $output = Join-Path $OutputRoot ($case.Name + '-' + $variant)
        $argsForRun = @{
            Frames=$case.Frames; Speed=$case.Speed; Scale=$case.Scale;
            CadenceMs=$case.Cadence; InferenceDelayMs=[math]::Min(5,$case.Cadence);
            Scenario=$case.Scenario; Output=$output
        }
        if ($variant -eq 'motion-identity') { $argsForRun.Config = $candidate }
        if (!$firstRun) { $argsForRun.SkipBuildAndTests = $true }
        & (Join-Path $PSScriptRoot 'run-closed-loop.ps1') @argsForRun
        $firstRun = $false
        if ($LASTEXITCODE) { throw "Closed-loop run failed: $($case.Name) $variant" }
        $result = Get-Content -LiteralPath (Join-Path $output 'closed-loop.json') -Raw | ConvertFrom-Json
        if (!$result.validity.valid) { throw "Invalid closed-loop run: $output" }
        if (!$result.correctness.passed) { throw "Closed-loop correctness checks failed or are missing: $output" }
        $rows.Add([pscustomobject]@{
            Case=$case.Name; Variant=$variant; Scenario=$case.Scenario;
            SpeedPxS=$case.Speed; Scale=$case.Scale; CadenceMs=$case.Cadence;
            Frames=$result.summary.frames;
            DetectionFrames=$result.summary.frames_with_detections;
            OutputReadyCoverage=$result.summary.output_ready_coverage;
            FramesWithinTarget=$result.summary.fraction_frames_within_target_region;
            MeanNormalizedError=$result.summary.mean_normalized_error_all_frames;
            WrongTargetMovements=$result.summary.movements_with_wrong_selected_target;
            MovementsPastLossDeadline=$result.summary.movements_after_loss_deadline;
            InferenceP95Ms=$result.summary.gpu_inference_wall_ms.p95;
            Result=Join-Path $output 'closed-loop.json'
        })
    }
}
$summaryPath = Join-Path $OutputRoot 'suite.json'
$rows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding utf8
& (Join-Path $portRoot 'tools/write-closed-loop-comparison.ps1') -SuiteRoot $OutputRoot
$rows | Format-Table Case,Variant,DetectionFrames,OutputReadyCoverage,FramesWithinTarget,WrongTargetMovements -AutoSize
Write-Host "Suite summary: $summaryPath"
