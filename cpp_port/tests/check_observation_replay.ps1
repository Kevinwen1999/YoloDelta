param([Parameter(Mandatory=$true)][string]$Recording,[string]$OutputRoot)
$ErrorActionPreference='Stop'
$portRoot=Split-Path $PSScriptRoot -Parent
if(!$OutputRoot){$OutputRoot=Join-Path $portRoot ('benchmark-results/replay-check-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))}
if(Test-Path -LiteralPath $OutputRoot){throw 'Replay output directory already exists.'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null
$previousPath=$env:PATH
try {
    $cudaRoot=if($env:CUDA_PATH){$env:CUDA_PATH}else{'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4'}
    $env:PATH=(Join-Path $cudaRoot 'bin')+';'+$previousPath
    $exe=Join-Path $portRoot 'build-cuda-benchmark/Release/delta_closed_loop_gpu.exe'
    $rows=foreach($variant in @('original-a','original-b','baseline','stability-only')) {
        $output=Join-Path $OutputRoot ($variant+'.json')
        $arguments=@('--replay',$Recording,'--output',$output)
        if($variant -in @('baseline','stability-only')) {
            $override=Join-Path $OutputRoot ($variant+'-config.json')
            @{motion_stability_enable=($variant -eq 'stability-only');motion_identity_enable=$false}|ConvertTo-Json|Set-Content -LiteralPath $override -Encoding ascii
            $arguments+=@('--config',$override)
        }
        & $exe @arguments
        if($LASTEXITCODE){throw "Replay failed: $variant"}
        $result=Get-Content -LiteralPath $output -Raw|ConvertFrom-Json
        if($variant -like 'original-*' -and $result.different_outcomes -ne 0){throw 'Recorded production outcomes were not reproduced exactly.'}
        [pscustomobject]@{Variant=$variant;Frames=$result.frames.Count;Different=$result.different_outcomes;
            Ready=@($result.frames|Where-Object output_ready).Count;
            WrongIdentityReady=$result.wrong_identity_output_ready_observations;
            UnknownIdentityReady=$result.unknown_identity_output_ready_observations}
    }
    if((Get-FileHash (Join-Path $OutputRoot 'original-a.json')).Hash -ne (Get-FileHash (Join-Path $OutputRoot 'original-b.json')).Hash){throw 'Identical observation replays must be byte-for-byte deterministic.'}
    $rows|Format-Table
    Write-Host 'Exact observation replay passed. Alternative-policy results are open-loop diagnostics with recorded feedback, not closed-loop quality claims.'
} finally {$env:PATH=$previousPath}
