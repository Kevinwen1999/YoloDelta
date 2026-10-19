param(
    [Parameter(Mandatory=$true)][string]$Controlled,
    [Parameter(Mandatory=$true)][string]$Visible
)
$ErrorActionPreference='Stop'
$runs=@((Get-Content -LiteralPath $Controlled -Raw|ConvertFrom-Json),(Get-Content -LiteralPath $Visible -Raw|ConvertFrom-Json))
if($runs[0].mode -ne 'controlled' -or $runs[1].mode -ne 'visible'){throw 'Provide controlled and visible runs in that order.'}
if($runs[0].build.source_hash -ne $runs[1].build.source_hash -or $runs[0].gpu.name -ne $runs[1].gpu.name -or $runs[0].gpu.ordinal -ne $runs[1].gpu.ordinal){throw 'Camera comparison requires the same executable source and inference GPU.'}
$responses=foreach($run in $runs) {
    if(!$run.validity.valid -or !$run.correctness.passed -or !$run.scene.viewport){throw 'Camera check requires valid schema 5 evidence with viewport geometry.'}
    $scene=$run.scene
    $fx=$scene.viewport[0]/2/[math]::Tan($scene.horizontal_fov_deg*[math]::PI/360)
    $fy=$scene.viewport[1]/2/[math]::Tan($scene.vertical_fov_deg*[math]::PI/360)
    $yaw=0.0;$pitch=0.0
    if(!$run.applied_movements.Count){throw 'Camera check requires actually applied commands.'}
    foreach($move in $run.applied_movements) {
        $expectedYaw=$yaw+$move.dx*$scene.mouse_radians_per_count
        $expectedPitch=$pitch+$move.dy*$scene.mouse_radians_per_count
        if([math]::Abs($expectedYaw-$move.camera_yaw_rad) -gt 1e-9 -or [math]::Abs($expectedPitch-$move.camera_pitch_rad) -gt 1e-9){throw 'Delivered backend counts do not match recorded camera rotation.'}
        $yaw=$move.camera_yaw_rad;$pitch=$move.camera_pitch_rad
    }
    [pscustomobject]@{Mode=$run.mode;Width=$scene.viewport[0];Height=$scene.viewport[1];
        Horizontal10CountsPx=$fx*[math]::Tan(10*$scene.mouse_radians_per_count);
        Vertical10CountsPx=$fy*[math]::Tan(10*$scene.mouse_radians_per_count);
        AppliedCommands=$run.applied_movements.Count}
}
if([math]::Abs($responses[0].Horizontal10CountsPx-$responses[1].Horizontal10CountsPx) -gt 1e-6 -or
    [math]::Abs($responses[0].Vertical10CountsPx-$responses[1].Vertical10CountsPx) -gt 1e-6){throw 'Camera response differs between test modes; match viewport, FOV and sensitivity.'}
$responses|Format-Table
Write-Host 'Applied camera rotations and known-count response agree across modes. Physical application calibration is still required.'
