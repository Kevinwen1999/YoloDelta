param([Parameter(Mandatory=$true)][string]$Recording)
$ErrorActionPreference='Stop'
$run=Get-Content -LiteralPath $Recording -Raw | ConvertFrom-Json
if($run.mode -ne 'visible' -or $run.scene.scenario -ne 'linear' -or !$run.validity.valid) {
    throw 'Use a valid visible constant-motion recording.'
}
$first=$run.frames[0]
$x=$first.target_center[0];$y=$first.target_center[1];$crop=$first.capture
if($x -lt $crop[0] -or $x -ge $crop[0]+$crop[2] -or $y -lt $crop[1] -or $y -ge $crop[1]+$crop[3]) {
    throw "Scene advanced outside its initial capture region before observation: t=$($first.render_ms)ms"
}
if($run.capture_hardware.motion_clock_start -ne 'after capture warm-up and scene window setup' -or
    $null -eq $run.capture_hardware.capture_warmup_ms) {
    throw 'Recording must identify capture warm-up as setup outside the world motion clock.'
}
Write-Output "Visible startup passed: first world update=$($first.render_ms)ms, setup capture=$($run.capture_hardware.capture_warmup_ms)ms"
