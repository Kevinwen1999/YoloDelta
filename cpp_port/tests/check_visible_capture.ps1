param([string]$OutputRoot)
$ErrorActionPreference = 'Stop'
$portRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path $portRoot -Parent
if (!$OutputRoot) { $OutputRoot = Join-Path $portRoot ('benchmark-results/visible-obstruction-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $OutputRoot) { throw 'Test output directory already exists.' }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$helperPath = Join-Path $OutputRoot 'obstruct.ps1'
$readyPath = Join-Path $OutputRoot 'ready'
$helper = @'
param([string]$ReadyPath)
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Threading;
public static class SceneObstruction {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left,Top,Right,Bottom; }
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern IntPtr FindWindow(string cls,string title);
    [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr window,out Rect rect);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] static extern IntPtr CreateWindowEx(uint ex,string cls,string title,uint style,int x,int y,int w,int h,IntPtr parent,IntPtr menu,IntPtr instance,IntPtr param);
    [DllImport("user32.dll")] static extern bool DestroyWindow(IntPtr window);
    [DllImport("user32.dll")] static extern bool SetProcessDpiAwarenessContext(IntPtr context);
    public static void Run(string ready) {
        SetProcessDpiAwarenessContext(new IntPtr(-4));
        System.IO.File.WriteAllText(ready,"ready");
        var deadline=DateTime.UtcNow.AddSeconds(40);
        IntPtr scene=IntPtr.Zero;
        while(scene==IntPtr.Zero && DateTime.UtcNow<deadline) {
            scene=FindWindow("DeltaNoDeviceSimulation",null);Thread.Sleep(10);
        }
        if(scene==IntPtr.Zero)throw new Exception("Scene never opened");
        Thread.Sleep(350);
        Rect rect;if(!GetWindowRect(scene,out rect))throw new Exception("Scene disappeared before obstruction");
        IntPtr overlay=CreateWindowEx(8,"STATIC","DELTA CAPTURE INVALIDITY TEST",0x90000000,
            (rect.Left+rect.Right)/2-60,(rect.Top+rect.Bottom)/2-60,120,120,
            IntPtr.Zero,IntPtr.Zero,IntPtr.Zero,IntPtr.Zero);
        if(overlay==IntPtr.Zero)throw new Exception("Cannot create test obstruction");
        try { Thread.Sleep(1500); } finally { DestroyWindow(overlay); }
    }
}
"@
[SceneObstruction]::Run($ReadyPath)
'@
[System.IO.File]::WriteAllText($helperPath,$helper)
$powershellExe = (Get-Process -Id $PID).Path
$helperArgs = @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+$helperPath+'"'),'-ReadyPath',('"'+$readyPath+'"'))
$worker = Start-Process -FilePath $powershellExe -ArgumentList $helperArgs -WindowStyle Hidden -PassThru
$previousPath = $env:PATH
try {
    $readyDeadline = (Get-Date).AddSeconds(15)
    while (!(Test-Path -LiteralPath $readyPath)) {
        if ($worker.HasExited -or (Get-Date) -gt $readyDeadline) { throw 'Obstruction helper did not initialize.' }
        Start-Sleep -Milliseconds 50
    }
    $cudaRoot = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4' }
    $env:PATH = (Join-Path $cudaRoot 'bin') + ';' + $previousPath
    $runPath = Join-Path $OutputRoot 'run'
    Push-Location $repoRoot
    try { & (Join-Path $portRoot 'build-cuda-benchmark/Release/delta_closed_loop_gpu.exe') --mode visible --frames 100 --speed 1500 --output $runPath }
    finally { Pop-Location }
    $runnerExit = $LASTEXITCODE
    $result = Get-Content -LiteralPath (Join-Path $runPath 'closed-loop.json') -Raw | ConvertFrom-Json
    if ($runnerExit -ne 2 -or $result.validity.valid -or !($result.validity.reasons -match 'obscured')) {
        throw 'Obstructed capture must return exit 2 and retain an invalid report naming the obstruction.'
    }
    Write-Host "Obstruction correctly invalidated capture; diagnostic report: $runPath/report.html"
} finally {
    $env:PATH = $previousPath
    if (!$worker.HasExited) { $worker.WaitForExit(5000) | Out-Null }
    if (!$worker.HasExited) { Stop-Process -Id $worker.Id }
}
