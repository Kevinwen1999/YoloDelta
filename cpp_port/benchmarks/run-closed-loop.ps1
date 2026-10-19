param(
    [ValidateRange(10, 10000)][int]$Frames = 200,
    [ValidateRange(0, 6000)][double]$Speed = 1500,
    [ValidateRange(0.25, 2)][double]$Scale = 1,
    [ValidateRange(2, 50)][double]$CadenceMs = 8,
    [ValidateRange(1, 50)][double]$InferenceDelayMs = 5,
    [ValidateSet('clean', 'occlusion', 'dropout', 'shake', 'crossing')][string]$Scenario = 'clean',
    [ValidateSet('cuda', 'tensorrt')][string]$Provider = 'tensorrt',
    [switch]$SkipBuildAndTests,
    [string]$Config,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$portRoot = Split-Path $PSScriptRoot -Parent
$buildRoot = Join-Path $portRoot 'build-cuda-benchmark'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vsRoot = & $vswhere -latest -property installationPath
$cmakeExe = Join-Path $vsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$cudaRoot = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4' }
$cudaRoot = $cudaRoot.Replace('\', '/').TrimEnd('/')
if (!$SkipBuildAndTests) {
    & $cmakeExe -S $portRoot -B $buildRoot -G 'Visual Studio 17 2022' -A x64 -T "cuda=$cudaRoot" -DDELTA_ENABLE_CUDA_PIPELINE=ON -DCMAKE_CUDA_ARCHITECTURES=86
    if ($LASTEXITCODE) { throw 'CUDA CMake configuration failed.' }
    & $cmakeExe --build $buildRoot --config Release --target delta_closed_loop_gpu delta_benchmark_tests delta_dispatch_tests -j 4
    if ($LASTEXITCODE) { throw 'Closed-loop build failed.' }
    $ctestExe = Join-Path (Split-Path $cmakeExe -Parent) 'ctest.exe'
    & $ctestExe --test-dir $buildRoot -C Release -R '^(delta_benchmark_tests|delta_dispatch_tests)$' --output-on-failure
    if ($LASTEXITCODE) { throw 'Benchmark integration checks failed.' }
}
if (!$Output) { $Output = Join-Path $portRoot ('benchmark-results/closed-loop-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$argsForRunner = @('--output', $Output, '--provider', $Provider, '--frames', $Frames,
    '--speed', $Speed, '--scale', $Scale, '--cadence-ms', $CadenceMs,
    '--inference-delay-ms', $InferenceDelayMs, '--scenario', $Scenario)
if ($Config) { $argsForRunner += @('--config', (Resolve-Path -LiteralPath $Config).Path) }
$previousPath = $env:PATH
try {
    $env:PATH = (Join-Path $cudaRoot 'bin') + ';' + $previousPath
    Push-Location $repoRoot
    try { & (Join-Path $buildRoot 'Release/delta_closed_loop_gpu.exe') @argsForRunner }
    finally { Pop-Location }
    if ($LASTEXITCODE) { throw 'Closed-loop run failed.' }
} finally { $env:PATH = $previousPath }
