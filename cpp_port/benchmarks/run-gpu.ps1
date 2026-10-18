param(
    [string]$Image,
    [switch]$Desktop,
    [ValidateRange(0, 15)][int]$GpuDevice,
    [ValidateSet('cuda', 'tensorrt')][string]$Provider = 'tensorrt',
    [ValidateRange(10, 10000)][int]$Frames = 200,
    [double]$Speed = 1500,
    [string]$Model,
    [string]$Config,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
if ($Desktop -eq [bool]$Image) { throw 'Choose -Image FILE or -Desktop.' }
$portRoot = Split-Path $PSScriptRoot -Parent
$buildRoot = Join-Path $portRoot 'build-cuda-benchmark'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vsRoot = & $vswhere -latest -property installationPath
$cmakeExe = Join-Path $vsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$cudaRoot = $env:CUDA_PATH
if (!$cudaRoot) { $cudaRoot = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4' }
$cudaRoot = $cudaRoot.Replace('\', '/').TrimEnd('/')
if (!(Test-Path (Join-Path $cudaRoot 'bin/nvcc.exe'))) { throw 'Set CUDA_PATH to the installed CUDA toolkit.' }
# Both GPUs on this workstation report compute capability 8.6. Compile native
# Ampere code instead of the legacy build's 5.2 target.
& $cmakeExe -S $portRoot -B $buildRoot -G 'Visual Studio 17 2022' -A x64 -T "cuda=$cudaRoot" -DDELTA_ENABLE_CUDA_PIPELINE=ON -DCMAKE_CUDA_ARCHITECTURES=86
if ($LASTEXITCODE) { throw 'CUDA CMake configuration failed.' }
& $cmakeExe --build $buildRoot --config Release --target delta_gpu_benchmark delta_benchmark_tests -j 4
if ($LASTEXITCODE) { throw 'GPU benchmark build failed.' }
$ctestExe = Join-Path (Split-Path $cmakeExe -Parent) 'ctest.exe'
if (!$Output) { $Output = Join-Path $portRoot ('benchmark-results/gpu-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$runnerArgs = @('--output', $Output, '--provider', $Provider, '--frames', $Frames, '--speed', $Speed)
# Without an explicit override, use StaticConfig's compiled device selection.
if ($PSBoundParameters.ContainsKey('GpuDevice')) { $runnerArgs += @('--gpu-device', $GpuDevice) }
if ($Desktop) { $runnerArgs += '--desktop' } else { $runnerArgs += @('--image', $Image) }
if ($Model) { $runnerArgs += @('--model', $Model) }
if ($Config) { $runnerArgs += @('--config', $Config) }
# The executable imports cudart before its own DLL search setup can run.
$previousPath = $env:PATH
try {
    $env:PATH = (Join-Path $cudaRoot 'bin') + ';' + $previousPath
    & $ctestExe --test-dir $buildRoot -C Release -R '^delta_benchmark_tests$' --output-on-failure
    if ($LASTEXITCODE) { throw 'Replay integration checks failed in the CUDA build.' }
    & (Join-Path $buildRoot 'Release/delta_gpu_benchmark.exe') @runnerArgs
    if ($LASTEXITCODE) { throw "GPU benchmark failed with exit code $LASTEXITCODE." }
} finally { $env:PATH = $previousPath }
