param(
    [ValidateSet('quick', 'sweep', 'speed')][string]$Suite = 'quick',
    [string]$Config,
    [string]$Baseline,
    [string]$Scenario,
    [switch]$Timing,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$portRoot = Split-Path $PSScriptRoot -Parent
$buildRoot = Join-Path $portRoot 'build-benchmark'
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCommand) {
    $cmakeExe = $cmakeCommand.Source
} else {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (!(Test-Path -LiteralPath $vswhere)) { throw 'Install CMake or Visual Studio with the C++ CMake tools.' }
    $vsRoot = & $vswhere -latest -property installationPath
    $cmakeExe = Join-Path $vsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
}
& $cmakeExe -S $portRoot -B $buildRoot -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE) { throw 'CMake configuration failed.' }
& $cmakeExe --build $buildRoot --config Release --target delta_benchmark delta_benchmark_tests -j 4
if ($LASTEXITCODE) { throw 'Benchmark build failed.' }
$ctestExe = Join-Path (Split-Path $cmakeExe -Parent) 'ctest.exe'
& $ctestExe --test-dir $buildRoot -C Release -R '^delta_benchmark' --output-on-failure
if ($LASTEXITCODE) { throw 'Benchmark integration checks failed.' }
if (!$Output) { $Output = Join-Path $portRoot ('benchmark-results/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$runnerArgs = @('--suite', $Suite, '--output', $Output)
if ($Config) { $runnerArgs += @('--config', $Config) }
if ($Baseline) { $runnerArgs += @('--baseline', $Baseline) }
if ($Scenario) { $runnerArgs += @('--scenario', $Scenario) }
if ($Timing) { $runnerArgs += '--timing' }
& (Join-Path $buildRoot 'Release/delta_benchmark.exe') @runnerArgs
if ($LASTEXITCODE) { throw "Benchmark failed with exit code $LASTEXITCODE. Inspect the reported error or correctness violations." }
