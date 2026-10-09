# Launch outside the interactive terminal session; inspect pipeline.json for status.
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskPython = (Get-Command python).Source
$taskScript = Join-Path $PSScriptRoot 'pipeline.py'
$taskOut = Join-Path $taskRoot 'reports\angular_mbo_causal_20261008'
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u', $taskScript, '--from-stage', 'training') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOut 'background.stdout.log') -RedirectStandardError (Join-Path $taskOut 'background.stderr.log')
@{pid=$taskProcess.Id; launched_utc=[DateTime]::UtcNow.ToString('o'); command='python -u experiments/angular_mbo/pipeline.py --from-stage training'; note='Restarts training from prepared data; no epoch checkpoint was completed in the interrupted launch.'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOut 'background-launch.json') -Encoding utf8
Write-Output "Background pipeline PID: $($taskProcess.Id)"
