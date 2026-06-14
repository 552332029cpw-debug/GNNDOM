param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$Repo = "/home/pivot/work/GNNDOM",
    [string]$Dataf = "data/full_train_v1",
    [string]$Log = "logs/full_camera_sampling_fixed_oblique_wide.log",
    [string]$PidFile = "logs/full_camera_sampling_fixed_oblique_wide.windows_pid"
)

$repoUnc = "\\wsl.localhost\$Distro" + ($Repo -replace "/", "\")
$pidUnc = Join-Path $repoUnc ($PidFile -replace "/", "\")

$bashCommand = "echo STARTED `$(date -Is) > $Log; exec bash scripts/render_isaac_camera_windows_from_wsl.sh --dataf $Dataf --overwrite >> $Log 2>&1"

$process = Start-Process `
    -FilePath "C:\Windows\System32\wsl.exe" `
    -ArgumentList @("-d", $Distro, "--cd", $Repo, "--", "bash", "-lc", $bashCommand) `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $pidUnc -Value $process.Id -NoNewline
Write-Output "WINPID=$($process.Id)"
