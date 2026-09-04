<#
.SYNOPSIS
    conquer3 docker-compose orchestration (Windows/PowerShell equivalent of Makefile).

.DESCRIPTION
    Three independent bring-up groups, matching docker-compose.yaml's profiles.
    None of the three depends on either other at the compose level (no
    cross-group `depends_on` edges) -- but within the first group, `ui`
    depends_on `scorer`, and Compose only resolves a profile-scoped depends_on
    when both profiles are active in the *same* command (see docker-compose.yaml's
    `ui` service comment). So `core` always brings up `serving`+`ui` together;
    splitting them would leave `ui` unable to resolve its dependency.

      core    -> profiles: serving, ui   (scorer = serving endpoint, ui)
      stream  -> profile:  stream        (pathway)
      airflow -> profile:  pipeline      (airflow-postgres, airflow-init,
                                          airflow-apiserver, airflow-scheduler,
                                          airflow-dag-processor, airflow-triggerer)

    `-wait` makes every `up` block until Compose's own `depends_on`/healthcheck
    chain actually resolves (or the timeout below trips), instead of returning
    as soon as containers are merely started -- the same guarantee
    scripts/startup.sh's hand-rolled polling gives, without duplicating it.

.EXAMPLE
    ./make.ps1 help
    ./make.ps1 core
    ./make.ps1 logs -Service scorer
#>

param(
    [Parameter(Position = 0)]
    [string]$Target = "help",

    [string]$Service = ""
)

$ErrorActionPreference = "Stop"

$CoreProfiles    = "serving,ui"
$StreamProfiles  = "stream"
$AirflowProfiles = "pipeline"
$AllProfiles     = "$CoreProfiles,$StreamProfiles,$AirflowProfiles"

# Generous margins over each group's slowest healthcheck (start_period + retries
# * interval) so a genuinely stuck container fails loud instead of hanging the
# terminal forever -- see each service's healthcheck in docker-compose.yaml.
$CoreWaitTimeout    = 300
$StreamWaitTimeout  = 120
$AirflowWaitTimeout = 300

function Test-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-Error "FAIL: .env not found. Run scripts/bootstrap.sh first."
        exit 1
    }
}

function Get-EnvValue([string]$Name) {
    if (-not (Test-Path ".env")) { return "" }
    $line = Select-String -Path ".env" -Pattern "^$Name=" -SimpleMatch:$false | Select-Object -First 1
    if (-not $line) { return "" }
    return ($line.Line -split "=", 2)[1].Trim()
}

function Invoke-Compose {
    param([string]$Profiles, [string[]]$ComposeArgs)
    $env:COMPOSE_PROFILES = $Profiles
    & docker compose @ComposeArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Show-Help {
    Write-Host @"
conquer3 docker-compose orchestration

  ./make.ps1 core                 up core services: scorer (serving endpoint) + ui
  ./make.ps1 stream                up streaming: pathway
  ./make.ps1 airflow               up airflow: postgres, init, apiserver,
                                   scheduler, dag-processor, triggerer
  ./make.ps1 up                    bring up all three groups
  ./make.ps1 down                  tear down every profile
  ./make.ps1 ps                    show status across every profile
  ./make.ps1 logs [-Service x]     follow logs (all profiles, optionally scoped)
  ./make.ps1 restart               restart every profile
  ./make.ps1 clean                 down -v -- deletes local volumes (airflow
                                   metadata, events, staging, models, pathway
                                   state, duckdb). Postgres/Redis are external/
                                   managed and untouched by this.

  ./make.ps1 build                 rebuild images for every service (all profiles)
  ./make.ps1 build -Service x      rebuild just one service's image, e.g.
                                   ./make.ps1 build -Service scorer
  ./make.ps1 core-build / stream-build / airflow-build   rebuild just one
                                   group's images (e.g. only airflow's, without
                                   touching scorer/ui)

  Per-group variants (swap the prefix): core-down, core-logs, core-ps,
  core-restart, core-build, stream-down, ..., airflow-down, ...
"@
}

function Up-Core {
    Test-EnvFile
    $mlflowUri = Get-EnvValue "MLFLOW_TRACKING_URI"
    if ([string]::IsNullOrWhiteSpace($mlflowUri)) {
        Write-Host "SKIP: MLFLOW_TRACKING_URI is empty in .env -- scorer resolves a"
        Write-Host "  champion at boot and refuses to start without one. Fill it in"
        Write-Host "  (and register+alias a champion), then re-run './make.ps1 core'."
        return
    }
    Invoke-Compose -Profiles $CoreProfiles -ComposeArgs @("up", "-d", "--build", "--wait", "--wait-timeout", "$CoreWaitTimeout")
}
function Down-Core    { Test-EnvFile; Invoke-Compose -Profiles $CoreProfiles -ComposeArgs @("down") }
function Ps-Core       { Test-EnvFile; Invoke-Compose -Profiles $CoreProfiles -ComposeArgs @("ps") }
function Restart-Core { Test-EnvFile; Invoke-Compose -Profiles $CoreProfiles -ComposeArgs @("restart") }
function Build-Core   { Test-EnvFile; Invoke-Compose -Profiles $CoreProfiles -ComposeArgs @("build") }
function Logs-Core {
    Test-EnvFile
    $composeArgs = @("logs", "-f")
    if ($Service) { $composeArgs += $Service }
    Invoke-Compose -Profiles $CoreProfiles -ComposeArgs $composeArgs
}

function Up-Stream {
    Test-EnvFile
    Invoke-Compose -Profiles $StreamProfiles -ComposeArgs @("up", "-d", "--build", "--wait", "--wait-timeout", "$StreamWaitTimeout")
}
function Down-Stream    { Test-EnvFile; Invoke-Compose -Profiles $StreamProfiles -ComposeArgs @("down") }
function Ps-Stream       { Test-EnvFile; Invoke-Compose -Profiles $StreamProfiles -ComposeArgs @("ps") }
function Restart-Stream { Test-EnvFile; Invoke-Compose -Profiles $StreamProfiles -ComposeArgs @("restart") }
function Build-Stream   { Test-EnvFile; Invoke-Compose -Profiles $StreamProfiles -ComposeArgs @("build") }
function Logs-Stream {
    Test-EnvFile
    $composeArgs = @("logs", "-f")
    if ($Service) { $composeArgs += $Service }
    Invoke-Compose -Profiles $StreamProfiles -ComposeArgs $composeArgs
}

function Up-Airflow {
    Test-EnvFile
    Invoke-Compose -Profiles $AirflowProfiles -ComposeArgs @("up", "-d", "--build", "--wait", "--wait-timeout", "$AirflowWaitTimeout")
}
function Down-Airflow    { Test-EnvFile; Invoke-Compose -Profiles $AirflowProfiles -ComposeArgs @("down") }
function Ps-Airflow       { Test-EnvFile; Invoke-Compose -Profiles $AirflowProfiles -ComposeArgs @("ps") }
function Restart-Airflow { Test-EnvFile; Invoke-Compose -Profiles $AirflowProfiles -ComposeArgs @("restart") }
function Build-Airflow   { Test-EnvFile; Invoke-Compose -Profiles $AirflowProfiles -ComposeArgs @("build") }
function Logs-Airflow {
    Test-EnvFile
    $composeArgs = @("logs", "-f")
    if ($Service) { $composeArgs += $Service }
    Invoke-Compose -Profiles $AirflowProfiles -ComposeArgs $composeArgs
}

# airflow -> stream -> core: no group depends on another (see header), this is
# just a stable order to bring everything up in one call.
function Up-All { Up-Airflow; Up-Stream; Up-Core }

# Naming a service explicitly (`docker compose build <service>`) bypasses profile
# scoping entirely (confirmed: it builds even with no COMPOSE_PROFILES set) --
# `-Service` is the same parameter `logs` already uses. Without it, all profiles
# must be active or `docker compose build` silently builds nothing.
function Build-All {
    Test-EnvFile
    if ($Service) {
        & docker compose build $Service
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Invoke-Compose -Profiles $AllProfiles -ComposeArgs @("build")
    }
}

function Down-All    { Test-EnvFile; Invoke-Compose -Profiles $AllProfiles -ComposeArgs @("down") }
function Ps-All       { Test-EnvFile; Invoke-Compose -Profiles $AllProfiles -ComposeArgs @("ps") }
function Restart-All { Test-EnvFile; Invoke-Compose -Profiles $AllProfiles -ComposeArgs @("restart") }
function Clean-All   { Test-EnvFile; Invoke-Compose -Profiles $AllProfiles -ComposeArgs @("down", "-v") }
function Logs-All {
    Test-EnvFile
    $composeArgs = @("logs", "-f")
    if ($Service) { $composeArgs += $Service }
    Invoke-Compose -Profiles $AllProfiles -ComposeArgs $composeArgs
}

switch ($Target) {
    "help"           { Show-Help }
    "core"           { Up-Core }
    "core-up"        { Up-Core }
    "core-down"      { Down-Core }
    "core-logs"      { Logs-Core }
    "core-ps"        { Ps-Core }
    "core-restart"   { Restart-Core }
    "core-build"     { Build-Core }
    "stream"         { Up-Stream }
    "stream-up"      { Up-Stream }
    "stream-down"    { Down-Stream }
    "stream-logs"    { Logs-Stream }
    "stream-ps"      { Ps-Stream }
    "stream-restart" { Restart-Stream }
    "stream-build"   { Build-Stream }
    "airflow"        { Up-Airflow }
    "airflow-up"     { Up-Airflow }
    "airflow-down"   { Down-Airflow }
    "airflow-logs"   { Logs-Airflow }
    "airflow-ps"     { Ps-Airflow }
    "airflow-restart" { Restart-Airflow }
    "airflow-build"  { Build-Airflow }
    "up"             { Up-All }
    "down"           { Down-All }
    "ps"             { Ps-All }
    "logs"           { Logs-All }
    "restart"        { Restart-All }
    "clean"          { Clean-All }
    "build"          { Build-All }
    default {
        Write-Error "Unknown target: $Target"
        Show-Help
        exit 1
    }
}
