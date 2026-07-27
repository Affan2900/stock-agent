<#
.SYNOPSIS
    Audit for surviving billable AWS resources after teardown.

.DESCRIPTION
    Native PowerShell port of verify_teardown.sh, for Windows hosts where `bash`
    resolves to the WSL launcher rather than Git Bash. Behaviour and exit codes
    are identical to the shell version.

.OUTPUTS
    Exit 0 — clean: zero surviving billable resources.
    Exit 1 — dirty: surviving resources detected.
    Exit 2 — inconclusive: one or more checks could not be run. Deliberately
             distinct from clean, so an expired credential never reads as "$0".
#>

$ErrorActionPreference = 'Continue'

Write-Output "================================================================="
Write-Output "        AWS SURVIVING BILLABLE RESOURCE TEARDOWN AUDIT          "
Write-Output "================================================================="

$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$script:SurvivingCount = 0
$script:ProbeFailures = 0

function Invoke-Probe {
    param(
        [Parameter(Mandatory = $true)][string]   $Name,
        [Parameter(Mandatory = $true)][string[]] $AwsArgs
    )

    $out = & aws @AwsArgs 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Output "[ERR ] could not query $Name (call failed - treat as UNKNOWN, not clean)."
        $script:ProbeFailures++
        return
    }

    $text = ($out | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($text) -or $text -eq "None") { $text = "0" }

    $count = 0
    if (-not [int]::TryParse($text, [ref]$count)) {
        Write-Output "[ERR ] unparseable response for $Name ('$text' - treat as UNKNOWN, not clean)."
        $script:ProbeFailures++
        return
    }

    if ($count -gt 0) {
        Write-Output "[FAIL] $count surviving $Name detected!"
        $script:SurvivingCount += $count
    }
    else {
        Write-Output "[ OK ] 0 $Name."
    }
}

Write-Output "Auditing AWS Region: $Region..."

# 1. EKS Clusters
Invoke-Probe "EKS Clusters" @(
    "eks", "list-clusters", "--region", $Region,
    "--query", "length(clusters)", "--output", "text")

# 2. Running EC2 Instances
Invoke-Probe "EC2 Instances (running/pending)" @(
    "ec2", "describe-instances", "--region", $Region,
    "--filters", "Name=instance-state-name,Values=running,pending",
    "--query", "length(Reservations[*].Instances[*][])", "--output", "text")

# 3. NAT Gateways
Invoke-Probe "NAT Gateways" @(
    "ec2", "describe-nat-gateways", "--region", $Region,
    "--filter", "Name=state,Values=available,pending",
    "--query", "length(NatGateways)", "--output", "text")

# 4a. Classic Load Balancers (ELBv1).
#     Kubernetes `type: LoadBalancer` on EKS without the AWS Load Balancer
#     Controller provisions CLASSIC ELBs, which do not appear under the elbv2
#     API and are not in Terraform state. Auditing only elbv2 reported a clean
#     teardown while two Classic ELBs kept billing.
Invoke-Probe "Classic Load Balancers (ELB)" @(
    "elb", "describe-load-balancers", "--region", $Region,
    "--query", "length(LoadBalancerDescriptions)", "--output", "text")

# 4b. Application / Network Load Balancers (ELBv2)
Invoke-Probe "Application/Network Load Balancers (ELBv2)" @(
    "elbv2", "describe-load-balancers", "--region", $Region,
    "--query", "length(LoadBalancers)", "--output", "text")

# 5a. Unattached EBS volumes - the actual leak case after an instance is gone.
Invoke-Probe "Unattached EBS Volumes" @(
    "ec2", "describe-volumes", "--region", $Region,
    "--filters", "Name=status,Values=available",
    "--query", "length(Volumes)", "--output", "text")

# 5b. Attached EBS volumes - expected while instances live, still billable.
Invoke-Probe "Attached EBS Volumes" @(
    "ec2", "describe-volumes", "--region", $Region,
    "--filters", "Name=status,Values=in-use",
    "--query", "length(Volumes)", "--output", "text")

# 6. Unassociated Elastic IPs - billed hourly precisely because they are idle.
Invoke-Probe "Unassociated Elastic IPs" @(
    "ec2", "describe-addresses", "--region", $Region,
    "--query", "length(Addresses[?AssociationId==null])", "--output", "text")

Write-Output "================================================================="
if ($script:ProbeFailures -gt 0) {
    Write-Output "VERIFICATION INCONCLUSIVE: $($script:ProbeFailures) check(s) could not be run."
    Write-Output "   Surviving resources found so far: $($script:SurvivingCount)"
    Write-Output "   Fix credentials/permissions and re-run before trusting this result."
    exit 2
}
elseif ($script:SurvivingCount -eq 0) {
    Write-Output "VERIFICATION SUCCESS: ZERO SURVIVING BILLABLE AWS RESOURCES!"
    exit 0
}
else {
    Write-Output "VERIFICATION FAILURE: $($script:SurvivingCount) SURVIVING BILLABLE AWS RESOURCES DETECTED!"
    exit 1
}
