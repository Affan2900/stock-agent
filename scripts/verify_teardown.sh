#!/usr/bin/env bash
set -uo pipefail

echo "================================================================="
echo "        AWS SURVIVING BILLABLE RESOURCE TEARDOWN AUDIT          "
echo "================================================================="

REGION="${AWS_REGION:-us-east-1}"
SURVIVING_COUNT=0
PROBE_FAILURES=0

# Run one AWS query. A failed call is reported as a failure, never as zero — an
# expired credential must not read as "nothing is running".
probe() {
    local name="$1"; shift
    local count
    if ! count=$("$@" 2>/dev/null); then
        echo "🔶 ERROR: could not query $name (call failed — treat as UNKNOWN, not clean)."
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        return
    fi
    if [ -z "$count" ] || [ "$count" = "None" ]; then
        count=0
    fi
    if [ "$count" -gt 0 ]; then
        echo "❌ ALERT: $count surviving $name detected!"
        SURVIVING_COUNT=$((SURVIVING_COUNT + count))
    else
        echo "✅ PASS: 0 $name."
    fi
}

echo "Auditing AWS Region: $REGION..."

# 1. EKS Clusters
probe "EKS Clusters" \
    aws eks list-clusters --region "$REGION" \
    --query "length(clusters)" --output text

# 2. Running EC2 Instances
probe "EC2 Instances (running/pending)" \
    aws ec2 describe-instances --region "$REGION" \
    --filters "Name=instance-state-name,Values=running,pending" \
    --query "length(Reservations[*].Instances[*][])" --output text

# 3. NAT Gateways
probe "NAT Gateways" \
    aws ec2 describe-nat-gateways --region "$REGION" \
    --filter "Name=state,Values=available,pending" \
    --query "length(NatGateways)" --output text

# 4a. Classic Load Balancers (ELBv1).
#     Kubernetes `type: LoadBalancer` on EKS without the AWS Load Balancer
#     Controller provisions CLASSIC ELBs, which do not appear under the elbv2
#     API and are not in Terraform state. Auditing only elbv2 reported a clean
#     teardown while two Classic ELBs kept billing.
probe "Classic Load Balancers (ELB)" \
    aws elb describe-load-balancers --region "$REGION" \
    --query "length(LoadBalancerDescriptions)" --output text

# 4b. Application / Network Load Balancers (ELBv2)
probe "Application/Network Load Balancers (ELBv2)" \
    aws elbv2 describe-load-balancers --region "$REGION" \
    --query "length(LoadBalancers)" --output text

# 5a. Unattached EBS volumes — the actual leak case after an instance is gone.
probe "Unattached EBS Volumes" \
    aws ec2 describe-volumes --region "$REGION" \
    --filters "Name=status,Values=available" \
    --query "length(Volumes)" --output text

# 5b. Attached EBS volumes — expected while instances live, still billable.
probe "Attached EBS Volumes" \
    aws ec2 describe-volumes --region "$REGION" \
    --filters "Name=status,Values=in-use" \
    --query "length(Volumes)" --output text

# 6. Unassociated Elastic IPs — billed hourly precisely because they are idle.
probe "Unassociated Elastic IPs" \
    aws ec2 describe-addresses --region "$REGION" \
    --query "length(Addresses[?AssociationId==null])" --output text

echo "================================================================="
if [ "$PROBE_FAILURES" -gt 0 ]; then
    echo "🔶 VERIFICATION INCONCLUSIVE: $PROBE_FAILURES check(s) could not be run."
    echo "   Surviving resources found so far: $SURVIVING_COUNT"
    echo "   Fix credentials/permissions and re-run before trusting this result."
    exit 2
elif [ "$SURVIVING_COUNT" -eq 0 ]; then
    echo "🎉 VERIFICATION SUCCESS: ZERO SURVIVING BILLABLE AWS RESOURCES!"
    exit 0
else
    echo "⚠️ VERIFICATION FAILURE: $SURVIVING_COUNT SURVIVING BILLABLE AWS RESOURCES DETECTED!"
    exit 1
fi
