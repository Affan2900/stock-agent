#!/usr/bin/env bash
set -eo pipefail

echo "================================================================="
echo "       AWS REAL-TIME RUNNING RESOURCE HOURLY COST ESTIMATOR      "
echo "================================================================="

REGION="us-east-1"

# Costs (approximate US-East-1)
EKS_HOURLY=0.10
T3_MEDIUM_HOURLY=0.0416
NAT_HOURLY=0.045
ALB_HOURLY=0.0225

EKS_COUNT=$(aws eks list-clusters --region "$REGION" --query "length(clusters)" --output text 2>/dev/null || echo 0)
EC2_COUNT=$(aws ec2 describe-instances --region "$REGION" --filters "Name=instance-state-name,Values=running" --query "length(Reservations[*].Instances[*])" --output text 2>/dev/null || echo 0)
NAT_COUNT=$(aws ec2 describe-nat-gateways --region "$REGION" --filter "Name=state,Values=available" --query "length(NatGateways)" --output text 2>/dev/null || echo 0)
ALB_COUNT=$(aws elbv2 describe-load-balancers --region "$REGION" --query "length(LoadBalancers)" --output text 2>/dev/null || echo 0)

HOURLY_COST=$(python -c "print(round($EKS_COUNT * $EKS_HOURLY + $EC2_COUNT * $T3_MEDIUM_HOURLY + $NAT_COUNT * $NAT_HOURLY + $ALB_COUNT * $ALB_HOURLY, 4))" 2>/dev/null || echo "0.0")

echo "Current Running Active Resources:"
echo " - EKS Clusters ($0.10/hr): $EKS_COUNT"
echo " - t3.medium EC2 Nodes ($0.0416/hr): $EC2_COUNT"
echo " - NAT Gateways ($0.045/hr): $NAT_COUNT"
echo " - Load Balancers ($0.0225/hr): $ALB_COUNT"
echo "-----------------------------------------------------------------"
echo "Estimated Hourly Spend: ~$${HOURLY_COST}/hr (~$$(python -c "print(round($HOURLY_COST * 24, 2))" 2>/dev/null || echo "0.0")/day)"
echo "================================================================="
