#!/usr/bin/env bash
set -eo pipefail

echo "================================================================="
echo "   AWS TERRAFORM DESTROY & EMERGENCY RESOURCE NUKE SCRIPT       "
echo "================================================================="

REGION="us-east-1"
CLUSTER_NAME="stock-agent-eks"

echo "[1/3] Executing terraform destroy..."
if [ -d "infra/terraform" ]; then
    cd infra/terraform
    terraform destroy -auto-approve || true
    cd ../..
fi

echo "[2/3] Checking for surviving EKS clusters or node groups..."
CLUSTERS=$(aws eks list-clusters --region "$REGION" --query "clusters" --output text 2>/dev/null || true)

if [[ "$CLUSTERS" == *"$CLUSTER_NAME"* ]]; then
    echo "WARNING: EKS cluster $CLUSTER_NAME still exists. Deleting..."
    aws eks delete-cluster --region "$REGION" --name "$CLUSTER_NAME" || true
fi

echo "[3/3] Running teardown verification..."
bash scripts/verify_teardown.sh
