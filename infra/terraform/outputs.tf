output "aws_region" {
  value = var.aws_region
}

output "eks_cluster_name" {
  value = aws_eks_cluster.eks.name
}

output "eks_cluster_endpoint" {
  value = aws_eks_cluster.eks.endpoint
}

output "ecr_repository_api_url" {
  value = aws_ecr_repository.api.repository_url
}

output "ecr_repository_ui_user_url" {
  value = aws_ecr_repository.ui_user.repository_url
}

output "ecr_repository_ui_ops_url" {
  value = aws_ecr_repository.ui_ops.repository_url
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions_deploy.arn
}

output "bedrock_irsa_role_arn" {
  value = aws_iam_role.bedrock_irsa.arn
}

output "kubeconfig_command" {
  value = "aws eks update-kubeconfig --region ${var.aws_region} --name ${aws_eks_cluster.eks.name}"
}
