variable "github_repo" {
  type        = string
  default     = "Affan2900/stock-agent"
  description = "GitHub organization/username and repository name for OIDC federation"
}

variable "github_repo_immutable" {
  type        = string
  default     = "Affan2900@123811141/stock-agent@1312615276"
  description = <<-EOT
    Same repository expressed with GitHub's immutable numeric owner and repo IDs.
    GitHub now issues the OIDC `sub` claim using this form, so a trust policy that
    only matches the plain owner/repo name is rejected with "Not authorized to
    perform sts:AssumeRoleWithWebIdentity". Read the current value with:
      gh api /repos/<owner>/<repo>/actions/oidc/customization/sub
    Because the IDs survive a rename, this is the stricter of the two patterns:
    it cannot be claimed by someone who later registers a freed repository name.
  EOT
}

# GitHub Actions OIDC Identity Provider
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1", "1c58a21855e340bf1d0c4ec3be8353d265ab50d3"]
}

# IAM Role assumed by GitHub Actions workflows (No static secret keys!)
resource "aws_iam_role" "github_actions_deploy" {
  name = "github-actions-deploy-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        # Both accepted: GitHub is mid-migration to the immutable-ID subject, and a
        # list here is an OR. Dropping either one risks CD breaking on whichever
        # format the runner happens to emit.
        StringLike = {
          "token.actions.githubusercontent.com:sub" = [
            "repo:${var.github_repo}:*",
            "repo:${var.github_repo_immutable}:*",
          ]
        }
      }
    }]
  })
}

# Policy allowing ECR push/pull and EKS deployment
resource "aws_iam_policy" "github_actions_policy" {
  name        = "github-actions-deploy-policy"
  description = "Permissions for GitHub Actions to build ECR images and deploy to EKS"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:GetRepositoryPolicy",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "eks:DescribeCluster",
          "eks:ListClusters"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "github_actions_attach" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = aws_iam_policy.github_actions_policy.arn
}
