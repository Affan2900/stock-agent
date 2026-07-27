data "tls_certificate" "eks" {
  url = aws_eks_cluster.eks.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks_oidc" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.eks.identity[0].oidc[0].issuer
}

# IAM Role for Service Accounts (IRSA) giving pod direct Bedrock access
resource "aws_iam_role" "bedrock_irsa" {
  name = "stock-agent-bedrock-irsa"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks_oidc.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${replace(aws_eks_cluster.eks.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:default:stock-agent-sa"
        }
      }
    }]
  })
}

# Bedrock InvokeModel Policy
resource "aws_iam_policy" "bedrock_policy" {
  name        = "stock-agent-bedrock-policy"
  description = "Allows EKS pods to invoke Amazon Bedrock LLM models"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ]
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "bedrock_irsa_attach" {
  role       = aws_iam_role.bedrock_irsa.name
  policy_arn = aws_iam_policy.bedrock_policy.arn
}
