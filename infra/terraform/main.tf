terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "stock-agent-ops"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "production"
}

# $10 CloudWatch Billing Alarm (Safety Non-Negotiable)
resource "aws_cloudwatch_metric_alarm" "billing_alarm" {
  alarm_name          = "stock-agent-billing-alarm-10usd"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600
  statistic           = "Maximum"
  threshold           = 10.0
  alarm_description   = "Alarm triggered if total estimated monthly charges exceed $10 USD"
  alarm_actions       = []

  treat_missing_data  = "ignore"
  dimensions = {
    Currency = "USD"
  }
}

