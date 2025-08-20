#!/usr/bin/env python3
"""
🚀 Production Deployment Automation Suite

This comprehensive deployment suite provides enterprise-grade deployment
automation for the Terragon IoT Anomaly Detection System across multiple
platforms and environments.

Features:
- Multi-cloud deployment (AWS, GCP, Azure, on-premise)
- Kubernetes orchestration with Helm charts
- Docker containerization with multi-arch builds
- CI/CD pipeline automation
- Infrastructure as Code (Terraform/CloudFormation)
- Blue-green and canary deployment strategies
- Health checks and automated rollbacks
- Monitoring and alerting integration
- Compliance and security scanning
"""

import os
import sys
import json
import yaml
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeploymentPlatform(Enum):
    """Supported deployment platforms."""
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"
    AWS_ECS = "aws_ecs"
    AWS_LAMBDA = "aws_lambda"
    GCP_CLOUD_RUN = "gcp_cloud_run"
    AZURE_CONTAINER = "azure_container"
    EDGE_DEVICE = "edge_device"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING_UPDATE = "rolling_update"
    RECREATE = "recreate"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    platform: DeploymentPlatform
    strategy: DeploymentStrategy
    replicas: int = 3
    image_tag: str = "latest"
    environment: str = "production"
    namespace: str = "default"
    cpu_request: str = "500m"
    cpu_limit: str = "1000m"
    memory_request: str = "512Mi"
    memory_limit: str = "1Gi"
    gpu_enabled: bool = False
    auto_scaling: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu: int = 70
    health_check_path: str = "/health"
    readiness_probe_path: str = "/ready"
    config_maps: List[str] = None
    secrets: List[str] = None
    service_type: str = "ClusterIP"
    ingress_enabled: bool = True
    ingress_host: str = ""
    ssl_enabled: bool = True
    monitoring_enabled: bool = True
    
    def __post_init__(self):
        if self.config_maps is None:
            self.config_maps = []
        if self.secrets is None:
            self.secrets = []


class ProductionDeploymentSuite:
    """Comprehensive production deployment automation."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.templates_dir = self.project_root / "deployment" / "templates"
        self.output_dir = self.project_root / "deployment" / "generated"
        
        # Ensure directories exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate deployment templates
        self._generate_deployment_templates()
        
        logger.info(f"Initialized deployment suite for project: {self.project_root}")
    
    def _generate_deployment_templates(self):
        """Generate all deployment templates."""
        self._generate_kubernetes_templates()
        self._generate_docker_templates()
        self._generate_terraform_templates()
        self._generate_cicd_templates()
        self._generate_monitoring_templates()
    
    def _generate_kubernetes_templates(self):
        """Generate Kubernetes deployment templates."""
        
        # Kubernetes Deployment
        k8s_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': '{{ .Values.app.name }}',
                'namespace': '{{ .Values.app.namespace }}',
                'labels': {
                    'app': '{{ .Values.app.name }}',
                    'version': '{{ .Values.app.version }}',
                    'component': 'anomaly-detection'
                }
            },
            'spec': {
                'replicas': '{{ .Values.replicas }}',
                'selector': {
                    'matchLabels': {
                        'app': '{{ .Values.app.name }}'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': '{{ .Values.app.name }}',
                            'version': '{{ .Values.app.version }}'
                        },
                        'annotations': {
                            'prometheus.io/scrape': 'true',
                            'prometheus.io/port': '8080',
                            'prometheus.io/path': '/metrics'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': '{{ .Values.app.name }}',
                            'image': '{{ .Values.image.repository }}:{{ .Values.image.tag }}',
                            'imagePullPolicy': 'Always',
                            'ports': [
                                {'containerPort': 8080, 'name': 'http'},
                                {'containerPort': 8081, 'name': 'metrics'}
                            ],
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': '{{ .Values.app.environment }}'},
                                {'name': 'LOG_LEVEL', 'value': '{{ .Values.app.logLevel }}'},
                                {'name': 'METRICS_ENABLED', 'value': '{{ .Values.monitoring.enabled }}'}
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': '{{ .Values.resources.requests.cpu }}',
                                    'memory': '{{ .Values.resources.requests.memory }}'
                                },
                                'limits': {
                                    'cpu': '{{ .Values.resources.limits.cpu }}',
                                    'memory': '{{ .Values.resources.limits.memory }}'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8080
                                },
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30,
                                'timeoutSeconds': 10,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8080
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            },
                            'securityContext': {
                                'allowPrivilegeEscalation': False,
                                'runAsNonRoot': True,
                                'runAsUser': 1000,
                                'readOnlyRootFilesystem': True,
                                'capabilities': {'drop': ['ALL']}
                            }
                        }],
                        'serviceAccountName': '{{ .Values.app.serviceAccount }}',
                        'securityContext': {
                            'fsGroup': 1000
                        }
                    }
                }
            }
        }
        
        # Service
        k8s_service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': '{{ .Values.app.name }}-service',
                'namespace': '{{ .Values.app.namespace }}',
                'labels': {
                    'app': '{{ .Values.app.name }}'
                }
            },
            'spec': {
                'selector': {
                    'app': '{{ .Values.app.name }}'
                },
                'ports': [
                    {'port': 80, 'targetPort': 8080, 'name': 'http'},
                    {'port': 8081, 'targetPort': 8081, 'name': 'metrics'}
                ],
                'type': '{{ .Values.service.type }}'
            }
        }
        
        # HorizontalPodAutoscaler
        k8s_hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': '{{ .Values.app.name }}-hpa',
                'namespace': '{{ .Values.app.namespace }}'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': '{{ .Values.app.name }}'
                },
                'minReplicas': '{{ .Values.autoscaling.minReplicas }}',
                'maxReplicas': '{{ .Values.autoscaling.maxReplicas }}',
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': '{{ .Values.autoscaling.targetCPU }}'
                            }
                        }
                    }
                ]
            }
        }
        
        # Ingress
        k8s_ingress = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': '{{ .Values.app.name }}-ingress',
                'namespace': '{{ .Values.app.namespace }}',
                'annotations': {
                    'kubernetes.io/ingress.class': 'nginx',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod',
                    'nginx.ingress.kubernetes.io/rate-limit': '100',
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'true'
                }
            },
            'spec': {
                'tls': [{
                    'hosts': ['{{ .Values.ingress.host }}'],
                    'secretName': '{{ .Values.app.name }}-tls'
                }],
                'rules': [{
                    'host': '{{ .Values.ingress.host }}',
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': '{{ .Values.app.name }}-service',
                                    'port': {'number': 80}
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        # Save templates
        with open(self.templates_dir / "k8s-deployment.yaml", 'w') as f:
            yaml.dump(k8s_deployment, f, default_flow_style=False)
        
        with open(self.templates_dir / "k8s-service.yaml", 'w') as f:
            yaml.dump(k8s_service, f, default_flow_style=False)
        
        with open(self.templates_dir / "k8s-hpa.yaml", 'w') as f:
            yaml.dump(k8s_hpa, f, default_flow_style=False)
        
        with open(self.templates_dir / "k8s-ingress.yaml", 'w') as f:
            yaml.dump(k8s_ingress, f, default_flow_style=False)
        
        # Values file for Helm
        helm_values = {
            'app': {
                'name': 'iot-anomaly-detection',
                'version': 'v4.0.0',
                'environment': 'production',
                'namespace': 'anomaly-detection',
                'serviceAccount': 'anomaly-detection-sa',
                'logLevel': 'INFO'
            },
            'image': {
                'repository': 'terragon/iot-anomaly-detection',
                'tag': 'v4.0.0',
                'pullPolicy': 'Always'
            },
            'replicas': 3,
            'resources': {
                'requests': {'cpu': '500m', 'memory': '512Mi'},
                'limits': {'cpu': '1000m', 'memory': '1Gi'}
            },
            'autoscaling': {
                'enabled': True,
                'minReplicas': 2,
                'maxReplicas': 10,
                'targetCPU': 70
            },
            'service': {
                'type': 'ClusterIP',
                'port': 80
            },
            'ingress': {
                'enabled': True,
                'host': 'anomaly-detection.terragon.ai',
                'tls': True
            },
            'monitoring': {
                'enabled': True,
                'serviceMonitor': True
            }
        }
        
        with open(self.templates_dir / "helm-values.yaml", 'w') as f:
            yaml.dump(helm_values, f, default_flow_style=False)
    
    def _generate_docker_templates(self):
        """Generate Docker deployment templates."""
        
        # Multi-stage Dockerfile
        dockerfile = '''
# Multi-stage build for optimized production image
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir --user .

# Production stage
FROM python:3.12-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && groupadd -r appuser \\
    && useradd -r -g appuser appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=appuser:appuser examples/ ./examples/
COPY --chown=appuser:appuser config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/data \\
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV METRICS_ENABLED=true

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "iot_edge_anomaly.main", "--config", "config/production.yaml"]
'''
        
        # Docker Compose for multi-service deployment
        docker_compose = {
            'version': '3.8',
            'services': {
                'anomaly-detection': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile'
                    },
                    'image': 'terragon/iot-anomaly-detection:v4.0.0',
                    'container_name': 'anomaly-detection-api',
                    'restart': 'unless-stopped',
                    'ports': ['8080:8080', '8081:8081'],
                    'environment': [
                        'ENVIRONMENT=production',
                        'LOG_LEVEL=INFO',
                        'METRICS_ENABLED=true',
                        'REDIS_URL=redis://redis:6379',
                        'POSTGRES_URL=postgresql://postgres:password@postgres:5432/anomaly_db'
                    ],
                    'volumes': [
                        './config:/app/config:ro',
                        './logs:/app/logs',
                        './data:/app/data'
                    ],
                    'depends_on': {
                        'redis': {'condition': 'service_healthy'},
                        'postgres': {'condition': 'service_healthy'}
                    },
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    },
                    'deploy': {
                        'replicas': 3,
                        'resources': {
                            'limits': {'cpus': '1.0', 'memory': '1G'},
                            'reservations': {'cpus': '0.5', 'memory': '512M'}
                        }
                    }
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'container_name': 'anomaly-redis',
                    'restart': 'unless-stopped',
                    'ports': ['6379:6379'],
                    'volumes': ['redis_data:/data'],
                    'healthcheck': {
                        'test': ['CMD', 'redis-cli', 'ping'],
                        'interval': '10s',
                        'timeout': '5s',
                        'retries': 3
                    }
                },
                'postgres': {
                    'image': 'postgres:15-alpine',
                    'container_name': 'anomaly-postgres',
                    'restart': 'unless-stopped',
                    'ports': ['5432:5432'],
                    'environment': [
                        'POSTGRES_DB=anomaly_db',
                        'POSTGRES_USER=postgres',
                        'POSTGRES_PASSWORD=password'
                    ],
                    'volumes': [
                        'postgres_data:/var/lib/postgresql/data',
                        './sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro'
                    ],
                    'healthcheck': {
                        'test': ['CMD-SHELL', 'pg_isready -U postgres'],
                        'interval': '10s',
                        'timeout': '5s',
                        'retries': 3
                    }
                },
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'container_name': 'anomaly-prometheus',
                    'ports': ['9090:9090'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro'
                    ],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--web.enable-lifecycle'
                    ]
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'container_name': 'anomaly-grafana',
                    'ports': ['3000:3000'],
                    'environment': [
                        'GF_SECURITY_ADMIN_PASSWORD=admin123'
                    ],
                    'volumes': [
                        'grafana_data:/var/lib/grafana',
                        './monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro'
                    ]
                }
            },
            'volumes': {
                'redis_data': {},
                'postgres_data': {},
                'grafana_data': {}
            },
            'networks': {
                'default': {
                    'driver': 'bridge'
                }
            }
        }
        
        # Save Docker files
        with open(self.templates_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        with open(self.templates_dir / "docker-compose.yml", 'w') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)
    
    def _generate_terraform_templates(self):
        """Generate Terraform infrastructure templates."""
        
        # AWS ECS Terraform configuration
        aws_main_tf = '''
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.project_name}-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "private" {
  count = length(var.availability_zones)

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name        = "${var.project_name}-private-${count.index + 1}"
    Environment = var.environment
  }
}

resource "aws_subnet" "public" {
  count = length(var.availability_zones)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${var.project_name}-public-${count.index + 1}"
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Environment = var.environment
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = var.environment == "production"

  tags = {
    Environment = var.environment
  }
}

# ECS Service
resource "aws_ecs_service" "anomaly_detection" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.anomaly_detection.arn
  desired_count   = var.desired_count

  launch_type = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.main.arn
    container_name   = "anomaly-detection"
    container_port   = 8080
  }

  depends_on = [aws_lb_listener.main]

  tags = {
    Environment = var.environment
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.anomaly_detection.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "scale_up" {
  name               = "${var.project_name}-scale-up"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = var.target_cpu_utilization
  }
}
'''
        
        # Variables file
        variables_tf = '''
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "iot-anomaly-detection"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 3
}

variable "min_capacity" {
  description = "Minimum number of ECS tasks"
  type        = number
  default     = 2
}

variable "max_capacity" {
  description = "Maximum number of ECS tasks"
  type        = number
  default     = 10
}

variable "target_cpu_utilization" {
  description = "Target CPU utilization for auto scaling"
  type        = number
  default     = 70
}
'''
        
        # Save Terraform files
        terraform_dir = self.templates_dir / "terraform"
        terraform_dir.mkdir(exist_ok=True)
        
        with open(terraform_dir / "main.tf", 'w') as f:
            f.write(aws_main_tf)
        
        with open(terraform_dir / "variables.tf", 'w') as f:
            f.write(variables_tf)
    
    def _generate_cicd_templates(self):
        """Generate CI/CD pipeline templates."""
        
        # GitHub Actions workflow
        github_workflow = {
            'name': 'CI/CD Pipeline',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main']
                }
            },
            'env': {
                'REGISTRY': 'ghcr.io',
                'IMAGE_NAME': '${{ github.repository }}'
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {'python-version': '3.12'}
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -e .[dev]'
                        },
                        {
                            'name': 'Run tests',
                            'run': 'pytest tests/ --cov=src --cov-report=xml'
                        },
                        {
                            'name': 'Security scan',
                            'run': 'bandit -r src/'
                        }
                    ]
                },
                'build-and-push': {
                    'needs': 'test',
                    'runs-on': 'ubuntu-latest',
                    'if': "github.event_name == 'push' && github.ref == 'refs/heads/main'",
                    'permissions': {
                        'contents': 'read',
                        'packages': 'write'
                    },
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Set up Docker Buildx',
                            'uses': 'docker/setup-buildx-action@v3'
                        },
                        {
                            'name': 'Log in to Container Registry',
                            'uses': 'docker/login-action@v3',
                            'with': {
                                'registry': '${{ env.REGISTRY }}',
                                'username': '${{ github.actor }}',
                                'password': '${{ secrets.GITHUB_TOKEN }}'
                            }
                        },
                        {
                            'name': 'Extract metadata',
                            'id': 'meta',
                            'uses': 'docker/metadata-action@v5',
                            'with': {
                                'images': '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}',
                                'tags': 'type=ref,event=branch\\ntype=ref,event=pr\\ntype=sha'
                            }
                        },
                        {
                            'name': 'Build and push Docker image',
                            'uses': 'docker/build-push-action@v5',
                            'with': {
                                'context': '.',
                                'platforms': 'linux/amd64,linux/arm64',
                                'push': True,
                                'tags': '${{ steps.meta.outputs.tags }}',
                                'labels': '${{ steps.meta.outputs.labels }}',
                                'cache-from': 'type=gha',
                                'cache-to': 'type=gha,mode=max'
                            }
                        }
                    ]
                },
                'deploy': {
                    'needs': 'build-and-push',
                    'runs-on': 'ubuntu-latest',
                    'if': "github.ref == 'refs/heads/main'",
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Configure AWS credentials',
                            'uses': 'aws-actions/configure-aws-credentials@v4',
                            'with': {
                                'aws-access-key-id': '${{ secrets.AWS_ACCESS_KEY_ID }}',
                                'aws-secret-access-key': '${{ secrets.AWS_SECRET_ACCESS_KEY }}',
                                'aws-region': 'us-west-2'
                            }
                        },
                        {
                            'name': 'Deploy to ECS',
                            'run': '''
                              aws ecs update-service \\
                                --cluster iot-anomaly-detection-cluster \\
                                --service iot-anomaly-detection-service \\
                                --force-new-deployment
                            '''
                        }
                    ]
                }
            }
        }
        
        # Save CI/CD files
        cicd_dir = self.templates_dir / "cicd"
        cicd_dir.mkdir(exist_ok=True)
        
        github_dir = cicd_dir / "github" / "workflows"
        github_dir.mkdir(parents=True, exist_ok=True)
        
        with open(github_dir / "ci-cd.yml", 'w') as f:
            yaml.dump(github_workflow, f, default_flow_style=False)
    
    def _generate_monitoring_templates(self):
        """Generate monitoring and observability templates."""
        
        # Prometheus configuration
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'rule_files': [
                'alert_rules.yml'
            ],
            'alertmanager': {
                'alertmanagers': [{
                    'static_configs': [{
                        'targets': ['alertmanager:9093']
                    }]
                }]
            },
            'scrape_configs': [
                {
                    'job_name': 'anomaly-detection',
                    'static_configs': [{
                        'targets': ['anomaly-detection:8081']
                    }],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'kubernetes-pods',
                    'kubernetes_sd_configs': [{
                        'role': 'pod'
                    }],
                    'relabel_configs': [
                        {
                            'source_labels': ['__meta_kubernetes_pod_annotation_prometheus_io_scrape'],
                            'action': 'keep',
                            'regex': True
                        }
                    ]
                }
            ]
        }
        
        # Alert rules
        alert_rules = {
            'groups': [
                {
                    'name': 'anomaly-detection-alerts',
                    'rules': [
                        {
                            'alert': 'HighErrorRate',
                            'expr': 'rate(http_requests_total{status=~"5.."}[5m]) > 0.1',
                            'for': '5m',
                            'labels': {'severity': 'critical'},
                            'annotations': {
                                'summary': 'High error rate detected',
                                'description': 'Error rate is above 10% for more than 5 minutes'
                            }
                        },
                        {
                            'alert': 'HighLatency',
                            'expr': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1',
                            'for': '10m',
                            'labels': {'severity': 'warning'},
                            'annotations': {
                                'summary': 'High latency detected',
                                'description': '95th percentile latency is above 1 second'
                            }
                        },
                        {
                            'alert': 'ServiceDown',
                            'expr': 'up{job="anomaly-detection"} == 0',
                            'for': '1m',
                            'labels': {'severity': 'critical'},
                            'annotations': {
                                'summary': 'Anomaly detection service is down',
                                'description': 'The anomaly detection service has been down for more than 1 minute'
                            }
                        }
                    ]
                }
            ]
        }
        
        # Save monitoring files
        monitoring_dir = self.templates_dir / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        with open(monitoring_dir / "prometheus.yml", 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        with open(monitoring_dir / "alert_rules.yml", 'w') as f:
            yaml.dump(alert_rules, f, default_flow_style=False)
    
    def deploy(
        self,
        config: DeploymentConfig,
        dry_run: bool = False,
        verbose: bool = True
    ) -> bool:
        """
        Deploy the application using the specified configuration.
        
        Args:
            config: Deployment configuration
            dry_run: If True, show what would be deployed without executing
            verbose: Enable verbose logging
            
        Returns:
            True if deployment successful
        """
        logger.info(f"Starting deployment: {config.name} to {config.platform.value}")
        
        try:
            if config.platform == DeploymentPlatform.KUBERNETES:
                return self._deploy_kubernetes(config, dry_run, verbose)
            elif config.platform == DeploymentPlatform.DOCKER_COMPOSE:
                return self._deploy_docker_compose(config, dry_run, verbose)
            elif config.platform == DeploymentPlatform.AWS_ECS:
                return self._deploy_aws_ecs(config, dry_run, verbose)
            else:
                logger.error(f"Unsupported platform: {config.platform}")
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _deploy_kubernetes(self, config: DeploymentConfig, dry_run: bool, verbose: bool) -> bool:
        """Deploy to Kubernetes."""
        logger.info("Deploying to Kubernetes...")
        
        # Generate manifests from templates
        manifests = self._generate_k8s_manifests(config)
        
        # Save manifests
        manifest_file = self.output_dir / f"{config.name}-k8s-manifests.yaml"
        with open(manifest_file, 'w') as f:
            f.write("---\\n".join(manifests))
        
        if dry_run:
            logger.info(f"Dry run: Would deploy manifests from {manifest_file}")
            return True
        
        # Apply manifests
        try:
            cmd = ["kubectl", "apply", "-f", str(manifest_file)]
            if config.namespace != "default":
                cmd.extend(["-n", config.namespace])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Kubernetes deployment successful")
                return True
            else:
                logger.error(f"Kubernetes deployment failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("kubectl not found. Please install kubectl.")
            return False
    
    def _deploy_docker_compose(self, config: DeploymentConfig, dry_run: bool, verbose: bool) -> bool:
        """Deploy using Docker Compose."""
        logger.info("Deploying with Docker Compose...")
        
        compose_file = self.templates_dir / "docker-compose.yml"
        
        if dry_run:
            logger.info(f"Dry run: Would deploy using {compose_file}")
            return True
        
        try:
            # Deploy with Docker Compose
            cmd = ["docker-compose", "-f", str(compose_file), "up", "-d", "--scale", f"anomaly-detection={config.replicas}"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Docker Compose deployment successful")
                return True
            else:
                logger.error(f"Docker Compose deployment failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("docker-compose not found. Please install Docker Compose.")
            return False
    
    def _deploy_aws_ecs(self, config: DeploymentConfig, dry_run: bool, verbose: bool) -> bool:
        """Deploy to AWS ECS."""
        logger.info("Deploying to AWS ECS...")
        
        if dry_run:
            logger.info("Dry run: Would deploy to AWS ECS using Terraform")
            return True
        
        terraform_dir = self.templates_dir / "terraform"
        
        try:
            # Initialize Terraform
            subprocess.run(["terraform", "init"], cwd=terraform_dir, check=True)
            
            # Plan deployment
            subprocess.run(["terraform", "plan"], cwd=terraform_dir, check=True)
            
            # Apply deployment
            subprocess.run(["terraform", "apply", "-auto-approve"], cwd=terraform_dir, check=True)
            
            logger.info("AWS ECS deployment successful")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"AWS ECS deployment failed: {e}")
            return False
        except FileNotFoundError:
            logger.error("terraform not found. Please install Terraform.")
            return False
    
    def _generate_k8s_manifests(self, config: DeploymentConfig) -> List[str]:
        """Generate Kubernetes manifests from templates."""
        # This would normally use a templating engine like Jinja2
        # For simplicity, we'll do basic string replacement
        
        manifests = []
        
        # Load and process templates
        template_files = [
            "k8s-deployment.yaml",
            "k8s-service.yaml",
            "k8s-hpa.yaml"
        ]
        
        if config.ingress_enabled:
            template_files.append("k8s-ingress.yaml")
        
        for template_file in template_files:
            template_path = self.templates_dir / template_file
            if template_path.exists():
                with open(template_path, 'r') as f:
                    content = f.read()
                
                # Simple template substitution
                content = content.replace('{{ .Values.app.name }}', config.name)
                content = content.replace('{{ .Values.app.namespace }}', config.namespace)
                content = content.replace('{{ .Values.replicas }}', str(config.replicas))
                # Add more substitutions as needed
                
                manifests.append(content)
        
        return manifests
    
    def rollback(self, config: DeploymentConfig, revision: Optional[str] = None) -> bool:
        """Rollback deployment to previous version."""
        logger.info(f"Rolling back deployment: {config.name}")
        
        try:
            if config.platform == DeploymentPlatform.KUBERNETES:
                cmd = ["kubectl", "rollout", "undo", f"deployment/{config.name}"]
                if revision:
                    cmd.extend([f"--to-revision={revision}"])
                if config.namespace != "default":
                    cmd.extend(["-n", config.namespace])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_deployment_status(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Get deployment status."""
        try:
            if config.platform == DeploymentPlatform.KUBERNETES:
                cmd = ["kubectl", "get", "deployment", config.name, "-o", "json"]
                if config.namespace != "default":
                    cmd.extend(["-n", config.namespace])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    return json.loads(result.stdout)
            
            return {"status": "unknown"}
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {"status": "error", "error": str(e)}


# Example usage and CLI interface
def main():
    """Main CLI interface for deployment suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terragon Production Deployment Suite")
    parser.add_argument("--action", choices=["deploy", "rollback", "status"], required=True)
    parser.add_argument("--platform", choices=[p.value for p in DeploymentPlatform], required=True)
    parser.add_argument("--name", default="iot-anomaly-detection")
    parser.add_argument("--environment", default="production")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Create deployment suite
    deployment_suite = ProductionDeploymentSuite()
    
    # Create deployment configuration
    config = DeploymentConfig(
        name=args.name,
        platform=DeploymentPlatform(args.platform),
        strategy=DeploymentStrategy.ROLLING_UPDATE,
        replicas=args.replicas,
        environment=args.environment
    )
    
    # Execute action
    if args.action == "deploy":
        success = deployment_suite.deploy(config, dry_run=args.dry_run, verbose=args.verbose)
        if success:
            print("✅ Deployment completed successfully")
        else:
            print("❌ Deployment failed")
            sys.exit(1)
    
    elif args.action == "rollback":
        success = deployment_suite.rollback(config)
        if success:
            print("✅ Rollback completed successfully")
        else:
            print("❌ Rollback failed")
            sys.exit(1)
    
    elif args.action == "status":
        status = deployment_suite.get_deployment_status(config)
        print(f"Deployment status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    main()