import asyncio
import logging
import typer
import subprocess
import os
from pathlib import Path
import json
import yaml
from typing import Dict
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer()

class MonitoringSetup:
    def __init__(self):
        """Initialize monitoring setup"""
        self.project_root = Path(__file__).parent.parent.parent
        self.backend_dir = self.project_root / "backend"
        
    def setup_prometheus_config(self):
        """Set up Prometheus configuration"""
        config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "rag_api",
                    "static_configs": [
                        {
                            "targets": ["localhost:8001"]
                        }
                    ],
                    "metrics_path": "/metrics"
                }
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [
                            {
                                "targets": ["localhost:9093"]
                            }
                        ]
                    }
                ]
            },
            "rule_files": ["./rules/*.yml"]
        }
        
        # Create Prometheus config directory
        prometheus_dir = self.backend_dir / "monitoring" / "prometheus"
        prometheus_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config
        with open(prometheus_dir / "prometheus.yml", "w") as f:
            yaml.dump(config, f)
            
        logger.info("Created Prometheus configuration")
        
    def setup_grafana_config(self):
        """Set up Grafana configuration"""
        config = {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://prometheus:9090",
                    "isDefault": True
                }
            ],
            "dashboards": [
                {
                    "name": "RAG Pipeline",
                    "type": "file",
                    "options": {
                        "path": "/etc/grafana/provisioning/dashboards"
                    }
                }
            ]
        }
        
        # Create Grafana config directory
        grafana_dir = self.backend_dir / "monitoring" / "grafana"
        grafana_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config
        with open(grafana_dir / "datasources.yml", "w") as f:
            yaml.dump(config, f)
            
        logger.info("Created Grafana configuration")
        
    def setup_alert_rules(self):
        """Set up monitoring alert rules"""
        rules = {
            "groups": [
                {
                    "name": "rag_alerts",
                    "rules": [
                        {
                            "alert": "HighLatency",
                            "expr": "http_request_duration_seconds > 5",
                            "for": "5m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High request latency",
                                "description": "Request latency is above 5 seconds"
                            }
                        },
                        {
                            "alert": "HighErrorRate",
                            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) > 0.1",
                            "for": "5m",
                            "labels": {
                                "severity": "critical"
                            },
                            "annotations": {
                                "summary": "High error rate",
                                "description": "Error rate is above 10%"
                            }
                        },
                        {
                            "alert": "LowCacheHitRate",
                            "expr": "rate(rag_cache_hits_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m])) < 0.5",
                            "for": "15m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "Low cache hit rate",
                                "description": "Cache hit rate is below 50%"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Create rules directory
        rules_dir = self.backend_dir / "monitoring" / "prometheus" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        
        # Write rules
        with open(rules_dir / "alerts.yml", "w") as f:
            yaml.dump(rules, f)
            
        logger.info("Created alert rules")
        
    def setup_grafana_dashboard(self):
        """Set up Grafana dashboard"""
        dashboard = {
            "title": "RAG Pipeline Metrics",
            "panels": [
                {
                    "title": "Request Latency",
                    "type": "graph",
                    "datasource": "Prometheus",
                    "targets": [
                        {
                            "expr": "rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])",
                            "legendFormat": "{{method}} {{endpoint}}"
                        }
                    ]
                },
                {
                    "title": "Error Rate",
                    "type": "graph",
                    "datasource": "Prometheus",
                    "targets": [
                        {
                            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
                            "legendFormat": "{{method}} {{endpoint}}"
                        }
                    ]
                },
                {
                    "title": "Cache Performance",
                    "type": "graph",
                    "datasource": "Prometheus",
                    "targets": [
                        {
                            "expr": "rate(rag_cache_hits_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m]))",
                            "legendFormat": "Cache Hit Rate"
                        }
                    ]
                }
            ]
        }
        
        # Create dashboard directory
        dashboard_dir = self.backend_dir / "monitoring" / "grafana" / "dashboards"
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Write dashboard
        with open(dashboard_dir / "rag_dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=2)
            
        logger.info("Created Grafana dashboard")
        
    def setup_logging_config(self):
        """Set up logging configuration"""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "fmt": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "rag.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "formatter": "json"
                }
            },
            "loggers": {
                "": {
                    "handlers": ["console", "file"],
                    "level": "INFO"
                }
            }
        }
        
        # Create logging config directory
        log_dir = self.backend_dir / "monitoring" / "logging"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config
        with open(log_dir / "logging.json", "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info("Created logging configuration")
        
    def create_docker_compose(self):
        """Create Docker Compose file for monitoring services"""
        config = {
            "version": "3.8",
            "services": {
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "ports": ["9090:9090"],
                    "volumes": [
                        "./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml",
                        "./monitoring/prometheus/rules:/etc/prometheus/rules"
                    ]
                },
                "grafana": {
                    "image": "grafana/grafana:latest",
                    "ports": ["3000:3000"],
                    "volumes": [
                        "./monitoring/grafana/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml",
                        "./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards"
                    ],
                    "environment": {
                        "GF_SECURITY_ADMIN_PASSWORD": "admin"
                    }
                },
                "alertmanager": {
                    "image": "prom/alertmanager:latest",
                    "ports": ["9093:9093"],
                    "volumes": [
                        "./monitoring/alertmanager/config.yml:/etc/alertmanager/config.yml"
                    ]
                }
            }
        }
        
        # Write Docker Compose file
        with open(self.backend_dir / "monitoring" / "docker-compose.yml", "w") as f:
            yaml.dump(config, f)
            
        logger.info("Created monitoring Docker Compose file")

@app.command()
def setup():
    """Set up monitoring configuration"""
    setup = MonitoringSetup()
    
    try:
        setup.setup_prometheus_config()
        setup.setup_grafana_config()
        setup.setup_alert_rules()
        setup.setup_grafana_dashboard()
        setup.setup_logging_config()
        setup.create_docker_compose()
        
        logger.info("Monitoring setup completed successfully")
        
    except Exception as e:
        logger.error(f"Monitoring setup failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def start():
    """Start monitoring services"""
    try:
        subprocess.run(
            ["docker-compose", "-f", "monitoring/docker-compose.yml", "up", "-d"],
            cwd=str(Path(__file__).parent.parent),
            check=True
        )
        logger.info("Monitoring services started")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start monitoring services: {str(e)}")
        raise typer.Exit(1)

@app.command()
def stop():
    """Stop monitoring services"""
    try:
        subprocess.run(
            ["docker-compose", "-f", "monitoring/docker-compose.yml", "down"],
            cwd=str(Path(__file__).parent.parent),
            check=True
        )
        logger.info("Monitoring services stopped")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop monitoring services: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
