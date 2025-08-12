#!/bin/bash
# Production entrypoint for Terragon IoT Edge Anomaly Detection

set -euo pipefail

# Default configuration
DEFAULT_CONFIG="config/production.yaml"
LOG_LEVEL=${LOG_LEVEL:-INFO}
WORKERS=${WORKERS:-1}
PORT=${PORT:-8080}

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Function to check health
check_health() {
    log "Performing health check..."
    python -c "
import sys
sys.path.append('/app/src')
from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
import torch
print('✅ Health check passed')
" || {
        log "❌ Health check failed"
        exit 1
    }
}

# Function to run the main application
run_app() {
    local config_file="$1"
    
    log "Starting Terragon IoT Edge Anomaly Detection v4.0"
    log "Configuration: $config_file"
    log "Log Level: $LOG_LEVEL"
    log "Workers: $WORKERS"
    log "Port: $PORT"
    
    # Set Python path
    export PYTHONPATH="/app/src:$PYTHONPATH"
    
    # Run the application
    if [[ "$config_file" == *"advanced"* ]]; then
        log "Running advanced ensemble mode..."
        python -m src.iot_edge_anomaly.advanced_main "$@"
    else
        log "Running standard mode..."
        python -m src.iot_edge_anomaly.main "$@"
    fi
}

# Function to run performance benchmark
run_benchmark() {
    log "Running performance benchmark..."
    
    export PYTHONPATH="/app/src:$PYTHONPATH"
    python -c "
import torch
import time
import sys
sys.path.append('/app/src')

from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
from iot_edge_anomaly.performance_optimizer import performance_monitor

print('🚀 Starting performance benchmark...')

# Create model
model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
optimized_model = performance_monitor.optimize_model_for_inference(model)

# Benchmark
test_data = torch.randn(10, 20, 5)
times = []

for i in range(100):
    start = time.time()
    with torch.no_grad():
        error = optimized_model.compute_reconstruction_error(test_data[0:1])
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
print(f'✅ Benchmark complete: {avg_time*1000:.2f}ms average inference time')

# Performance report
report = performance_monitor.get_performance_report()
print(f'📊 System memory: {report[\"system\"][\"memory_usage_mb\"]:.1f}MB')
print(f'📊 CPU usage: {report[\"system\"][\"cpu_percent\"]:.1f}%')
"
}

# Function to export models
export_models() {
    log "Exporting models for deployment..."
    
    export PYTHONPATH="/app/src:$PYTHONPATH"
    python -c "
import torch
import sys
sys.path.append('/app/src')

from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
from iot_edge_anomaly.performance_optimizer import performance_monitor

# Create and optimize model
model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
optimized_model = performance_monitor.optimize_model_for_inference(model)

# Save optimized model
torch.save(optimized_model.state_dict(), '/app/models/optimized_model.pth')
print('✅ Model exported to /app/models/optimized_model.pth')
"
}

# Main command dispatcher
case "${1:-run}" in
    "health")
        check_health
        ;;
    "benchmark")
        run_benchmark
        ;;
    "export")
        export_models
        ;;
    "run"|"--config"*)
        # Default run mode
        config_file="${2:-$DEFAULT_CONFIG}"
        run_app "$@"
        ;;
    *)
        log "Usage: $0 [health|benchmark|export|run] [options...]"
        log "Commands:"
        log "  health     - Run health check"
        log "  benchmark  - Run performance benchmark"
        log "  export     - Export optimized models"
        log "  run        - Run the application (default)"
        exit 1
        ;;
esac