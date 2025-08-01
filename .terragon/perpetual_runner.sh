#!/bin/bash
#
# Terragon Perpetual Value Discovery and Execution
# Continuous SDLC enhancement with intelligent scheduling
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TERRAGON_DIR="${REPO_ROOT}/.terragon"
LOG_FILE="${TERRAGON_DIR}/perpetual.log"

# Ensure log directory exists
mkdir -p "${TERRAGON_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

run_discovery() {
    log "🔍 Running value discovery cycle..."
    cd "${REPO_ROOT}"
    
    if python3 "${TERRAGON_DIR}/discover_value.py"; then
        log "✅ Value discovery completed successfully"
        return 0
    else
        log "❌ Value discovery failed"
        return 1
    fi
}

run_execution() {
    log "🚀 Running autonomous execution cycle..."
    cd "${REPO_ROOT}"
    
    if python3 "${TERRAGON_DIR}/autonomous_executor.py"; then
        log "✅ Autonomous execution completed successfully"
        return 0
    else
        log "❌ Autonomous execution failed"
        return 1
    fi
}

run_security_scan() {
    log "🔒 Running security scan..."
    
    # Check for new vulnerabilities
    if command -v safety >/dev/null 2>&1; then
        if safety check --json > "${TERRAGON_DIR}/security-scan.json" 2>/dev/null; then
            log "✅ Security scan completed - no vulnerabilities found"
        else
            log "⚠️  Security scan found potential vulnerabilities"
            # Re-run discovery to capture security items
            run_discovery
        fi
    else
        log "ℹ️  Safety tool not available - skipping security scan"
    fi
}

run_performance_check() {
    log "⚡ Running performance check..."
    
    # Check if performance tests exist and run them
    if [ -f "${REPO_ROOT}/tests/performance/test_benchmarks.py" ]; then
        cd "${REPO_ROOT}"
        if python3 -m pytest tests/performance/ -v --tb=short > "${TERRAGON_DIR}/performance-results.txt" 2>&1; then
            log "✅ Performance tests passed"
        else
            log "⚠️  Performance regression detected"
            # Re-run discovery to capture performance issues
            run_discovery
        fi
    else
        log "ℹ️  No performance tests found - skipping performance check"
    fi
}

run_comprehensive_analysis() {
    log "📊 Running comprehensive analysis..."
    
    # Run discovery, then execution if items found
    if run_discovery; then
        # Check if high-value items were discovered
        if [ -f "${TERRAGON_DIR}/value-metrics.json" ]; then
            high_priority_count=$(python3 -c "
import json
try:
    with open('${TERRAGON_DIR}/value-metrics.json', 'r') as f:
        data = json.load(f)
    print(data.get('value_metrics', {}).get('high_priority_items', 0))
except:
    print(0)
")
            
            if [ "${high_priority_count}" -gt 0 ]; then
                log "📈 Found ${high_priority_count} high-priority items - running execution"
                run_execution
            else
                log "📊 No high-priority items found - repository in good state"
            fi
        fi
    fi
}

show_status() {
    log "📊 Terragon Autonomous SDLC Status"
    log "   Repository: iot-edge-graph-anomaly"
    log "   Maturity: ADVANCED (92%)"
    log "   Mode: Perpetual Value Discovery"
    
    if [ -f "${TERRAGON_DIR}/value-metrics.json" ]; then
        total_items=$(python3 -c "
import json
try:
    with open('${TERRAGON_DIR}/value-metrics.json', 'r') as f:
        data = json.load(f)
    print(data.get('discovery_stats', {}).get('total_items', 0))
except:
    print(0)
")
        log "   Items in backlog: ${total_items}"
    fi
    
    if [ -f "${TERRAGON_DIR}/execution-metrics.json" ]; then
        log "   Last execution: $(python3 -c "
import json
try:
    with open('${TERRAGON_DIR}/execution-metrics.json', 'r') as f:
        data = json.load(f)
    if data:
        print(data[-1].get('timestamp', 'Unknown'))
    else:
        print('Never')
except:
    print('Unknown')
")"
    fi
}

main() {
    case "${1:-help}" in
        "discovery")
            run_discovery
            ;;
        "execution")
            run_execution
            ;;
        "security")
            run_security_scan
            ;;
        "performance")
            run_performance_check
            ;;
        "comprehensive")
            run_comprehensive_analysis
            ;;
        "status")
            show_status
            ;;
        "continuous")
            log "🔄 Starting continuous perpetual execution..."
            while true; do
                log "⏰ Running scheduled comprehensive analysis..."
                run_comprehensive_analysis
                log "😴 Sleeping for 1 hour until next cycle..."
                sleep 3600  # 1 hour
            done
            ;;
        "help"|*)
            echo "Terragon Perpetual SDLC Runner"
            echo ""
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  discovery      - Run value discovery cycle"
            echo "  execution      - Run autonomous execution cycle"
            echo "  security       - Run security vulnerability scan"
            echo "  performance    - Run performance regression check"
            echo "  comprehensive  - Run full analysis and execution"
            echo "  continuous     - Run continuous perpetual execution (1h intervals)"
            echo "  status         - Show current system status"
            echo "  help           - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 comprehensive    # Run full cycle once"
            echo "  $0 continuous       # Run perpetually (background with nohup)"
            echo "  $0 status          # Check current state"
            ;;
    esac
}

main "$@"