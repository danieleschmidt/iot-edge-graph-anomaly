#!/bin/bash

# =============================================================================
# IoT Edge Anomaly Detection - Development Environment Setup
# =============================================================================

set -euo pipefail

echo "🚀 Setting up IoT Edge Anomaly Detection development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# System Updates
# =============================================================================

log_info "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential development tools
log_info "Installing development tools..."
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    vim \
    nano \
    htop \
    tree \
    jq \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# =============================================================================
# Python Environment Setup
# =============================================================================

log_info "Setting up Python development environment..."

# Upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

# Install development dependencies
log_info "Installing Python development dependencies..."
pip install -e .[dev,test,lint,security,docs]

# Install additional development tools
pip install --upgrade \
    pip-tools \
    pip-audit \
    pre-commit \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    pytest-html \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    pytest-benchmark \
    tensorboard \
    wandb \
    mlflow

# =============================================================================
# Pre-commit Hooks Setup
# =============================================================================

log_info "Setting up pre-commit hooks..."
if [ -f .pre-commit-config.yaml ]; then
    pre-commit install --install-hooks
    pre-commit install --hook-type commit-msg
    log_success "Pre-commit hooks installed"
else
    log_warning "No .pre-commit-config.yaml found, skipping pre-commit setup"
fi

# =============================================================================
# Container Tools Setup
# =============================================================================

log_info "Setting up container development tools..."

# Install hadolint for Dockerfile linting
log_info "Installing Hadolint for Dockerfile linting..."
sudo wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
sudo chmod +x /usr/local/bin/hadolint

# =============================================================================
# Project Structure Setup
# =============================================================================

log_info "Creating project directories..."
mkdir -p logs data/raw data/processed models/checkpoints models/experiments
mkdir -p notebooks/exploratory notebooks/reports
mkdir -p .vscode outputs artifacts

# =============================================================================
# Git Configuration
# =============================================================================

log_info "Configuring Git..."

# Set up Git configuration if not already set
if [ -z "$(git config --global user.name 2>/dev/null || true)" ]; then
    log_info "Setting up Git configuration..."
    git config --global user.name "Developer"
    git config --global user.email "developer@terragon-labs.com"
    git config --global init.defaultBranch main
    git config --global pull.rebase false
fi

# Set up Git aliases for common operations
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'

log_success "Git configuration complete"

# =============================================================================
# VS Code Workspace Configuration
# =============================================================================

log_info "Setting up VS Code workspace..."

# Create workspace settings if they don't exist
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.terminal.activateEnvironment": false,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.provider": "isort",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.watcherExclude": {
        "**/.git/objects/**": true,
        "**/.git/subtree-cache/**": true,
        "**/node_modules/*/**": true,
        "**/__pycache__/**": true,
        "**/.pytest_cache/**": true,
        "**/.mypy_cache/**": true,
        "**/htmlcov/**": true,
        "**/.coverage": true
    },
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "editor.rulers": [88],
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true
}
EOF

# Create launch configuration for debugging
cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Main Application",
            "type": "python",
            "request": "launch",
            "module": "iot_edge_anomaly.main",
            "console": "integratedTerminal",
            "args": ["--config", "config/default.yaml"],
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Python: Advanced Main",
            "type": "python",
            "request": "launch",
            "module": "iot_edge_anomaly.advanced_main",
            "console": "integratedTerminal",
            "args": ["--enable-all-models"],
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Python: Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        }
    ]
}
EOF

log_success "VS Code workspace configured"

# =============================================================================
# Jupyter Setup
# =============================================================================

log_info "Setting up Jupyter environment..."
python -m ipykernel install --user --name iot-edge-anomaly --display-name "IoT Edge Anomaly"

# =============================================================================
# Environment File Setup
# =============================================================================

log_info "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    log_success "Environment file created from template"
    log_warning "Don't forget to update .env with your specific configuration"
else
    log_info "Environment file already exists"
fi

# =============================================================================
# Environment Validation
# =============================================================================

log_info "Validating development environment..."

# Check Python installation and key packages
if python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
    log_success "PyTorch installation verified"
else
    log_error "PyTorch installation failed"
fi

if python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')" 2>/dev/null; then
    log_success "PyTorch Geometric installation verified"
else
    log_warning "PyTorch Geometric may not be properly installed"
fi

# Generate initial test coverage report
log_info "Generating initial coverage report..."
if [ -f "Makefile" ]; then
    make test-coverage || log_warning "Test coverage generation failed - continuing setup"
else
    pytest --cov=src --cov-report=html tests/ || log_warning "Test coverage generation failed - continuing setup"
fi

# =============================================================================
# Development Shortcuts
# =============================================================================

log_info "Creating development shortcuts..."

# Create useful aliases in .bashrc and .zshrc
for rc_file in ~/.bashrc ~/.zshrc; do
    if [ -f "$rc_file" ]; then
        cat >> "$rc_file" << 'EOF'

# IoT Edge Anomaly Detection Development Aliases
alias ide-test="pytest tests/ -v"
alias ide-cov="pytest --cov=src --cov-report=html"
alias ide-lint="flake8 src tests && mypy src"
alias ide-format="black src tests && isort src tests"
alias ide-security="bandit -r src && safety check"
alias ide-build="docker build -t iot-edge-graph-anomaly ."
alias ide-run="python -m iot_edge_anomaly.main"
alias ide-advanced="python -m iot_edge_anomaly.advanced_main --enable-all-models"
alias ide-clean="find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
alias ide-jupyter="jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"

# Development environment
export PYTHONPATH="/workspace/src:$PYTHONPATH"
export DOCKER_BUILDKIT=1

EOF
    fi
done

# =============================================================================
# Setup Complete
# =============================================================================

log_success "🎉 Development environment setup complete!"

echo ""
echo "============================================================================="
echo "📋 Development Environment Summary"
echo "============================================================================="
echo ""
echo "🐍 Python Environment:"
echo "   - Python $(python --version | cut -d' ' -f2)"
echo "   - pip $(pip --version | cut -d' ' -f2)"
echo "   - Development packages installed"
echo ""
echo "🔧 Development Tools:"
echo "   - Pre-commit hooks configured"
echo "   - VS Code workspace settings created"
echo "   - Jupyter Lab environment ready"
echo ""
echo "🚀 Quick Start Commands:"
echo "   - ide-test       : Run all tests"
echo "   - ide-lint       : Run linting"
echo "   - ide-format     : Format code"
echo "   - ide-run        : Start application"
echo "   - ide-build      : Build Docker image"
echo "   - ide-jupyter    : Start Jupyter Lab"
echo ""
echo "📁 Project Structure Created:"
echo "   - src/           : Source code"
echo "   - tests/         : Test files"
echo "   - docs/          : Documentation"
echo "   - config/        : Configuration files"
echo "   - notebooks/     : Jupyter notebooks"
echo "   - models/        : Model artifacts"
echo ""
echo "📋 Next steps:"
echo "  1. Update .env file with your configuration"
echo "  2. Run 'ide-test' to verify everything works"
echo "  3. Run 'ide-run' to start the application"
echo "  4. Open Jupyter Lab with 'ide-jupyter'"
echo ""
echo "Ready to start developing! 🚀"
echo ""