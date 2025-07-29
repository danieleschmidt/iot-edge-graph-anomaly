#!/bin/bash
# Development container setup script for IoT Edge Graph Anomaly Detection

set -e

echo "🚀 Setting up development environment..."

# Update pip and install project dependencies
echo "📦 Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev,test,lint,security,docs]

# Install and setup pre-commit hooks
echo "🎣 Setting up pre-commit hooks..."
pre-commit install --install-hooks

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs data/raw data/processed models/checkpoints

# Setup Git configuration if not already set
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Setting up Git configuration..."
    git config --global user.name "Developer"
    git config --global user.email "developer@terragon-labs.com"
    git config --global init.defaultBranch main
    git config --global pull.rebase false
fi

# Generate initial test coverage report
echo "📊 Generating initial coverage report..."
make test-coverage || echo "⚠️  Test coverage generation failed - continuing setup"

# Setup Jupyter kernel
echo "🪐 Setting up Jupyter kernel..."
python -m ipykernel install --user --name iot-edge-anomaly --display-name "IoT Edge Anomaly"

# Create local environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔧 Creating local environment file..."
    cp .env.example .env
    echo "⚠️  Don't forget to update .env with your specific configuration"
fi

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"

echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Update .env file with your configuration"
echo "  2. Run 'make test' to verify everything works"
echo "  3. Run 'make dev' to start development server"
echo "  4. Open Jupyter Lab at http://localhost:8888"
echo ""
echo "🆘 Need help? Run 'make help' to see available commands"