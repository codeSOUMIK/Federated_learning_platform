#!/bin/bash

# Federated Learning Platform Setup Script
# This script automates the setup process for local development

echo "🚀 Setting up Federated Learning Platform..."
echo "=============================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js v18+ first."
    echo "   Download from: https://nodejs.org/"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18+ is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python $(python3 --version) detected"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi

echo "✅ Node.js dependencies installed"

# Setup Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv fl_env

# Activate virtual environment
source fl_env/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install flwr[simulation]>=1.6.0
pip install torch>=2.0.0 torchvision>=0.15.0
pip install numpy>=1.21.0 matplotlib>=3.5.0
pip install scikit-learn>=1.0.0 pandas>=1.3.0
pip install tqdm>=4.62.0 requests>=2.28.0
pip install psutil>=5.9.0

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "✅ Python dependencies installed"

# Create environment file
echo "⚙️  Creating environment configuration..."
cat > .env.local << EOF
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:3000

# Flower Server Configuration
FLOWER_SERVER_ADDRESS=localhost:8080
FLOWER_SERVER_ROUNDS=10

# Development Settings
NODE_ENV=development
EOF

echo "✅ Environment configuration created"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data models logs results configs

echo "✅ Project directories created"

# Build the Next.js application
echo "🔨 Building Next.js application..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Next.js application"
    exit 1
fi

echo "✅ Next.js application built successfully"

# Deactivate virtual environment
deactivate

echo ""
echo "🎉 Setup completed successfully!"
echo "=============================================="
echo ""
echo "To start the application:"
echo "1. Start the development server:"
echo "   npm run dev"
echo ""
echo "2. In a new terminal, activate Python environment:"
echo "   source fl_env/bin/activate"
echo ""
echo "3. Register a test client:"
echo "   python scripts/client_registration_example.py"
echo ""
echo "4. Open your browser and go to:"
echo "   http://localhost:3000"
echo ""
echo "For more detailed instructions, see SETUP_GUIDE.md"
echo ""
