# 🚀 Federated Learning Platform - Local Setup Guide

This comprehensive guide will help you set up and run the federated learning platform on your local device.

## 📋 Prerequisites

Before starting, ensure you have the following installed on your system:

### Required Software
- **Node.js** (v18.0.0 or higher) - [Download here](https://nodejs.org/)
- **npm** or **yarn** package manager
- **Git** for version control
- **Python** (v3.8 or higher) for FL scripts
- **pip** for Python package management

### Optional but Recommended
- **VS Code** or your preferred code editor
- **Docker** (for containerized deployment)
- **CUDA** (if you have NVIDIA GPU for accelerated training)

## 🛠️ Step-by-Step Setup

### Step 1: Clone and Setup the Project

\`\`\`bash
# Clone the repository (if you have it in a repo)
# Or create a new directory for your project
mkdir federated-learning-platform
cd federated-learning-platform

# If you're starting fresh, initialize the project
npm init -y
\`\`\`

### Step 2: Install Dependencies

\`\`\`bash
# Install all required Node.js dependencies
npm install next@14.2.25 react@19 react-dom@19 typescript@5.7.3

# Install UI and styling dependencies
npm install @radix-ui/react-accordion@1.2.2 @radix-ui/react-alert-dialog@1.1.4
npm install @radix-ui/react-avatar@1.1.2 @radix-ui/react-checkbox@1.1.3
npm install @radix-ui/react-dialog@1.1.4 @radix-ui/react-dropdown-menu@2.1.4
npm install @radix-ui/react-label@2.1.1 @radix-ui/react-progress@1.1.1
npm install @radix-ui/react-select@2.1.4 @radix-ui/react-separator@1.1.1
npm install @radix-ui/react-slider@1.2.2 @radix-ui/react-tabs@1.1.2
npm install @radix-ui/react-toast@1.2.4 @radix-ui/react-tooltip@1.1.6

# Install utility libraries
npm install class-variance-authority@0.7.1 clsx@2.1.1 tailwind-merge@2.5.5
npm install lucide-react@0.454.0 recharts@2.15.0 archiver@6.0.1
npm install react-hook-form@7.54.1 @hookform/resolvers@3.9.1 zod@3.24.1

# Install development dependencies
npm install -D @types/node@22 @types/react@18 @types/react-dom@18
npm install -D tailwindcss@3.4.17 postcss@8.5 autoprefixer@10.4.20
npm install -D tailwindcss-animate@1.0.7
\`\`\`

### Step 3: Setup Python Environment

\`\`\`bash
# Create a Python virtual environment
python -m venv fl_env

# Activate the virtual environment
# On Windows:
fl_env\Scripts\activate
# On macOS/Linux:
source fl_env/bin/activate

# Install Python dependencies for federated learning
pip install flwr[simulation]>=1.6.0
pip install torch>=2.0.0 torchvision>=0.15.0
pip install numpy>=1.21.0 matplotlib>=3.5.0
pip install scikit-learn>=1.0.0 pandas>=1.3.0
pip install tqdm>=4.62.0 requests>=2.28.0
pip install psutil>=5.9.0 GPUtil>=1.4.0
\`\`\`

### Step 4: Configure Next.js and Tailwind

Create the following configuration files:

**tailwind.config.ts:**
\`\`\`typescript
import type { Config } from "tailwindcss"

const config: Config = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config

export default config
\`\`\`

**postcss.config.js:**
\`\`\`javascript
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
\`\`\`

**next.config.mjs:**
\`\`\`javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['archiver']
  }
}

export default nextConfig
\`\`\`

### Step 5: Setup Environment Variables

Create a `.env.local` file in your project root:

\`\`\`bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:3000

# Flower Server Configuration (optional)
FLOWER_SERVER_ADDRESS=localhost:8080
FLOWER_SERVER_ROUNDS=10

# Development Settings
NODE_ENV=development
\`\`\`

### Step 6: Create Project Structure

Create the following directory structure:

\`\`\`
federated-learning-platform/
├── app/
│   ├── api/
│   │   ├── clients/
│   │   ├── projects/
│   │   ├── training/
│   │   └── models/
│   ├── clients/
│   ├── create-project/
│   ├── project/
│   ├── training/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── ui/
│   └── [other components]
├── lib/
├── hooks/
├── scripts/
│   ├── models/
│   ├── flower_server.py
│   ├── flower_client.py
│   └── [other scripts]
├── public/
└── [config files]
\`\`\`

### Step 7: Setup Global Styles

Create `app/globals.css`:

\`\`\`css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    --popover: 0 0% 100%;
    --popover-foreground: 222.2 84% 4.9%;
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
    --secondary: 210 40% 96%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    --muted: 210 40% 96%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96%;
    --accent-foreground: 222.2 47.4% 11.2%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;
    --radius: 0.5rem;
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;
    --popover: 222.2 84% 4.9%;
    --popover-foreground: 210 40% 98%;
    --primary: 210 40% 98%;
    --primary-foreground: 222.2 47.4% 11.2%;
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 212.7 26.8% 83.9%;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}
\`\`\`

### Step 8: Create Utility Functions

Create `lib/utils.ts`:

\`\`\`typescript
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
\`\`\`

## 🚀 Running the Application

### Step 1: Start the Development Server

\`\`\`bash
# Start the Next.js development server
npm run dev

# The application will be available at:
# http://localhost:3000
\`\`\`

### Step 2: Test Client Registration

In a new terminal, run the client registration example:

\`\`\`bash
# Activate Python environment
source fl_env/bin/activate  # or fl_env\Scripts\activate on Windows

# Run the client registration script
python scripts/client_registration_example.py
\`\`\`

### Step 3: Setup Flower Server (Optional)

For actual federated learning, start the Flower server:

\`\`\`bash
# In a new terminal, activate Python environment
source fl_env/bin/activate

# Start the Flower server
python scripts/flower_server.py
\`\`\`

### Step 4: Run Flower Clients (Optional)

Start multiple clients for testing:

\`\`\`bash
# Terminal 1 - Client 1
python scripts/flower_client.py --client-id 1 --server localhost:8080

# Terminal 2 - Client 2  
python scripts/flower_client.py --client-id 2 --server localhost:8080
\`\`\`

## 🎯 Usage Guide

### 1. Access the Web Interface
- Open your browser and go to `http://localhost:3000`
- You'll see the main dashboard with project management

### 2. Create a New Project
- Click "Create New Project"
- Fill in project details (name, model type, dataset)
- Configure training parameters
- Save the project

### 3. Register Clients
- Go to the "Clients" page (`/clients`)
- Click "Register New Client"
- Fill in client information or run the Python registration script
- Verify clients appear in the dashboard

### 4. Assign Clients to Projects
- Open a project page
- Go to the "Clients" tab
- Click "Add Clients"
- Select clients from the available list
- Confirm assignment

### 5. Start Training
- In the project page, click "Start Training"
- Configure training parameters
- Select participating clients
- Run pre-flight checks
- Launch training session

### 6. Monitor Training
- View real-time training progress
- Monitor client performance
- Check accuracy and loss metrics
- Download trained models

## 🔧 Troubleshooting

### Common Issues and Solutions

**Port Already in Use:**
\`\`\`bash
# Kill process using port 3000
npx kill-port 3000

# Or use a different port
npm run dev -- -p 3001
\`\`\`

**Python Dependencies Issues:**
\`\`\`bash
# Upgrade pip first
pip install --upgrade pip

# Install dependencies one by one if batch install fails
pip install flwr[simulation]
pip install torch torchvision
\`\`\`

**Node.js Version Issues:**
\`\`\`bash
# Check Node.js version
node --version

# Use Node Version Manager (nvm) to switch versions
nvm install 18
nvm use 18
\`\`\`

**Permission Issues (macOS/Linux):**
\`\`\`bash
# Fix npm permissions
sudo chown -R $(whoami) ~/.npm
\`\`\`

## 📊 Testing the System

### 1. Basic Functionality Test
- Create a test project
- Register 2-3 mock clients
- Assign clients to the project
- Start a training session with 3 rounds
- Verify training completes successfully

### 2. Client Management Test
- Register clients with different capabilities
- Test client filtering and selection
- Verify client status updates
- Test client removal from projects

### 3. Training Configuration Test
- Test different aggregation strategies
- Vary client participation fractions
- Test convergence thresholds
- Verify pre-flight checks work correctly

## 🚀 Production Deployment

For production deployment, consider:

1. **Database Setup**: Replace in-memory storage with PostgreSQL/MongoDB
2. **Authentication**: Add user authentication and authorization
3. **Security**: Implement API rate limiting and input validation
4. **Monitoring**: Add logging and monitoring systems
5. **Scaling**: Use Docker containers and orchestration
6. **SSL/TLS**: Enable HTTPS for secure communication

## 📞 Support

If you encounter issues:

1. Check the browser console for JavaScript errors
2. Check the terminal for server-side errors
3. Verify all dependencies are installed correctly
4. Ensure Python environment is activated
5. Check port availability and firewall settings

The platform is now ready for local development and testing! 🎉
