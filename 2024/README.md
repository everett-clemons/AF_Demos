# Manufacturing Process Optimization with Machine Learning
## Rockwell Automation Fair AI in MES Demo 2024

This repository demonstrates the process of building machine learning models to optimize manufacturing processes, as presented at Rockwell Automation's Automation Fair 2024. It serves as an educational resource showing how to approach manufacturing optimization problems using AI/ML techniques.

## 🎯 Project Overview

This project demonstrates:
1. Analyzing manufacturing process data
2. Building predictive models for key performance indicators (KPIs):
   - Throughput
   - First Pass Quality
   - Scrap Rate
3. Using these models to optimize process parameters
4. Generating natural language instructions for engineers and operators   

**Important Note:** This is NOT a plug-and-play solution for your manufacturing operation. Instead, it's an educational demonstration of the process and methodology you might consider following when developing your own AI/ML solutions. 

## ⚠️ Disclaimer and Code Status
This repository contains demonstration code developed specifically for educational purposes at the Rockwell Automation Fair. Please note:

### 🎯 Purpose

#### Educational resource demonstrating AI/ML concepts in manufacturing
#### Proof-of-concept implementation
#### Example workflow for similar projects
#### Teaching tool for industrial ML applications

### 🚫 Not Intended For
#### Production deployment
#### Direct implementation in live systems
#### Critical process control
#### Safety-related applications

### 📝 Code Status
#### The code is provided AS-IS and is:
#### Optimized for readability over performance
#### Simplified for demonstration purposes
#### Not exhaustively tested
#### Missing production-level error handling
#### Not regularly maintained

### 💡 Usage Guidelines
#### Use as a learning resource
#### Understand the concepts demonstrated
#### Adapt and enhance for your specific needs
#### Follow proper software development practices for production implementations
#### Consider safety and regulatory requirements

### 🔄 Contributions
#### While we appreciate the community's interest:
#### Pull requests are not being accepted
#### Issues can be opened for discussion purposes
#### Feel free to fork and adapt for your own use
#### Share your learnings through discussions

### ✅ Best Practices
When adapting this code, consider:

#### Adding comprehensive error handling
#### Implementing proper logging
#### Including thorough testing
#### Adding security measures
#### Optimizing performance
#### Following industry-specific standards


**Remember**: This is a teaching tool, not a turnkey solution. Always follow your organization's software development and safety guidelines when implementing AI/ML in industrial settings.

## 📊 Process Flow

1. **Data Analysis**
   - Correlation analysis between process parameters and KPIs
   - Feature importance identification
   - Data visualization

2. **Model Development**
   - Feature engineering
   - Model training with XGBoost
   - Hyperparameter optimization using Optuna
   - Model performance evaluation

3. **Process Optimization**
   - Multi-objective optimization considering:
     - Maximizing throughput
     - Maximizing first pass quality
     - Minimizing scrap
   - Parameter recommendation generation
   - Additional LLM (Large-Language Model) Insights

## 🛠️ Technical Stack

- **Python 3.8+**
- **Key Libraries:**
  - XGBoost for predictive modeling
  - Optuna for hyperparameter optimization
  - Pandas for data manipulation
  - Scikit-learn for data preprocessing
  - Matplotlib/Seaborn for visualization

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)

### 2. Installation Steps

#### Clone the Repository
```bash
git clone https://github.com/everett-clemons/AF_Demos.git
cd AF_Demos
```

#### Set Up Python Environment

1. Install Python 3.8 or higher:
   - Windows: Download from [Python.org](https://www.python.org/downloads/)
   - macOS: Use [Homebrew](https://brew.sh/): `brew install python`
   - Linux: `sudo apt install python3`

2. (Optional but recommended) Create a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 3. Configuration

1. Create a `.env` file in the project root:
```bash
# Create .env file
touch .env  # macOS/Linux
# or manually create in Windows
```

2. Add your OpenAI API key to the `.env` file:
```plaintext
OPENAI_API_KEY=your_api_key_here
```

### 4. Verify Installation

Run the following to verify your setup:
```bash
python -c "import pandas as pd; import numpy as np; import xgboost; print('Setup successful!')"
```

### 5. Common Issues & Solutions

- **Permission errors**: Use `sudo` (Linux/macOS) or run as administrator (Windows)
- **Package conflicts**: Try using a fresh virtual environment
- **OpenAI API errors**: Verify your API key and internet connection

## 📘 Key Components

1. **Data Preprocessing (`feature_engineering class`)**
   - Data cleaning and validation
   - Feature creation
   - Scaling and normalization

2. **Model Training (`model_trainer class`)**
   - XGBoost model configuration
   - Hyperparameter optimization
   - Cross-validation
   - Performance evaluation

3. **Process Optimization (`parameter_optimizer class`)**
   - Multi-objective optimization
   - Parameter bounds handling
   - Results visualization

## 📊 Example Results

The repository includes example outputs showing:
- Correlation analysis visualizations
- Feature importance rankings
- Model performance metrics
- Optimization results and recommendations

## ⚠️ Important Considerations

1. **Data Requirements**
   - This demo uses simulated data
   - Real implementations need careful data validation
   - Consider data quality and quantity requirements

2. **Model Limitations**
   - Models are only as good as their training data
   - Regular retraining may be necessary
   - Consider model explainability requirements

3. **Implementation Challenges**
   - Process constraints may vary
   - KPI priorities might be different
   - Safety considerations are paramount

## 🤝 Contributing

This is an educational resource. Feel free to:
1. Fork the repository
2. Alter anything
3. This code is free and open source. So use it at your will
4. Share your learnings and suggestions

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 📞 Contact

For questions about the demo or discussions about manufacturing optimization:
- Everett Clemons
- everett.clemons@rockwellautomation.com

## 🙏 Acknowledgments

- Rockwell Automation for the opportunity to present
- John Clemons and Tim Gellner of Rockwell Automation

---
*Last updated: December 2024*
