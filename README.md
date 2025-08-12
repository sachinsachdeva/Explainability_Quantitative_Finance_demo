# Explainable AI Demo with LLM Agent for Finance

This demo showcases how a Large Language Model (LLM) agent can orchestrate tools to explain quantitative financial models trained on tabular data. It combines prediction, explanation, and monitoring capabilities to deliver role-specific insights for clients, regulators, and executives.

## Features

- Synthetic credit approval dataset with sensitive attributes for bias analysis.
- Gradient Boosting model for loan approval prediction.
- Tools exposing:
  - Prediction on individual customers.
  - Local explanation using SHAP values.
  - Model drift detection via statistical tests.
  - Bias audit calculating fairness metrics.
- DSPy + Google Gemini LLM integration for agent-driven interactive explainability.
- Role-based output formatting: plain language, technical audit, and executive summary.

## Setup

1. Clone or download the gist.
2. Install required Python packages:

   ```bash
   pip install numpy pandas scikit-learn shap matplotlib dspy
