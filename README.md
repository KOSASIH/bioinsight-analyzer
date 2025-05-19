# BioInsight Analyzer

A powerful tool for researchers that analyzes complex biomedical data, generating insights and predictive models to accelerate drug discovery.

## Features

- **Principal Component Analysis (PCA)**: Reduce dimensionality of complex biomedical data while preserving variance
- **Clustering Analysis**: Group similar samples to identify natural subgroups in your data
- **Predictive Modeling**: Build models to predict drug response based on gene expression data

## Getting Started

### Prerequisites

- Python 3.8+
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bioinsight-analyzer.git
   cd bioinsight-analyzer
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

## Usage

1. **Upload Data**: Upload your biomedical data in CSV format or use the provided sample data
2. **Select Analysis Type**: Choose from PCA, Clustering, or Predictive Modeling
3. **Analyze**: Click the "Analyze Data" button to process your data
4. **View Results**: Explore visualizations, statistics, and insights generated from your data

## Sample Data

The application includes sample biomedical data that simulates gene expression profiles and drug response measurements. You can use this sample data to explore the features of BioInsight Analyzer.

## Screenshots

![BioInsight Analyzer Screenshot](screenshot.png)

## License

This project is licensed under the MIT License - see the LICENSE file for details.