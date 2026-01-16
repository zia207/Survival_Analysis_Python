# Survival Analysis in Python

A comprehensive educational resource and interactive tutorial website covering all major methods and techniques in survival analysis using Python. This project provides detailed implementations, theoretical foundations, and practical examples for analyzing time-to-event data across various domains including medical research, engineering, and customer analytics.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Content Sections](#content-sections)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#datasets)
- [Python Packages](#python-packages)
- [Topics Covered](#topics-covered)
- [Contributing](#contributing)
- [Resources](#resources)
- [License](#license)
- [Author](#author)

## 🎯 Overview

**Survival Analysis in Python** is an educational website and tutorial series that provides a comprehensive introduction to survival analysis, also known as time-to-event analysis. This resource covers everything from fundamental non-parametric methods to advanced machine learning approaches, all implemented in Python.

Survival analysis is a statistical method that examines the time until a specific event occurs, such as death, disease relapse, machine failure, or customer churn. It is particularly valuable for studying event timing and accounts for cases where some individuals do not experience the event during the study period, known as censoring.

### Key Concepts Covered

- **Event**: Time to death, disease onset, equipment failure, contract duration, etc.
- **Survival Time**: Time from starting point to the event of interest
- **Censoring**: When the event hasn't occurred for some subjects during observation
- **Survival Function** ($S(t)$): Probability that time to event is greater than time $t$
- **Hazard Function** ($h(t)$): Instantaneous rate at which events occur at time $t$
- **Cumulative Hazard Function** ($H(t)$): Total risk of experiencing the event by time $t$

## 📁 Project Structure

```
Survival_Analysis_Python/
├── index.qmd                          # Main landing page and introduction
├── _quarto.yml                        # Quarto website configuration
├── _publish.yml                       # Publishing configuration
├── README.rmd                         # This file
├── styles.css                         # Custom CSS styling
│
├── Image/                             # Images and logos
│   ├── banner_survival_analysis_python.jpg
│   ├── chapter_logo_survival_analysis_python.png
│   └── ...
│
├── Data/                              # Datasets (referenced from GitHub)
│   └── lung_dataset.csv
│
├── Non-Parametric Methods                  
│   ├── 00-introduction
│   ├── 01-kaplan-meier
│   └── 02-nelson-aalen
│
├── Semi-Parametric Methods                   
│   ├── 00-introduction
│   ├── 01-cox-regression
│   ├── 02-time-dependent-covariates
│   └── 03-stratified-cox-model
│
├── Parametric Methods                     
│   ├── 00-introduction
│   ├── 01-exponential-model
│   ├── 02-weibull-model
│   ├── 03-log-normal-model
│   ├── 04-log-logistic-model
│   ├── 05-generalized-gamma-model
│   └── 06-gompertz-model
│
├── Recurrent Event Models                     
│   ├── 00-introduction
│   ├── 01-andersen-gill-model
│   ├── 02-prentice-williams-peterson-model
│   ├── 03-frailty-models
│   └── 04-marginal-models
│
├── Risk Regression                  
│   ├── 00-introduction
│   ├── 01-cause-specific-hazard-regression
│   ├── 02-subdistribution-hazard-regression
│   ├── 03-absolute-risk-regression
│   └── 04-aalen-model
│
├── Joint Modeling                    
│   ├── 00-introduction
│   ├── 01-standard-joint-model
│   ├── 02-baseline-hazard-function
│   ├── 03-causal-effects
│   ├── 04-competing-risks
│   ├── 05-dynamic-joint-model
│   ├── 06-joint-frailty-modeling
│   └── 07-PyMC-joint-modeling
│
└── Machine Learning Methods                     
    ├── 00-introduction
    ├── 01-cart
    ├── 02-random-survival-forest
    ├── 03-gradient-boosted
    ├── 04-svm
    ├── 05-deep-survival-cpu
    ├── 06-deep-survival-gpu
    ├── 07-deephit
    ├── 08-nnet-survival
    ├── 09-coxnnet
    ├── 10-coxnnet-omics-data
    ├── 11-lstm-cox
    └── 12-stack-ensemble
```

## 📚 Content Sections

### 1. Introduction to Survival Analysis 

Comprehensive overview covering:
- Key concepts and terminology
- Types of censoring (right, left, interval)
- Survival, hazard, and cumulative hazard functions
- Overview of all methods covered in the project
- Applications across different fields

### 2. Non-Parametric Methods 

Methods that make no assumptions about the underlying distribution:
- **Kaplan-Meier Estimator**: Most widely used method for estimating survival functions
- **Nelson-Aalen Estimator**: Estimates cumulative hazard function
- **Log-Rank Test**: Compares survival distributions between groups

**Key Features:**
- Manual implementation from scratch
- Step-by-step calculations
- Visualizations with confidence intervals
- Group comparisons and statistical testing

### 3. Semi-Parametric Methods (`02-07-02-*.qmd`)

Methods that assume a specific form for covariate relationships but not survival times:
- **Cox Proportional Hazards Model**: The most popular survival regression model
- **Time-Dependent Covariates**: Handling covariates that change over time
- **Stratified Cox Model**: Handling non-proportional hazards

**Key Features:**
- Manual Cox model implementation
- Univariate and multivariate regression
- Hazard ratio interpretation
- Proportional hazards assumption checking
- Baseline survival function estimation

### 4. Parametric Methods 

Methods that assume specific distributions for survival times:
- **Exponential Model**: Constant hazard rate
- **Weibull Model**: Flexible hazard (increasing/decreasing)
- **Log-Normal Model**: Skewed survival times
- **Log-Logistic Model**: Alternative skewed distribution
- **Generalized Gamma Model**: Very flexible, encompasses other models
- **Gompertz Model**: Exponential hazard increase

**Key Features:**
- Distribution-specific implementations
- Maximum likelihood estimation
- Model comparison and selection
- Goodness-of-fit assessments

### 5. Recurrent Event Models 

Models for multiple events occurring over time:
- **Andersen-Gill (AG) Model**: Extended Cox model for multiple events
- **Prentice-Williams-Peterson (PWP) Models**: Conditional models for ordered events
- **Frailty Models**: Accounting for unobserved heterogeneity
- **Marginal Models**: Treating each event as a separate observation

**Key Features:**
- Handling correlated events
- Gap time vs. total time approaches
- Random effects for patient-level clustering

### 6. Risk Regression 

Advanced regression methods for competing risks:
- **Cause-Specific Hazard Regression**: Modeling hazards for specific event types
- **Subdistribution Hazard Regression (Fine-Gray Model)**: Direct CIF modeling
- **Absolute Risk Regression**: Direct cumulative incidence function modeling
- **Aalen's Additive Regression Model**: Flexible time-varying effects

**Key Features:**
- Competing risks handling
- Cumulative incidence functions
- Time-varying covariate effects

### 7. Joint Modeling 

Models for longitudinal and survival data:
- **Standard Joint Model**: Shared random effects approach
- **Baseline Hazard Function**: Estimation in joint models
- **Causal Effects**: Causal inference in joint models
- **Competing Risks**: Joint models with multiple event types
- **Dynamic Joint Model**: Time-dependent predictions
- **Joint Frailty Models**: For recurrent and terminal events
- **PyMC Joint Modeling**: Bayesian approach using PyMC

**Key Features:**
- Longitudinal marker trajectories
- Association between markers and survival
- Dynamic prediction capabilities

### 8. Machine Learning Methods (`02-07-07-*.qmd`)

Modern ML approaches to survival analysis:
- **Survival Trees (CART)**: Decision trees for survival
- **Random Survival Forests**: Ensemble tree methods
- **Gradient Boosted Models**: XGBoost/LightGBM for survival
- **Support Vector Machines**: SVM-based survival models
- **DeepSurv**: Deep learning Cox model (CPU & GPU versions)
- **DeepHit**: Deep learning for competing risks
- **NNet-Survival**: Discrete-time neural survival models
- **CoxNNet**: Neural network extension of Cox model
- **LSTMCox**: LSTM-based Cox model for recurrent events
- **Stack Ensemble**: Combining multiple survival models

**Key Features:**
- Handling high-dimensional data
- Non-linear relationships
- Feature importance analysis
- Model ensemble strategies

## ✨ Key Features

- **Comprehensive Coverage**: From basic concepts to advanced ML methods
- **Hands-On Tutorials**: Step-by-step implementations with real datasets
- **Manual Implementations**: Understanding algorithms from scratch
- **Real-World Datasets**: Using actual medical and engineering data
- **Visualizations**: Publication-quality plots and survival curves
- **Code Examples**: Well-documented Python code in every section
- **Interactive Website**: Built with Quarto for easy navigation
- **Best Practices**: Following Python data science conventions

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Quarto (for building the website)
- Git (for version control)

### Python Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com/zia207/Survival_Analysis_Python.git
cd Survival_Analysis_Python
```

2. **Create a virtual environment:**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install required packages:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install lifelines scikit-survival pycox
pip install torch torchvision  # For deep learning models
pip install pymc  # For Bayesian joint modeling
```

Or install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
            lifelines scikit-survival pycox torch torchvision pymc
```

## 📊 Datasets

The project uses several datasets, including:

### Primary Dataset: Lung Cancer Dataset
- **Source**: Real-world clinical trial data
- **Variables**: Survival time, event status, age, sex, performance scores
- **Location**: Available on GitHub repository
- **Usage**: Primary example dataset for most tutorials

### Other Datasets
- **SUPPORT2**: Study to Understand Prognoses and Preferences
- **Synthetic Competing Risks**: Generated data for competing risks examples
- **TCGA Data**: The Cancer Genome Atlas data for omics examples

All datasets are available in the `Data/` directory or via GitHub links in the notebooks.

## 🐍 Python Packages

### Core Data Science Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations

### Survival Analysis Libraries
- **lifelines**: Comprehensive survival analysis library
  - Kaplan-Meier estimation
  - Cox regression
  - Parametric models
  - Competing risks
  - Time-varying covariates

- **scikit-survival**: Scikit-learn compatible survival analysis
  - CoxPH with sklearn API
  - Random survival forests
  - Gradient boosting for survival
  - Survival SVM

- **pycox**: Deep learning for survival analysis
  - DeepSurv
  - DeepHit
  - CoxTime
  - Neural MTLR

### Machine Learning Libraries
- **scikit-learn**: Machine learning utilities
- **torch**: PyTorch for deep learning models
- **pymc**: Bayesian modeling and probabilistic programming
- **Keras and Tensorflow**

## 📖 Topics Covered

### Statistical Methods
1. ✅ Non-parametric survival estimation
2. ✅ Semi-parametric regression (Cox models)
3. ✅ Parametric survival distributions
4. ✅ Recurrent event analysis
5. ✅ Competing risks analysis
6. ✅ Joint modeling of longitudinal and survival data
7. ✅ Frailty models
8. ✅ Time-dependent covariates

### Machine Learning Methods
1. ✅ Tree-based methods (CART, Random Forests)
2. ✅ Gradient boosting
3. ✅ Support Vector Machines
4. ✅ Deep learning (DeepSurv, DeepHit)
5. ✅ Neural networks for survival
6. ✅ Ensemble methods

### Advanced Topics
1. ✅ Model diagnostics and validation
2. ✅ Feature importance analysis
3. ✅ Model comparison and selection
4. ✅ Handling missing data
5. ✅ Hyperparameter tuning
6. ✅ Cross-validation strategies

## 🚀 Usage

### Running Individual Tutorials

Each `.qmd` file is a self-contained tutorial. To run a specific tutorial:

1. Open the `.qmd` file in RStudio, VS Code with Quarto extension, or Jupyter
2. Execute the code chunks interactively
3. Or render to HTML: `quarto render filename.qmd`

### Using Code from Tutorials

You can copy code chunks from any tutorial and adapt them for your own data:

```python
# Example: Basic Kaplan-Meier analysis
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations=df['time'], event_observed=df['event'])
kmf.plot_survival_function()
```


## 🤝 Contributing

Contributions are welcome! Areas for contribution include:

- Additional survival analysis methods
- More example datasets
- Improved visualizations
- Code optimizations
- Documentation improvements
- Translation to other languages

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Resources

### Textbooks
- Klein & Moeschberger — *Survival Analysis: Techniques for Censored and Truncated Data*
- Collett — *Modelling Survival Data in Medical Research*
- Therneau & Grambsch — *Modeling Survival Data: Extending the Cox Model*
- Hosmer, Lemeshow, May — *Applied Survival Analysis*

### Online Resources
- [lifelines Documentation](https://lifelines.readthedocs.io/)
- [scikit-survival Documentation](https://scikit-survival.readthedocs.io/)
- [pycox Documentation](https://pycox.readthedocs.io/)
- [Quarto Documentation](https://quarto.org/)

### Python Tutorials
- [Cameron Davidson-Pilon's Survival Analysis Book](https://dataorigami.net/books/survival-analysis-in-python/)
- [Towards Data Science Survival Analysis](https://towardsdatascience.com/survival-analysis-with-python-907d8405e0e4)
- [Real Python Survival Analysis Guide](https://realpython.com/python-survival-analysis/)

## 📄 License

This project is licensed under CC-By (Creative Commons Attribution). See the footer in `_quarto.yml` for details.

**Copyright**: © CC-By Zia Ahmed, University at Buffalo 2026

## 👤 Author

**Zia Ahmed**

- **Website**: [R Data Science Environment](https://sites.google.com/view/r-data-sci-env/home)
- **GitHub**: [@zia207](https://github.com/zia207)
- **LinkedIn**: [Zia Ahmed](https://www.linkedin.com/in/zia-ahmed207)
- **Email**: zia207@gmail.com
- **Affiliation**: University at Buffalo
- **Organization**: RENEW Institute

## 🌟 Acknowledgments

- **RENEW Institute** for support and resources
- **University at Buffalo** for institutional support
- Contributors and users of the survival analysis Python packages
- The open-source community for amazing tools and libraries

## 📝 Citation

If you use this resource in your research or teaching, please cite:

```bibtex
@misc{ahmed2024survivalpython,
  title={Survival Analysis in Python: A Comprehensive Guide},
  author={Ahmed, Zia},
  year={2024},
  publisher={University at Buffalo},
  url={https://github.com/zia207/Survival_Analysis_Python}
}
```

## 🔗 Links

- **Live Website**: [Quarto Pub](https://zia207.quarto.pub/survival-analysis-in-python)
- **GitHub Repository**: [Survival_Analysis_Python](https://github.com/zia207/Survival_Analysis_Python)
- **Colab Notebooks**: Available in the repository's `Colab_Notebook/` directory

## 📈 Future Plans

- [ ] Add more real-world case studies
- [ ] Include Bayesian survival analysis tutorials
- [ ] Add interactive visualizations
- [ ] Expand competing risks examples
- [ ] Include more high-dimensional genomics examples
- [ ] Add R comparison tutorials (already done)
- [ ] Create video tutorials
- [ ] Develop accompanying exercises and quizzes

---

**Note**: This is an educational resource. Always consult with domain experts and statisticians when applying these methods to real-world problems, especially in clinical or safety-critical applications.

