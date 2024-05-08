# EDA - Customer Loans in Finance
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![Postgresql](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white) ![AWS](https://img.shields.io/badge/Amazon_AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![Git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white) ![VSCode](	https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)</div>
___
## Table of Contents
[1. Project Description](#project-description)
[2. Installation](#installation)
[3. File Structure](#file-structure)
[4. License](#license)

___
## 1. Description
The focus of this exploratory data analysis (EDA) project is the examination of loan data within a large financial institution. 

The goal is to gain a thorough understanding of the diverse aspects of the loan data by revealing patterns, relationships, and anomalies through the application of data visualization and statistical techniques. This undertaking is pivotal for making well-informed decisions concerning loan approvals, risk management, and pricing strategies. 

By delving into the intricacies of the financial loan data, the project aspires to bolster the institution's decision-making capabilities, ultimately enhancing the overall performance and profitability of the loan portfolio.
___
## 2. Installation
**1. Clone the project repository from GitHub:**
```python
git clone https://github.com/your-username/your-repository.git
cd your-repository
```
**2. Install the dependencies:**
```python
pip install -r requirements.txt
```
**3. Set up database credentials:**
Create your own `credentials.yaml` file containing your PostgreSQL credentials.
___
## 3. File Structure
- `db_utils.py` - All code used in notebooks
- `credentials.yaml` - Contains credentials for database engine
- `loan_payments.csv` - Initial dataset extracted from RDS database
- `data_cleaning.ipynb` - Cleaning of data and imputation of null values
- `cleaned_loan_data.csv` - Data after cleaning and imputation
- `skew_transformation.ipynb` - Exploring skewness and data transformation
- `skew_data.csv` - Data after transformations to improve skewness
- `outliers.ipynb` - Detect outliers and leverages
- `outlier_loan_data.csv` - Data after handling outliers
- `collinearity.ipynb` - Identify relationship and highly correlated columns
- `transformed_data.csv` - Data after EDA process
- `loan_analysis` - Analysis of the cleaned loan data
___
## 4. License
**MIT License**

A short and simple permissive license with conditions only requiring preservation of copyright and license notices. 
Licensed works, modifications, and larger works may be distributed under different terms and without source code.



