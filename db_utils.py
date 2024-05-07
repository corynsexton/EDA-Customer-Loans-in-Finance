from sklearn.preprocessing import PowerTransformer
from scipy.stats import stats, normaltest
from sqlalchemy import create_engine
from statsmodels.graphics.gofplots import qqplot
from thefuzz import fuzz
import csv
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import pylab as py
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml

class RDSDatabaseConnector():
    
    def __init___(self, db_creds):
        self.db_creds = db_creds

    def read_db_creds(self):
        with open('credentials.yaml', 'r') as f:
            db_creds = yaml.safe_load(f)
        return db_creds
    
    def db_engine(self):
        db_creds = self.read_db_creds()

        db_type = 'postgresql'
        db_api = 'psycopg2'
        db_host = db_creds['RDS_HOST']
        db_password = db_creds['RDS_PASSWORD']
        db_user = db_creds['RDS_USER']
        db_database = db_creds['RDS_DATABASE']

        engine = create_engine(f"{db_type}+{db_api}://{db_user}:{db_password}@{db_host}/{db_database}")
        return engine

    def extract_data_from_db(self):
        engine = self.db_engine()
        loans_table = pd.read_sql_table("loan_payments", engine)
        
        return loans_table
    
    def save_df_to_csv_file(self):
        df = self.extract_data_from_db()
        csv_file = df.to_csv('loan_payments.csv', index=False)
        return csv_file
    

def read_csv_file(filename):
    """
    Reads specified CSV file
    """
    csv_file = csv.reader(filename)
    
    return csv_file


def save_changes(df, filename):
    """
    Saves data as new CSV file to current file path on the local machine
    """
    df.to_csv(filename, index=False)


class DataTransform():

    def read_data(self, filename):
        """
        Reads CSV file
        """
        df = pd.read_csv(filename)
        return df
    
    def change_dtype(self, df, column, dtype):
        """
        Changes data type of column specified
        """
        df[column] = df[column].astype(dtype)
 
    def change_column_to_datetime(self, df, column):
        """
        Changes column specified to datatime type
        """
        df[column] = df[column].apply(pd.to_datetime, errors='coerce')

    def extract_numerical_chars(self, df, column):
        """
        Extracts numerical characters from column entries
        """
        df[column] = df[column].astype(str)
        df[column] = df[column].str.extract('([\d.]+)')

    def rename_column(self, df, column_name, new_column_name):
        """
        Remanes column as specified
        """
        df.rename(columns={column_name : new_column_name}, inplace=True)



class DataFrameInfo():
    
    def describe_all_columns(self, df):
        """
        Shows statistical information of whole dataframe
        """
        print(df.describe())

    def size_of_dataframe(self, df):
        """
        Shows shape/size of whole dataframe
        """
        print(f'Size of dataframe: {df.shape}')

    def get_mean(self, df, column=None):
        """
        Returns mean of numeric column
        If no column provided, returns mean of whole dataframe for all numeric columns
        """
        if column != None:
            print(f'Mean of {column} is: \n{round(df[column].mean(numeric_only=True), 2)}')
        else:
            print(f'Mean of each numeric column is: \n{round(df.mean(), 2)}')

    def get_median(self, df, column=None):
        """
        Returns median of numeric column
        If no column provided, returns median of whole dataframe for all numeric columns
        """
        if column != None:
            print(f'Median of {column} is: \n{df[column].median(numeric_only=True)}')
        else:
            print(f'Median of each numeric column is: \n{df.median()}')
    
    def get_mode(self, df, column=None):
        """
        Returns mode of numeric column
        If no column provided, returns mode of whole dataframe for all numeric columns
        """
        if column != None:
            print(f'Most frequent value in {column} is: \n{df[column].mode()}')
        else:
            print('Please enter selected column')

    def get_std(self, df, column=None):
        """
        Returns std of numeric column
        If no column provided, returns mean of whole dataframe for all numeric columns
        """
        if column != None:
            print(f'Standard deviation of {column} is: \n{df[column].std(numeric_only=True)}')
        else:
            print(f'Standard deviation of each numeric column is: \n{df.std()}')

    def get_unique_values(self, df, column=None):
        """
        Checks all values in columns are unique
        """
        if column != None:
            print(f'Number of unique values in {column}: \n{df[column].nunique()}')
        else:
            print(f'Number of unique values in each column: \n{df.nunique()}')

    def get_distinct_categories(self, df, column):
        """
        Returns different categories in column and number of entries of each
        """
        if df[column].dtype == 'category':
            print(f'Categories in {column}: \n{df[column].value_counts()}')
        else:
            print("This column does is not type 'category'. Convert first." )

    def null_percentage(self, df):
        """
        Shows only columns containing nulls and returns null percentage
        """
        null_percentage = df.isnull().sum() * 100 /len(df)
        df_missing_val = pd.DataFrame({'percentage_missing' : null_percentage})
        df_missing_val = df_missing_val.loc[df_missing_val['percentage_missing'] > 0]
       
        return df_missing_val
    
    def dagostino_test(self, df, column):
        '''
        Performs D'Agostino's K^2 test - Shows probability that null hypothesis is false
        The probability estimate - p-value close to 0 means data are normally distributed.
        '''
        stat, p = normaltest(df[column], nan_policy='omit')
        print('Statistics = %.3f, p=%.3f' % (stat, p))

    def get_similarity_between_columns(self, df, column1, column2):
        """
        Provides similarity score between two columns
        """
        print(f"Similarity score: {fuzz.ratio(df[column1], df[column2])}")


    def ols_regression(self, df, column1, column2, column3):
        """
        Fits linear regression model
        """
        model = smf.ols(f'{column1} ~ {column2} + {column3}', df).fit()
        return model.summary()


    def r_squared_model(self, df, column1, column2):
        """
        Fits one model of two exogenous variables against each other
        """
        model = smf.ols(f'{column1} ~ {column2}', df).fit().rsquared
        print(f'R^2 for model: \n exog_{column1}_model: {model}')
        return model
        
    def r_squared_multi_model(self, df, column1, column2, column3):
        """
        Fits 3 models which model one of the exogenous variables against the other two
        """
        model1 = smf.ols(f'{column1} ~ {column2} + {column3}', df).fit().rsquared
        model2 = smf.ols(f'{column2} ~ {column1} + {column3}', df).fit().rsquared
        model3 = smf.ols(f'{column3} ~ {column1} + {column2}', df).fit().rsquared
        print(f'R^2 for model: \n exog_{column1}_model: {model1} \n exog_{column2}_model: {model2} \n exog_{column3}_model: {model3}')

        return model1, model2, model3


    def vif(self, r2_model):
        """
        Provides Variation Inflation Factor (VIF) score
        """
        vif_score =  1/(1-r2_model)
        print(f'R^2 model: {r2_model}, VIF score: {vif_score}')


class Plotter():
    """
    Class to visualise insights into the data
    """

    def null_matrix(self, df):
        """
        Shows matrix of missing values in all columns
        """
        print(msno.matrix(df))

    def bar_chart_of_nulls(self, df):
        """
        Shows bar chart of all columns and the number of entries against missing values
        """
        print(msno.bar(df))

    def correlation_heatmap_matrix(self, df):
        """
        Shows correlation heatmap
        """
        fig, ax = plt.subplots(figsize=(20,12))
        mask = np.triu(np.ones_like(df.corr()))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', mask=mask)
        plt.tight_layout()
        plt.show()

    def one_boxplot(self, df, column):
        df.boxplot(column)
        plt.show()

    def two_boxplots(self, df, column1, column2):
        df.boxplot(column=column1, by=column2, figsize=(5,6))
        plt.show()

    def histogram(self, df, column, bins):
        df[column].hist(bins=bins)
        plt.show()

    def scatterplot(self, df, x_column, y_column):
        sns.scatterplot(x=df[x_column], y=df[y_column])
        plt.show()

    def qq_plot(self, df, column):
        sm.qqplot(df[column], line='q')
        py.show()

    def IQR(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        print(f'Lower limit: {lower_limit}')
        print(f'Q1: {Q1}')
        print(f'IQR: {IQR}')
        print(f'Q2: {Q3}')
        print(f'Upper limit: {upper_limit}')

    def hist_skew(self, df, column, bins):
        """
        Tells us the skew value and shows in histogram and qqplot
        """
        print(f"Skew of {column} is {df[column].skew()}")
        df[column].hist(bins=bins)
        qqplot(df[column], scale=1, line='q', fit=True)
        plt.show()
        
    def check_skew(self, df):
        """
        This prints all skew values
        And shows all histograms so we can view
        """
        print(df.skew())
        df.hist(figsize=(20,15))
        plt.show()

    def box_plots(self, df, columns, nrows, ncols):
        """
        Plots box plots for all data variables
        """
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 30))
        axes = axes.flatten()
        for idx, column in enumerate(columns):
            df.boxplot(column=column, ax=axes[idx])
            axes[idx].set_title(column)
        plt.tight_layout()
        plt.show()

    def pie_chart(self, df, column):
        column_data = df.groupby([column])[column].count().reset_index(name='count')
        fig = px.pie(column_data, values='count', names=column, title=f'{column} of loans with loan status of Charged Off, Defaulted or Late payments.')
        fig.show()

    def kde_plot(self, df, column):
        sns.histplot(data=df, x=column, kde=True)
        sns.despine()

    def lineplot(self, df, column1, column2):
        col1 = df[column1].reset_index(drop=True)
        sns.lineplot(x=df[column2], y=df[column1], data=col1)

    def multi_lineplot(self, df, column1, column2, column3):
        col2 = df[column2].reset_index(drop=True)
        col3 = df[column3].reset_index(drop=True)
        sns.lineplot(x=df[column1], y=df[column2], data=col2)
        sns.lineplot(x=df[column1], y=df[column3], data=col3)


class DataFrameTransforms():
    """
    Class to perform EDA transformations on data
    """

    def drop_columns(self, df, columns):
        df = df.drop(columns=columns, axis=1)
        return df
    
    def drop_row(self, df, column):
        df.dropna(subset=[column], inplace=True)
    
    def impute_mode(self, df, column):
        df[column] = df[column].fillna(df[column].mode()[0])
        
    def impute_median(self, df, column):
        df[column] = df[column].fillna(df[column].median())

    def impute_from_other_column(self, df, column1, column2):
        df[column1] = df[column1].fillna(df[column2])


    def column_transform(self, df, column, method):
        '''
        Applies Yeo-Johnson or Box-Cox Transformation for positive skewness 
        ''' 
        print(f'Skewness of {column} before transformation: {df[column].skew()}')
        power_transformer = PowerTransformer(method=method)
        transformed_column = power_transformer.fit_transform(df[[column]])
        df[[column]] = transformed_column
        print(f'Skewness of {column} after transformation: {df[column].skew()}')

        return df, power_transformer
    

    def reverse_column_transform(self, df, column, power_transformer):
        '''
        Reverses Yeo-Johnson or Box-Cox Transformation back to original skew
        ''' 
        inverse_transformed_column = power_transformer.inverse_transform(df[[column]])
        df[[column]] = inverse_transformed_column
        print(f'Skewness of {column} after inverse transformation: {df[column].skew()}')


    def yj_transform(self, df, column):
        '''
        Applies Yeo-Johnson Transformation for positive skewness 
        ''' 
        print(f'Skewness of {column} before transformation: {df[column].skew()}')
        power_transformer = PowerTransformer(method='yeo-johnson')
        transformed_column = power_transformer.fit_transform(df[[column]])
        df[[column]] = transformed_column
        print(f'Skewness of {column} after transformation: {df[column].skew()}')

        return df, power_transformer


    def reverse_yj_transform(self, df, column, power_transformer):
        '''
        Reverses Yeo-Johnson Transformation back to original skew
        ''' 
        inverse_transformed_column = power_transformer.inverse_transform(df[[column]])
        df[[column]] = inverse_transformed_column
        print(f'Skewness of {column} after inverse transformation: {df[column].skew()}')

        return df


    def log_transform(self, df, column):
        """
        Applies Log Transformation
        Use for Count Type Data, Positively Skewed and Positive Data only
        """
        print(f'Original Skew: {df[column].skew()}')
        df[column] = df[column].map(lambda x: np.log(x) if x > 0 else 0)
        t=sns.histplot(df[column],label="Skewness: %.2f"%(df[column].skew()) )
        t.legend()
        print(f'Transformed Skew: {df[column].skew()}')
        
        return df


    def remove_IQR_outliers(self, df, column):
        """
        Finds upper and lower bounds of IQR
        And removes outliers outside of these ranges
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        print(f'Lower limit: {lower_limit}')
        print(f'Upper limit: {upper_limit}')
        print(f'IQR: {IQR}')

        df_outliers = df[~((df[column] < lower_limit) | (df[column]  > upper_limit))]
        
        return df_outliers
    

    def remove_zscore_outliers(self, df, column, threshold):
        """
        Finds z_scores and removes outliers outside the threshold given
        """
        z_scores = stats.zscore(df[column])

        outlier_mask = (abs(z_scores) > threshold)
        df_outliers = df[~outlier_mask]

        return df_outliers
    
    def remove_certain_value_outliers(self, df, column, threshold):
        """
        Removes outlier that is outside + or - chosen value
        """
        outlier_df = df[(df[column] < threshold) & (df[column] > threshold * -1)]
        return outlier_df
    


if __name__ == '__main__':
    dt = DataTransform()
    dfi = DataFrameInfo(dt)
    pl = Plotter(dt)
    transform = DataFrameTransforms(dt)
