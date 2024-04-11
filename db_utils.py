# Extract data from the RDS database
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import PowerTransformer
from scipy.stats import normaltest, zscore
from sqlalchemy import create_engine
from thefuzz import fuzz
import csv
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
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


# if __name__ == '__main__' :
#     db_connector = RDSDatabaseConnector()
#     db_connector.read_db_creds()
#     db_connector.db_engine()
#     db_connector.extract_data_from_db()
#     db_connector.save_df_to_csv_file()
    

def read_csv_file():
    read_loan_csv = csv.reader('loan_payments.csv')
    
    return read_loan_csv


def save_changes(df, filename):
    """
    Saves data as new CSV file to current file path on the local machine
    """
    df.to_csv(filename, index=False)


class DataTransform():

    def read_data(self, filename):
        df = pd.read_csv(filename)

        return df


    def correct_column_format(self):
        df = self.read_data('loan_payments.csv')
        """
        Change column datatype and format of columns where necessary
        """
        category_cols = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'payment_plan', 'purpose', 'application_type', 'employment_length']
        df[category_cols] = df[category_cols].astype('category')
        datetime_cols = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']
        df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime, format='%b-%Y', errors='coerce')
        datetime_types = df.select_dtypes('datetime64')
        df[datetime_types.columns] = datetime_types.apply(lambda x: x.dt.strftime('%b-%Y'))
        df['loan_amount'] = df['loan_amount'].astype(float)
        df['term'] = df['term'].astype(str)
        df['term'] = df['term'].str.extract('([\d.]+)').astype('category')
        df.rename(columns={'term' : 'term (months)'}, inplace=True)
        
        return df
    
    def change_dtype(self, column, dtype):
        df = self.read_data('loan_payments.csv')
        df[column] = df[column].astype(dtype)
        return df

    def change_column_to_datetime(self, column, format):
        df = self.read_data('loan_payments.csv')
        df[column] = df[column].apply(pd.to_datetime, format=format, errores='coerce')

        return df
    
    def extract_numerical_chars(self, column):
        df = self.read_data('loan_payments.csv')
        df[column] = df[column].astype(str)
        df[column] = df[column].str.extract('([\d.]+)')
        return df

    def rename_column(self, column_name, new_column_name):
        df = self.read_data('loan_payments.csv')
        df.rename(columns={column_name : new_column_name}, inplace=True)
        return df




class DataFrameInfo():
    
    def __init__(self, dt):
        self.df = dt.correct_column_format()

    def describe_all_columns(self):
        """
        Shows statistical information of whole dataframe
        """
        print(self.df.describe())

    def size_of_dataframe(self):
        """
        Shows shape/size of whole dataframe
        """
        print(self.df.shape)

    def get_mean(self, column=None):
        """
        Returns mean of numeric column
        If no column provided, returns mean of whole dataframe for all numeric columns
        """
        if column != None:
            print(f'Mean of {column} is: \n{round(self.df[column].mean(numeric_only=True), 2)}')
        else:
            print(f'Mean of each numeric column is: \n{round(self.df.mean(), 2)}')

    def get_median(self, column=None):
        """
        Returns median of numeric column
        If no column provided, returns median of whole dataframe for all numeric columns
        """
        if column != None:
            print(f'Median of {column} is: \n{self.df[column].median(numeric_only=True)}')
        else:
            print(f'Median of each numeric column is: \n{self.df.median()}')
    
    def get_mode(self, column=None):
        """
        Returns mode of numeric column
        If no column provided, returns mode of whole dataframe for all numeric columns
        """
        if column != None:
            print(f'Most frequent value in {column} is: \n{self.df[column].mode()}')
        else:
            print('Please enter selected column')

    def get_std(self, column=None):
        """
        Returns std of numeric column
        If no column provided, returns mean of whole dataframe for all numeric columns
        """
        if column != None:
            print(f'Standard deviation of {column} is: \n{self.df[column].std(numeric_only=True)}')
        else:
            print(f'Standard deviation of each numeric column is: \n{self.df.std()}')

    def get_unique_values(self, column=None):
        """
        Checks all values in columns are unique
        """
        if column != None:
            print(f'Number of unique values in {column}: \n{self.df[column].nunique()}')
        else:
            print(f'Number of unique values in each column: \n{self.df.nunique()}')

    def get_distinct_categories(self, column):
        """
        Returns different categories in column and number of entries of each
        """
        if self.df[column].dtype == 'category':
            print(f'Categories in {column}: \n{self.df[column].value_counts()}')
        else:
            print("This column does is not type 'category'. Convert first." )

    def null_percentage(self):
        """
        Shows only columns containing nulls and returns null percentage
        """
        null_percentage = self.df.isnull().sum() * 100 /len(self.df)
        df_missing_val = pd.DataFrame({'percentage_missing' : null_percentage})
        df_missing_val = df_missing_val.loc[df_missing_val['percentage_missing'] > 0]
       
        return df_missing_val
    
    def dagostino_test(self, column):
        '''
        Performs D'Agostino's K^2 test - Shows probability that null hypothesis is false
        The probability estimate - p-value close to 0 means data are normally distributed.
        '''
        data = self.df[column]
        stat, p = normaltest(data, nan_policy='omit')
        print('Statistics = %.3f, p=%.3f' % (stat, p))

    def get_similarity_with_other_column(self, column1, column2):
        print(f"Similarity score: {fuzz.ratio(self.df[column1], self.df[column2])}")







class Plotter():
    """
    Class to visualise insights into the data
    """
    # Inherit initialised class from earlier class
    def __init__(self, dt):
        self.df = dt.correct_column_format()
        # def __init__(self):
        #     super(DataFrameInfo, self).__init__()

    def null_matrix(self):
        """
        Shows matrix of missing values in all columns
        """
        print(msno.matrix(self.df))

    def bar_chart_of_nulls(self):
        """
        Shows bar chart of all columns and the number of entries against missing values
        """
        print(msno.bar(self.df))

    def one_boxplot(self, column):
        self.df.boxplot(column)
        plt.show()

    def two_boxplots(self, column1, column2):
        self.df.boxplot(column=column1, by=column2, figsize=(5,6))
        plt.show()

    def histogram(self, column, bins):
        self.df[column].hist(bins=bins)

    def scatterplot(self, x_column, y_column):
        sns.scatterplot(x=self.df[x_column], y=self.df[y_column])
        plt.show()

    def qq_plot(self, column):
        qqplot(self.df[column], scale=1, line='q', fit=True)
        plt.show()

    def hist_skew(self, column, bins):
        """
        Tells us the skew value and shows in histogram and qqplot
        """
        print(f"Skew of {column} is {self.df[column].skew()}")
        self.df[column].hist(bins=bins)
        qqplot(self.df[column], scale=1, line='q', fit=True)
        plt.show()
        
    def check_skew(self):
        """
        This prints all skew values
        And shows all histograms so we can view
        """
        print(self.df.skew())
        self.df.hist(figsize=(20,15))
        plt.show()
        
        
            


class DataFrameTransforms():
    """
    Class to perform EDA transformations on data

    Drop columns - only if high percentage of nulls 
    Drop rows - not recommended for a dataset with lots of columns
    Impute - Mean, Median, Mode, Constant Value (like 'Unknown' or '0'), 
    Impute - Regression - Recommended when the column is highly correlated with one or more other columns
    KNN - Uses the k most similar distance (average for numerical variables, mode for categorical)

    """

    def __init__(self, dt):
        self.df = dt.correct_column_format()

    
    def drop_columns(self):
        """
        Drop columns with high percentage of nulls
        """
        self.df = self.df.drop(columns=['mths_since_last_delinq', 'mths_since_last_record', 'next_payment_date', 'mths_since_last_major_derog'], axis=1)
    
        return self.df
    
    def drop_row(self, column):
        self.df.dropna(subset=[column], inplace=True)

        return self.df
    
    def impute_mode(self, column):
        self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        
        return self.df
    
    def impute_median(self, column):
        self.df[column] = self.df[column].fillna(self.df[column].median())

        return self.df

    def impute_from_other_column(self, column1, column2):
        self.df[column1] = self.df[column1].fillna(self.df[column2])
        
        return self.df
    
    def yj_transform(self, column):
        '''
        Applies Yeo-Johnson Transformation for positive skewness 
        ''' 
        print(f'Original Skew: {self.df[column].skew()}')
        power_transformer = PowerTransformer(method='yeo-johnson')
        self.df[[column]] = power_transformer.fit_transform(self.df[[column]])
        print(f'Transformed Skew: {self.df[column].skew()}')
    

    def new_yj_transform(self, column, reverse=None):
        '''
        Applies Yeo-Johnson Transformation and reverses back to Original Skew if 'reverse' is stated.
        ''' 
        print(f'Original Skew: {self.df[column].skew()}')
        power_transformer = PowerTransformer(method='yeo-johnson')
        self.df[[column]] = power_transformer.fit_transform(self.df[[column]])
        if reverse == 'reverse':
            print(f'Transformed Skew: {self.df[column].skew()}')
            self.df[[column]] = power_transformer.inverse_transform(self.df[[column]])
            print(f'Reversed Skew: {self.df[column].skew()}')
            self.df[[column]] = power_transformer.inverse_transform(self.df[[column]])
            print(f'Reversed Skew: {self.df[column].skew()}')
        else:
            print(f'Transformed Skew: {self.df[column].skew()}')


    def reverse_yj_transform(self, column):
        '''
        Reverses Yeo-Johnson Transformation back to original skew
        ''' 
        print(f'Transformed Skew: {self.df[column].skew()}')
        power_transformer = PowerTransformer(method='yeo-johnson')
        self.df[[column]] = power_transformer.inverse_transform(self.df[column])
        print(f'Original Skew: {self.df[column].skew()}')


    def boxcox_transform(self, column):
        """
        Applies Box-Cox Transformation to strictly positive values only
        """
        print(f'Original Skew: {self.df[column].skew()}')
        power_transformer = PowerTransformer(method='box-cox')
        self.df[[column]] = power_transformer.fit_transform(self.df[[column]])
        print(f'Transformed Skew: {self.df[column].skew()}')


    # def yj_transform_old(self, column):
    #     '''
    #     Applies Yeo-Johnson Transformation for positive skewness 
    #     ''' 
    #     print(f'Original Skew: {self.df[column].skew()}')
    #     yj_transform = pd.Series(yeojohnson(self.df[column])[0])
    #     t=sns.histplot(yj_transform,label="Skewness: %.2f"%(yj_transform.skew()) )
    #     t.legend()
    #     self.df[column] = yj_transform
    #     print(f'Transformed Skew: {self.df[column].skew()}')


    def log_transform(self, column):
        """
        Applies Log Transformation
        Use for Count Type Data, Positively Skewed and Positive Data only
        """
        print(f'Original Skew: {self.df[column].skew()}')
        self.df[column] = self.df[column].map(lambda x: np.log(x) if x > 0 else 0)
        t=sns.histplot(self.df[column],label="Skewness: %.2f"%(self.df[column].skew()) )
        t.legend()
        print(f'Transformed Skew: {self.df[column].skew()}')



if __name__ == '__main__':
    dt = DataTransform()
    dfi = DataFrameInfo(dt)
    pl = Plotter(dt)
    transform = DataFrameTransforms(dt)

    transform.drop_columns()
    transform.impute_mode('collections_12_mths_ex_med')
    transform.impute_mode('employment_length')
    transform.impute_mode('term (months)')
    transform.impute_median('int_rate')
    transform.impute_from_other_column('funded_amount', 'loan_amount')
    transform.drop_row('last_payment_date')
    transform.drop_row('last_credit_pull_date')
    
    transform.yj_transform('instalment')

    # pl.qq_plot('loan_amount')
    # pl.scatterplot('loan_amount', 'funded_amount')
    # pl.check_skew()

    # dt.read_data()
    # dfi.get_distinct_categories()
    # dt.correct_column_format()
    # dfi.describe_all_columns()
    # dfi.size_of_dataframe()
    # dfi.get_mean(column=['loan_amount', 'total_payment'])
    # dfi.get_median('loan_amount')
    # dfi.get_mode(dt)
    # dfi.get_std(dt)
    # dfi.get_unique_values()

