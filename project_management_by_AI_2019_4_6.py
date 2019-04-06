'''
Created on Apr 6, 2019

@author: Jinxu Ding

Analysis:

This is a supervised learning problem. 
Based on the given features, try to predict the schedule of a project.
It can be solved by random forest to do a classification about whether a project will be done timely or not.

The features that need to be considered for the machine learning model:
(1) Project Type 
(2) DSF
(3) Project Phase, Status, Actual Start Date and planed edn date
(4) School name and Project Description because they may have geo/FY/facility-information

Methodology:
Step 1: data cleaning by removing duplicated data or missing data
Step 2: use NLP to get geo-information from school name
Step 3: partition the dataset into two parts:  
            training set, test set  
Step 4: use random-forest algorithm to do classification based on the above features
Step 5: use training set to train the model and use test set to evaluate the model's accuracy.

'''
import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
np.random.seed(0)

class ProjectSolution():
    def __init__(self, fn):
        """ 
        Load data into a dataframe and check its validity and remove any duplicated rows 
        and rows with missing values change their types suitable for further analysis.
        
        Parameters
        ----------
        fn : str
            The file location of the spreadsheet
        
        Returns:
        ----------
            Some brief summaries about the cleaned data are printed out to console. 
        """
        # read file and load data
        self.__fn = fn 
        data_type = {'Project Geographic District': str, 'Project Building Identifier': str,
                     'Project School Name': str, 'Project Type': str,
                     'Project Description': str, 'Project Phase Name': str, 
                     'Project Status Name': str, 'Project Phase Actual Start Date': str, 
                     'Project Phase Planned End Date': str, 'Project Phase Actual End Date': str, 
                     'Project Budget Amount': int, 'Final Estimate of Actual Costs Through End of Phase Amount': int,
                     'Total Phase Actual Spending Amount': int, 'DSF Number(s)': str} 
        
        self.__df = pd.read_excel(self.__fn, sheet_name = 'AI Dataset', dtype = data_type, header = 0).drop_duplicates().dropna()
        print('After data cleaning, the input file has ' + str(len(self.__df)) + ' valid rows.')
        self.__df.rename(index = str, columns={'Project Geographic District': 'PGD', 
                                               'Project Building Identifier': 'PBI',
                                               'Project School Name': 'PSN', 
                                               'Project Type': 'PT', 
                                               'Project Description': 'PD',
                                               'Project Phase Name': 'PHN', 
                                               'Project Status Name': 'PSN', 
                                               'Project Phase Actual Start Date': 'PHASD', 
                                               'Project Phase Planned End Date' : 'PHPED', 
                                               'Project Phase Actual End Date': 'PHAED',
                                               'Project Budget Amount': 'PBA', 
                                               'Final Estimate of Actual Costs Through End of Phase Amount': 'FEAC',
                                               'Total Phase Actual Spending Amount' :'THAS', 
                                               'DSF Number(s)': 'DSF'
                                               })
        self.__df['PHASD'] = pd.to_datetime(self.__df['PHASD'], format='%m/%d/%Y')
        self.__df['PHPED'] = pd.to_datetime(self.__df['PHPED'], format='%Y-%m-%d')
        self.__df['PHAED'] = pd.to_datetime(self.__df['PHAED'], format='%Y-%m-%d')
        
        # label the predicted variable
        self.__df['ON_TIME'] = 1 if (self.__df['PHPED'] - self.__df['PHAED']).astype('timedelta64[m]') > 0 else 0
        
        
    def analyze_by_random_forest(self):
        
        # use NLP to extract more insights about the projects from 'Project Description'
        # use Chinking to get noun for possible school name, FY, facility name
        grammar = r"""
                    NP:
                    {<.*>+} # Chunk everything
                    }<NP>+{ # Chink sequences of Noun
                    """
        cp = nltk.RegexpParser(grammar)
        self.__df['PD_CHINK'] = cp.parse(self.__df['PD'])
            
        # use NLP to extract more insights about the projects from 'Project School Name'
        # use Chinking to get noun for possible school name, FY, facility name
        self.__df['PSN_CHINK'] = cp.parse(self.__df['PSN'])
        
        # create Training and test DataSet
        self.__df['is_train'] = np.random.uniform(0, 1, len(self.__df)) <= .75
        train, test = self.__df[self.__df['is_train'] == True], self.__df[self.__df['is_train'] == False]
        
        # get most possible features and then drop them if they are not important after the randomForest model is trained/tested
        features = self.__df.columns['PGD', 'PSN_CHINK', 'PD_CHINK', 'PT', 'PHN', 'PSN', 'PHASD', 'PHPED', 'PHAED', 'PBA', 'FEAC', 'THAS', 'DSF']
        
        # convert non-numerical column into a number
        for ft, col in zip(features.dttypes, features.columns):
            if not np.issubdtype(ft, np.datetime64):
                self._df[col] = pd.factorize(col)[0]
        
        
        # Train the random Forest classifier
        rand_forest_clf = RandomForestClassifier(n_jobs=2, random_state=0)
        
        # train the classifier 
        rand_forest_clf.fit(train[features], self.__df['ON_TIME'])

        # check it with test data
        rand_forest_clf.predict(test[features])
        
        # check the predicted probabilities
        rand_forest_clf.predict_proba(test[features])
        
        # evaluate the classifier's performance
        # by creating the confusion matrix to show the overal performance
        preds = rand_forest_clf.predict(test[features])
        pd.crosstab(test['ON_TIME'], preds, rownames=['Actual Results'], colnames=['Predicted Results'])
        
        # check feature importance
        # based on this, the important features can be identified.
        # then, go back to re-train the model by removing the unimportant features.
        list(zip(train[features], rand_forest_clf.feature_importances_))
        

if __name__ == '__main__':
    fn = 'C:\temp\AI Dataset.xlsx'
    ProjectSolution(fn)
#EOF