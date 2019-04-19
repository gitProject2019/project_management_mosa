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
(3) Project Phase, Status, Actual Start Date and planed ending date
(4) School name and Project Description because they may have geo/FY/facility-information

Methodology:
Step 1: data cleaning by removing duplicated data or missing data
Step 2: use NLP to get information about project types
Step 3: partition the dataset into two parts:  
            training set, test set  
Step 4: use random-forest algorithm to do classification based on the above features
Step 5: use training set to train the model and use test set to evaluate the model's accuracy.


Further work:
In order to improve the prediction accuracy, it is helpful to explore some follow-up questions: 
(1) weather conditions ? 
(2) project manager experiences ?
(3) local regulations about civil engineering projects ? (green energy building ? e.g. LEED, Leadership in Energy and Environmental Design)
'''

import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from nltk.tree import Tree
from sklearn import preprocessing
from datetime import datetime
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
                     'Project Budget Amount': str, 'Final Estimate of Actual Costs Through End of Phase Amount': str,
                     'Total Phase Actual Spending Amount': int, 'DSF Number(s)': str} 
        try:
            self.__df = pd.read_excel(self.__fn, sheet_name = 'AI Dataset', dtype = data_type, header = 0, delim_whitespace=True).drop_duplicates().dropna()
             
        except NameError:
            print('file not found!')
        
        except Exception as e:
            print('reading file with error :' + str(e))
        
        original_len = len(self.__df)
        print('After data cleaning, the input file has ' + str(original_len ) + ' rows.')
        self.__df.columns = self.__df.columns.str.strip()
        self.__df.rename(columns={'Project Geographic District': 'PGD',
                                               'Project Building Identifier': 'PBI',
                                               'Project School Name': 'PSN',
                                               'Project Type': 'PT',
                                               'Project Description': 'PD',
                                               'Project Phase Name': 'PHN', 
                                               'Project Status Name': 'PST', 
                                               'Project Phase Actual Start Date': 'PHASD', 
                                               'Project Phase Planned End Date' : 'PHPED', 
                                               'Project Phase Actual End Date': 'PHAED',
                                               'Project Budget Amount': 'PBA', 
                                               'Final Estimate of Actual Costs Through End of Phase Amount': 'FEAC',
                                               'Total Phase Actual Spending Amount' :'THAS', 
                                               'DSF Number(s)': 'DSF'
                                               }, inplace = True)
        
        print('before drop rows with illegal datetime format, the input file has ' + str(original_len) + ' rows.')
        
        # remove rows of their 'PHASD', or 'PHPED' or 'PHAED' that are not in correct datetime format.
        self.__df['PHASD'] = pd.to_datetime(self.__df['PHASD'], format='%Y-%m-%d',  errors = 'coerce')
        self.__df['PHPED'] = pd.to_datetime(self.__df['PHPED'], format='%Y-%m-%d', errors = 'coerce')
        self.__df['PHAED'] = pd.to_datetime(self.__df['PHAED'], format='%Y-%m-%d', errors = 'coerce')
        self.__df['PBA'] = pd.to_numeric(self.__df['PBA'], downcast = 'float', errors = 'coerce')
        self.__df['FEAC'] = pd.to_numeric(self.__df['FEAC'],  downcast = 'float', errors = 'coerce')
        self.__df['THAS'] = pd.to_numeric(self.__df['THAS'], downcast = 'float', errors = 'coerce')
        self.__df = self.__df.dropna(subset = ['PHASD', 'PHPED', 'PHAED', 'PBA', 'FEAC', 'THAS'])
        
        self.__df.dropna(axis=0)
        original_len_1 =  len(self.__df)
        print('after dropping rows with illegal datetime format, the input file has ' + str(original_len_1) + ' valid rows with ' 
              + str(round((original_len - original_len_1)/original_len * 100.0, 2)) + '% of the rows dropped.')
        
        # label the predicted variable
        self.__df['ON_TIME'] = (self.__df['PHPED'] - self.__df['PHAED']).astype('timedelta64[D]').astype(int)
        
        # create some date features
        self.__df['PHASD_YEAR'] = pd.DatetimeIndex(self.__df['PHASD']).year[0] 
        self.__df['PHASD_MONTH'] = pd.DatetimeIndex(self.__df['PHASD']).month[0] 
        
        self.__df['PHPED_YEAR'] = pd.DatetimeIndex(self.__df['PHPED']).year[0] 
        self.__df['PHPED_MONTH'] = pd.DatetimeIndex(self.__df['PHPED']).month[0] 
        
        self.__df['PHAED_YEAR'] = pd.DatetimeIndex(self.__df['PHAED']).year[0] 
        self.__df['PHAED_MONTH'] = pd.DatetimeIndex(self.__df['PHAED']).month[0] 
        
        # create some budget features
        self.__df['TIME_ESTIMATE'] = (self.__df['PHPED'] - self.__df['PHASD']).astype('timedelta64[D]').astype(int)
        self.__df['BUDGET_RUNOVER'] = (self.__df['THAS'] == self.__df['PBA']).astype(int)
        self.__df['BUDGET_RUNOVER_ESTIMATE'] = (self.__df['FEAC'] == self.__df['PBA']).astype(int)
        
        self.__df['ON_TIME'] = self.__df['ON_TIME'].apply(lambda x : 1 if x >= 0 else 0)
        
    def analyze_by_random_forest(self):
        
        # create training and test DataSet
        self.__df['is_train'] = np.random.uniform(0, 1, len(self.__df)) <= .85
        self.__train, self.__test = self.__df[self.__df['is_train'] == True], self.__df[self.__df['is_train'] == False]
        
        self.__train_chunked, self.__test_chunked = [], []
        
        for data_set in (self.__train, self.__test):
            
            sentences = [] 
            for row in data_set['PD']:
                row = row.replace('/', ' ')
                sentence = nltk.sent_tokenize(row)
                sentence = [nltk.word_tokenize(sent) for sent in sentence]
                sentence = [nltk.pos_tag(sent) for sent in sentence]
                sentences.append(sentence)
            
            # use NLP to extract more insights about the projects from 'Project Description'
            # use Chinking to get noun for possible school name, FY, facility name
            grammar = r"""
                        NP: {<DT|NN.*>+} # Chunk sequences of DT, JJ, NN
                        DT: }<DT>{ # Chink sequences of DT
                        IN: }<IN>{ # Chink sequences of IN
                        JJ: }<JJ>{ # Chink sequences of JJ                    
                    """          
            cp = nltk.RegexpParser(grammar)
            if data_set is self.__train:
                chunked = self.__train_chunked
            else:
                chunked = self.__test_chunked
            for sentence in sentences:
                for sent in sentence:
                    for s in cp.parse(sent):
                        if s[-1] in ('IN', 'DT', 'JJ'):
                            continue
                        t = []
                        if type(s) is Tree:
                            if s.label() == 'NP':
                                t.extend([x[0] for x in s.leaves()])
                        elif type(s) is tuple:
                            t.extend(s[0])
                    chunked.append(' '.join(t))
        
        self.__train['PD_NLP'] = pd.Series(self.__train_chunked)
        self.__test['PD_NLP'] = pd.Series(self.__test_chunked)
        self.__train['PD_NLP'].astype(str)
        self.__test['PD_NLP'].astype(str)
        
        # evaluated the cp performance
        # get most possible features and then drop them if they are not important after the randomForest model is trained/tested
        features = ['PGD', 'PD_NLP', 'PT', 'PHN', 'PST', 'PHASD', 'PHPED', 'PHAED', 'PBA', 'FEAC', 'THAS', 'DSF']
        ''' 
        Predicted Results    0   1
        Actual Results            
        0                  115  42
        1                  137  96
        Accuracy = 54.1%
        Precision = 73.25%
        Recall = 45.63%
        True Negative Rate = 69.57%
        True Positive Rate = 45.63%
        Balanced Accuracy = 57.6%
        f_score = 1.7553081534418096
        [('PGD', 0.044400393751473866), ('PD_NLP_NUM', 0.016161363728547522), ('PT_NUM', 0.03885061283583019), ('PHN_NUM', 0.02213157054085249), ('PST_NUM', 0.0), ('PHASD_NUM', 0.11885367792577754), ('PHPED_NUM', 0.2473641460127869), ('PHAED_NUM', 0.17355018098196656), ('PBA', 0.07529751990197274), ('FEAC', 0.08928385044441696), ('THAS', 0.08593032252447722), ('DSF_NUM', 0.08817636135189796)]
        '''
        #features = ['PD_NLP', 'PT', 'PHN', 'PHASD', 'PHPED', 'PBA', 'FEAC', 'THAS']
        ''' 
        Predicted Results    0   1
        Actual Results            
        0                  135  22
        1                  174  59
        Accuracy = 49.74%
        Precision = 85.99%
        Recall = 43.69%
        True Negative Rate = 72.84%
        True Positive Rate = 43.69%
        Balanced Accuracy = 58.26%
        f_score = 1.6726428479298339
        [('PD_NLP_NUM', 0.0396840748664649), ('PT_NUM', 0.048475702376965094), ('PHN_NUM', 0.033219934478092183), ('PHASD_NUM', 0.1903697010748368), ('PHPED_NUM', 0.273902705607454), ('PBA', 0.11619593574517754), ('FEAC', 0.13226494868077437), ('THAS', 0.16588699717023503)]

        '''
        #features = ['PD_NLP', 'PT', 'PHN', 'PHASD_YEAR', 'PHASD_MONTH',  'PHPED_YEAR', 'PHPED_MONTH' , 
        #             'PHAED_YEAR',  'PHAED_MONTH', 'PBA', 'FEAC', 'THAS']
        '''  
        Predicted Results   0    1
        Actual Results            
        0                  69   88
        1                  63  170
        Accuracy = 61.28%
        Precision = 43.95%
        Recall = 52.27%
        True Negative Rate = 65.89%
        True Positive Rate = 52.27%
        Balanced Accuracy = 59.08%
        f_score = 2.10944008418468
        [('PD_NLP_NUM', 0.0853350228092222), ('PT_NUM', 0.0391615804350083), ('PHN_NUM', 0.05490074050655712), ('PHASD_YEAR', 0.0), ('PHASD_MONTH', 0.0), ('PHPED_YEAR', 0.0), ('PHPED_MONTH', 0.0), ('PHAED_YEAR', 0.0), ('PHAED_MONTH', 0.0), ('PBA', 0.23051786508381827), ('FEAC', 0.28575159331713385), ('THAS', 0.3043331978482602)]

        '''
        
        #features = ['PD_NLP', 'PT', 'PHN', 'PBA', 'FEAC', 'THAS']
        ''' 
        Predicted Results   0    1
        Actual Results            
        0                  69   88
        1                  86  147
        Accuracy = 55.38%
        Precision = 43.95%
        Recall = 44.52%
        True Negative Rate = 62.55%
        True Positive Rate = 44.52%
        Balanced Accuracy = 53.54%
        f_score = 2.0077614379084965
        [('PD_NLP_NUM', 0.08787346207948757), ('PT_NUM', 0.04218997281856189), ('PHN_NUM', 0.057412138946347654), ('PBA', 0.23936964748452128), ('FEAC', 0.2844332609155825), ('THAS', 0.28872151775549904)]

        '''
        
        #features = ['PD_NLP', 'PT', 'PBA', 'FEAC', 'THAS']
        ''' 
        Predicted Results   0    1
        Actual Results            
        0                  63   94
        1                  79  154
        Accuracy = 55.64%
        Precision = 40.13%
        Recall = 44.37%
        True Negative Rate = 62.1%
        True Positive Rate = 44.37%
        Balanced Accuracy = 53.24%
        f_score = 2.062082092830299
        [('PD_NLP_NUM', 0.08168740895116247), ('PT_NUM', 0.058635254138564866), ('PBA', 0.2448850816853207), ('FEAC', 0.30547347859645907), ('THAS', 0.30931877662849294)]

        '''
        
        #features = ['PT', 'PBA', 'FEAC', 'THAS']
        ''' 
        Predicted Results   0    1
        Actual Results            
        0                  83   74
        1                  91  142
        Accuracy = 57.69%
        Precision = 52.87%
        Recall = 47.7%
        True Negative Rate = 65.74%
        True Positive Rate = 47.7%
        Balanced Accuracy = 56.72%
        f_score = 1.9401574195539777
        [('PT_NUM', 0.05572370400858062), ('PBA', 0.26318680312692266), ('FEAC', 0.3227149681211723), ('THAS', 0.3583745247433244)]
        '''
        
        #features = ['PBA', 'FEAC', 'THAS']
        ''' 
        Predicted Results   0    1
        Actual Results            
        0                  80   77
        1                  78  155
        Accuracy = 60.26%
        Precision = 50.96%
        Recall = 50.63%
        True Negative Rate = 66.81%
        True Positive Rate = 50.63%
        Balanced Accuracy = 58.72%
        f_score = 1.9961095610484538
        [('PBA', 0.2895537486303944), ('FEAC', 0.3566504126213195), ('THAS', 0.3537958387482861)]
        '''
        #features = ['PBA', 'FEAC', 'THAS', 'BUDGET_RUNOVER', 'BUDGET_RUNOVER_ESTIMATE']
        ''' 
        Predicted Results   0    1
        Actual Results            
        0                  84   73
        1                  79  154
        Accuracy = 61.03%
        Precision = 53.5%
        Recall = 51.53%
        True Negative Rate = 67.84%
        True Positive Rate = 51.53%
        Balanced Accuracy = 59.68%
        f_score = 1.977742627951644
        [('PBA', 0.28337326520712863), ('FEAC', 0.363099301732332), ('THAS', 0.34013786544267427), ('BUDGET_RUNOVER', 0.004724024003040658), ('BUDGET_RUNOVER_ESTIMATE', 0.008665543614824441)]
        '''
        
        #features = ['PT', 'PD_NLP', 'BUDGET_RUNOVER', 'BUDGET_RUNOVER_ESTIMATE']
        '''  
        Predicted Results    0   1
        Actual Results            
        0                  128  29
        1                  145  88
        Accuracy = 55.38%
        Precision = 81.53%
        Recall = 46.89%
        True Negative Rate = 75.21%
        True Positive Rate = 46.89%
        Balanced Accuracy = 61.05%
        f_score = 1.7214015710034587
        [('PT_NUM', 0.20741611962161038), ('PD_NLP_NUM', 0.16232984239400508), ('BUDGET_RUNOVER', 0.0033845717513915673), ('BUDGET_RUNOVER_ESTIMATE', 0.011774857406320286), ('TIME_ESTIMATE', 0.6150946088266728)]

        '''       
        
        #features = ['PT', 'PD_NLP', 'BUDGET_RUNOVER', 'BUDGET_RUNOVER_ESTIMATE', 'TIME_ESTIMATE']
        ''' 
        Predicted Results    0   1
        Actual Results            
        0                  128  29
        1                  145  88
        Accuracy = 55.38%
        Precision = 81.53%
        Recall = 46.89%
        True Negative Rate = 75.21%
        True Positive Rate = 46.89%
        Balanced Accuracy = 61.05%
        f_score = 1.7214015710034587
        [('PT_NUM', 0.20741611962161038), ('PD_NLP_NUM', 0.16232984239400508), ('BUDGET_RUNOVER', 0.0033845717513915673), ('BUDGET_RUNOVER_ESTIMATE', 0.011774857406320286), ('TIME_ESTIMATE', 0.6150946088266728)]
        '''
        
        #features = ['PT', 'PD_NLP', 'TIME_ESTIMATE']
        ''' 
        Predicted Results    0   1
        Actual Results            
        0                  124  33
        1                  141  92
        Accuracy = 55.38%
        Precision = 78.98%
        Recall = 46.79%
        True Negative Rate = 73.6%
        True Positive Rate = 46.79%
        Balanced Accuracy = 60.19%
        f_score = 1.733754238923658
        [('PT_NUM', 0.19896021825198257), ('PD_NLP_NUM', 0.16115652449675658), ('TIME_ESTIMATE', 0.6398832572512608)]

        '''
        #features = ['PD_NLP', 'TIME_ESTIMATE']
        ''' 
        Predicted Results    0    1
        Actual Results             
        0                  107   50
        1                  106  127
        Accuracy = 60.0%
        Precision = 68.15%
        Recall = 50.23%
        True Negative Rate = 71.75%
        True Positive Rate = 50.23%
        Balanced Accuracy = 60.99%
        f_score = 1.833472725583124
        [('PD_NLP_NUM', 0.18363923414366087), ('TIME_ESTIMATE', 0.8163607658563391)]

        '''
        #features = ['TIME_ESTIMATE'] 
        ''' 
        Predicted Results   0    1
        Actual Results            
        0                  92   65
        1                  64  169
        Accuracy = 66.92%
        Precision = 58.6%
        Recall = 58.97%
        True Negative Rate = 72.22%
        True Positive Rate = 58.97%
        Balanced Accuracy = 65.6%
        f_score = 2.003783617956846
        [('TIME_ESTIMATE', 1.0)]

        '''
        #features = ['PT', 'PD_NLP', 'TIME_ESTIMATE']
        ''' 
        Predicted Results    0   1
        Actual Results            
        0                  124  33
        1                  141  92
        Accuracy = 55.38%
        Precision = 78.98%
        Recall = 46.79%
        True Negative Rate = 73.6%
        True Positive Rate = 46.79%
        Balanced Accuracy = 60.19%
        f_score = 1.733754238923658
        [('PT_NUM', 0.19896021825198257), ('PD_NLP_NUM', 0.16115652449675658), ('TIME_ESTIMATE', 0.6398832572512608)]

        '''
        #features = ['PT', 'TIME_ESTIMATE']
        ''' 
        PPredicted Results    0    1
        Actual Results             
        0                  106   51
        1                   86  147
        Accuracy = 64.87%
        Precision = 67.52%
        Recall = 55.21%
        True Negative Rate = 74.24%
        True Positive Rate = 55.21%
        Balanced Accuracy = 64.72%
        f_score = 1.8864705339850596
        [('PT_NUM', 0.2304332557276556), ('TIME_ESTIMATE', 0.7695667442723444)]

        '''
        #features = ['PT', 'TIME_ESTIMATE', 'PSN']
        ''' 
        Predicted Results    0    1
        Actual Results             
        0                  100   57
        1                  100  133
        Accuracy = 59.74%
        Precision = 63.69%
        Recall = 50.0%
        True Negative Rate = 70.0%
        True Positive Rate = 50.0%
        Balanced Accuracy = 60.0%
        f_score = 1.8652382202388766
        [('PT_NUM', 0.09772917777744022), ('TIME_ESTIMATE', 0.4127442538706245), ('PSN_NUM', 0.48952656835193525)]
        '''
        
        features_new = []
        
        # convert non-numerical column into a number
        for ft in features:
            if np.issubdtype(self.__train[ft].dtype, np.int) or np.issubdtype(self.__train[ft].dtype, np.float):
                features_new.append(ft)
            else:
                le = preprocessing.LabelEncoder()
                self.__train[str(ft +'_NUM')] = le.fit_transform(self.__train[ft].astype(str))
                features_new.append(ft +'_NUM')
            
            if not np.issubdtype(self.__test[ft].dtype, np.int) and not np.issubdtype(self.__test[ft].dtype, np.float):
                le = preprocessing.LabelEncoder()
                self.__test[str(ft +'_NUM')] = le.fit_transform(self.__test[ft].astype(str))
                
        # train the random forest classifier
        rand_forest_clf = RandomForestClassifier(n_jobs=2, random_state=0)
        
        # train the classifier 
        rand_forest_clf.fit(self.__train[features_new], self.__train['ON_TIME'])

        # test it with the test dataset
        rand_forest_clf.predict(self.__test[features_new])
        
        # check the predicted probabilities
        rand_forest_clf.predict_proba(self.__test[features_new])
        
        # evaluate the classifier's performance
        # by creating the confusion matrix to show the overall performance
        preds = rand_forest_clf.predict(self.__test[features_new])
        conf_matrix = pd.crosstab(self.__test['ON_TIME'], preds, rownames=['Actual Results'], colnames=['Predicted Results'])
        print(conf_matrix)
       
        tp, tn, fp, fn = conf_matrix[0][0], conf_matrix[1][1], conf_matrix[1][0],conf_matrix[0][1]
        print('Accuracy = ' + str(round((tp + tn) / (tp + tn + fn + fp) * 100.0, 2)) + "%")
        
        precision = round(tp / (tp + fp) * 100.0, 2)
        print('Precision = ' + str(precision) + "%")
        
        recall = round(tp / (tp + fn) * 100.0, 2)
        print('Recall = ' + str(recall) + "%")
        
        tnr  = round(tn / (tn + fp) * 100.0, 2)
        print('True Negative Rate = ' + str(tnr) + "%")
        
        tpr  = round(tp / (tp + fn) * 100.0, 2)
        print('True Positive Rate = ' + str(tpr) + "%")
        print('Balanced Accuracy = ' + str(round((tnr + tpr)/2, 2)) + "%")
        
        beta = 2.0
        beta_2 = beta * beta
        f_score = (1 + beta_2 ) * (precision + recall) / (beta_2 * precision + recall)
        print('f_score = ' + str(f_score))
        
        # check feature importance
        # based on this, the important features can be identified.
        # then, go back to re-train the model by removing the unimportant features.
        print(list(zip(self.__train[features_new], rand_forest_clf.feature_importances_)))
        

if __name__ == '__main__':
    fn = r'C:\temp\AI_Dataset.xlsx'
    ProjectSolution(fn).analyze_by_random_forest()
    
#EOF
