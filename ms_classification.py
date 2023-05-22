import pandas as pd
import numpy as np
import re
import warnings
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import os

print("hello")

def trim(df):
    # SAS - strip removes trailing characters at start and end, and whitespaces / newlines
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    # Lower case
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    print("All column names have been striped, lowered case, replaced space with underscore if any")
    print("Dropped duplicated instances if any")
    print("Categorical instances have been striped")
    return df

def shape(df, df_name):
    print(f'STATUS: Dimension of "{df_name}" = {df.shape}')


pd.set_option('display.max_colwidth', 255)
df = pd.read_csv('mtsamples.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df = trim(df)

df.head(3)

#df = df[df['medical_specialty'].isin(['Neurosurgery','ENT - Otolaryngology','Discharge Summary','Neurology'])]
df = df[df['medical_specialty'].isin(['ENT - Otolaryngology','Neurology'])]
#df = df[df['medical_specialty'].isin(['Neurosurgery','Neurology'])]

shape(df,'df')

# Task: Try and predict Medical speciality from keywords

# First check that all data has keywords

for medical_specialty in df['medical_specialty'].unique():
    print(medical_specialty)
    dum=1

#NOTE,: Thesetranscribed medical transcription sample reports and examples are provided by various users andare for reference purpose only. MTHelpLine does not certify accuracy and quality of sample reports.These transcribed medical transcription sample reports may include some uncommon or unusual formats;this would be due to the preference of the dictating physician. All names and dates have beenchanged (or removed) to keep confidentiality. Any resemblance of any type of name or date orplace or anything else to real world is purely incidental.,

# make sure indexes pair with number of rows
df = df.reset_index()
preClean_L=len(df)

subString="MTHelpLine does not certify accuracy"
fullString="NOTE,: Thesetranscribed medical transcription sample reports and examples are provided by various users andare for reference purpose only. MTHelpLine does not certify accuracy and quality of sample reports.These transcribed medical transcription sample reports may include some uncommon or unusual formats;this would be due to the preference of the dictating physician. All names and dates have beenchanged (or removed) to keep confidentiality. Any resemblance of any type of name or date orplace or anything else to real world is purely incidental.,"
dropNfull=len(fullString)
for row in df.itertuples():
    # Load current keyword and convert to strong
    current_kw=str(row.keywords)
    index=row.Index
    #current_kw=str(row['keywords'])

    if current_kw=='nan':
        #df=df.drop(index,axis=0)
        df = df.drop(index)
        #print("Dropping row "+index+"with content: "+current_kw)
        print("Found a NaN. Dropping row "+str(index))
    elif len(current_kw) < 10:
        df = df.drop(index)
        print("Found short keywords. Dropping row " + str(index))
    elif subString in current_kw:
        print("Found a keywords with Note at index "+str(index)+". Removed the final N characters from keywords.")
        df.at[index,'keywords']=current_kw[:-dropNfull]

print("Dataframe length reduced from "+str(preClean_L)+" to "+str(len(df)))

# Check that the Notes have actually been dropped.
for row in df.itertuples():
    current_kw = str(row.keywords)
    index = row.Index
    if subString in current_kw:
        print("Found a keywords with Note at index "+str(index)+". Should have been removed.")


dum=1
# Apply bag of words algorithm from sk-learn
count_vect=CountVectorizer()
X=count_vect.fit_transform(df.iloc[:]['keywords'])
feats=count_vect.get_feature_names_out()
feat_count=np.sum(X,0) # Of shape 1,2075
feat_count_sorted=np.sort(feat_count)
feats_sorted=feats[feat_count.argsort()]

print("Still have "+str(np.count_nonzero(df['medical_specialty']=='Neurology'))+" neurology entries")

#########################################################################################
# Set keywords as input data, and medical_specialty as target data
# Split the training data, and use NB or SVM classifiers
y=df.iloc[:]['medical_specialty']
dum=1;
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

X_array=X.toarray()
X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.5, random_state=101)
gnb = GaussianNB()
mnb = MultinomialNB()
#svc = SVC()
sgd = SGDClassifier()
log_loss=SGDClassifier(loss='log_loss')

gnb.fit(X_train, y_train)
mnb.fit(X_train, y_train)
sgd.fit(X_train, y_train)
#svc.fit(X_train,y_train)
log_loss.fit(X_train,y_train)


# Making Predictions
gnb_pred = gnb.predict(X_test)
mnb_pred = mnb.predict(X_test)
sgd_pred = sgd.predict(X_test)
#svc_pred = svc.predict(X_test)
ll_pred = log_loss.predict(X_test)


gnb_score = np.count_nonzero(y_test==gnb_pred)/len(y_test)
mnb_score = np.count_nonzero(y_test==mnb_pred)/len(y_test)
sgd_score = np.count_nonzero(y_test==sgd_pred)/len(y_test)
#svc_score = np.count_nonzero(y_test==svc_pred)/len(y_test)
lL_score = np.count_nonzero(y_test==ll_pred)/len(y_test)

print("Gaussian NB has "+str(gnb_score*100)+"%")
print("Multinomial NB has "+str(mnb_score*100)+"%")
print("svm with sgd has "+str(sgd_score*100)+"%")
print("log_loss with SGD has "+str(lL_score*100)+"%")

dum=1

