import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel('gg.xlsx')
d = {'Y': 1, 'N': 0}
e = {'adult': 1, 'young adult': 2, 'Elderly': 3,'teenager': 0}
f = {'White': 0, 'Black': 1, 'Mixed': 2, 'Any other': 3, 'Asian': 4 }
g = {'M': 1, 'F': 0}
h = {'Retired': 0, 'Other (please specify)': 1, 'Unemployed': 2, 'Wage earner, part-time': 3, 'Wage earner, full-time': 4, 'Homemaker' : 5, 'Student, part-time': 6, 'Student, full-time': 7}
i = {'C7': 0, 'C8': 1, 'C9': 2, 'C10': 3, 'C11': 4, 'C12': 5, 'C13': 6, 'C14': 7, 'C15': 8, 'C17': 9, 'C18': 10, 'C23': 11, 'C24': 12}

df['Commissioner'] = df['Commissioner'].map(i)
df['AGE CLASS'] = df['AGE CLASS'].map(e)
df['Gender'] = df['Gender'].map(g)
df['Ethnicity'] = df['Ethnicity'].map(f)
df['Employment Status'] = df['Employment Status'].map(h)
df['BMI'] = df['BMI'].astype('int64')
df['Cancer'] = df['Cancer'].map(d)
df['Ischaemic Heart Disease'] = df['Ischaemic Heart Disease'].map(d)
df['Cardiomyopathy'] = df['Cardiomyopathy'].map(d)
df['Heart Failure'] = df['Heart Failure'].map(d)
df['Peripheral Vascular Disease'] = df['Peripheral Vascular Disease'].map(d)
df['Myocardial Infarction'] = df['Myocardial Infarction'].map(d)
df['Thyroid'] = df['Thyroid'].map(d)
df['Type 1 Diabetes'] = df['Type 1 Diabetes'].map(d)
df['Type 2 Diabetes'] = df['Type 2 Diabetes'].map(d)
df['Polycystic Ovary Syndrome'] = df['Polycystic Ovary Syndrome'].map(d)
df['Fatty Liver'] = df['Fatty Liver'].map(d)
df['Osteoarthritis'] = df['Osteoarthritis'].map(d)
df['Other Arthritis'] = df['Other Arthritis'].map(d)
df['Asthma / COPD'] = df['Asthma / COPD'].map(d)
df['Obstructive Sleep Apnoea'] = df['Obstructive Sleep Apnoea'].map(d)
df['Anorexia Nervosa'] = df['Anorexia Nervosa'].map(d)
df['Binge Eating Disorder'] = df['Binge Eating Disorder'].map(d)
df['Bulimia Nervosa'] = df['Bulimia Nervosa'].map(d)
df['Anxiety'] = df['Anxiety'].map(d)
df['Depression'] = df['Depression'].map(d)
df['Drug Abuse'] = df['Drug Abuse'].map(d)
df['Alcoholism'] = df['Alcoholism'].map(d)
df['Transient Ischaemic Attack'] = df['Transient Ischaemic Attack'].map(d)

training_data = df.sample(frac=0.6, random_state=25)
testing_data = df.drop(training_data.index)
df.info()
x_train = training_data.iloc[:,[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
y_train = training_data ['STATUS']
x_test = testing_data.iloc[:,[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
y_test = testing_data ['STATUS']
clf = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
clf = clf.fit(x_train,y_train)

#this is where the predict function is
clf.predict ([[0,1,70,0,0,42,0,1,0,1,0,1,0,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0]])
