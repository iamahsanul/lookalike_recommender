from preprocessing import *
from regular_classifier import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
#import warnings
#warnings.filterwarnings("ignore")
#%matplotlib inline
#%pylab inline

#Some display parameters to see the data structure
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)

#Reading data
DATAPATH = 'C:\MAH-ALL\Dropbox\MAH\Python project\Wunderman_LookAlike\Data\data_mah_copy.parquet'
df = pd.read_parquet(DATAPATH)
df.info()

# Sampling the first 5000 rows
#df = df.iloc[:5000]
df.head()

#Preparing features and labels for machine learning
#df["seed"] = df["seed"].astype(int)
#df_features = df.groupby(['user_id'])['page_urlpath'].apply(lambda x: ', '.join(x.astype(str))).reset_index()
#df_labels = df.groupby(['user_id']).agg({'seed':np.mean})

#Preparing features and labels for machine learning
df["seed"] = df["seed"].astype(int)
df_features = df.groupby(['user_id'])['page_urlpath'].apply(lambda x: ', '.join(x.astype(str))).reset_index()
#df_cleaned = clean_sentences(df_features)
df_features = df_features[['page_urlpath']]
df_features.rename(columns={'page_urlpath':'sentence'}, inplace=True)
df_labels = df.groupby(['user_id']).agg({'seed':np.mean})
df_labels = df_labels[['seed']]

train_features, test_features, train_labels, test_labels = train_test_split(df_features, df_labels, test_size=0.33, random_state=42)

test_probability = []
predict_probability = []

test_probability, predict_probability = test_regular_classifier(train_features, test_features, train_labels, test_labels)

for sentence, prob in zip(test_features['sentence'], test_labels['seed']):
    test_probability.append(prob)
    predict_index = test_regular_classifier(train_features, sentence)
    predict_probability.append(((train_labels[['seed']].iloc[predict_index]).values[0])[0])


model_DT = DecisionTreeClassifier()
model_DT.fit(train_features, train_labels)
predictions_DT = model_DT.predict(test_features)

print('Decision tree accuracy:', accuracy_score(predictions_DT, test_labels))
print('')
print('Confusion matrix:')
print(confusion_matrix(test_labels,predictions_DT))

# Renaming the description column
df_features.rename(columns={'page_urlpath':'sentence'}, inplace=True)
df_cleaned = clean_sentences(df_features)




#df.duplicated().sum()
#df.describe()

# write
#df.to_parquet('my_newfile.parquet')

plt.plot([i for i in range(10)])
plt.show()

print(df.head(10))
df.head(10)
plt.plot(df.head(10))
plt.show()

print(df.head(10))

print('adsf')