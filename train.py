
import pandas as pd
import re
import pickle
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# read raw data file
df = pd.read_csv('path/to/sms_all_data.csv', encoding = "latin-1")
df.drop(columns = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
df.rename(columns={'v1':'target', 'v2':'sms'}, inplace=True)

# put aside data for streaming prediction
train_df, predict_df = train_test_split(df, test_size=0.15, random_state=0)
predict_df.to_csv('path/to/sms_unseen.csv')

# text preprocessing
train_df['sms_processed'] = train_df['sms'].map(lambda x: x.lower())
train_df['sms_processed'] = train_df['sms_processed'].map(lambda x : re.sub('[^a-z]', ' ', x))
nltk.download('stopwords')
sw_list = stopwords.words('english')
stemmer = SnowballStemmer('english')
train_df['sms_processed'] = train_df['sms_processed'].map(lambda x : ' '.join([stemmer.stem(w) for w in x.split() if w not in sw_list]))

# word embedding
cv = CountVectorizer(max_features=3000, ngram_range=[1, 3])
features_tf = cv.fit_transform(train_df['sms_processed'])
pickle.dump(cv, open('path/to/embedding_model.pickle', 'wb'))

# train, evaluate and save classification model
df_tf = pd.DataFrame(features_tf.toarray(), columns=cv.get_feature_names())
train_df = pd.get_dummies(train_df, columns=['target'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(df_tf, train_df['target_spam'], test_size=0.10, random_state=0)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("The accuracy score is : \n",accuracy_score(y_test, y_pred), "\n")
print("The precision is : \n",precision_score(y_test,y_pred), "\n")
pickle.dump(clf, open('path/to/classification_model.pickle', 'wb'))
