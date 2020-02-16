import pandas as pd
import numpy as np
import statistics as stat
train = pd.read_csv("S:\\Hackathon\\train.csv", na_values=' ')
test = pd.read_csv("S:\\Hackathon\\test.csv", na_values=' ')

#remove the unnamed columns
train.shape
train.isnull().sum()
train = train.drop(['Unnamed: 19', 'Unnamed: 20'],axis=1)

#Dropping the unused variables
test= test.drop(["Video_id",'trending_date','description','comment_disabled','like dislike disabled','tag appered in title'],axis=1)
train= train.drop(["Video_id",'trending_date','description','comment_disabled','like dislike disabled','tag appered in title'],axis=1)

train_views = train['views']#safing for future use

df=pd.concat([train,test],axis=0,ignore_index=True)
df = df.drop(['publish_date'],axis=1)

#missing value treatment
df['Trend_day_count'] =  df['Trend_day_count'].fillna(stat.mode())
df['Tag_count'] =  df['Tag_count'].fillna(stat.mode())
df['Trend_tag_count'] =  df['Trend_tag_count'].fillna(stat.mode())
df['likes'] =  df['likes'].fillna(median())
df['subscriber']=df['subscriber'].fillna(median())
df['comment_count'] =  df['comment_count'].fillna(0)
df['category_id'] =  df['category_id'].fillna(stat.mode())


#outlier treatments
df = df[df.Trend_tag_count!='>']
df = df[df.Trend_day_count!=4444]
df = df[df.Tag_count!=3222]
df = df[df.Tag_count!=3225]
df = df[df.Trend_tag_count!='9903']
df = df[df.category_id!='2225']
df = df[df.views!='#VALUE!']
df["category_id"] = df['category_id'].replace("“24","24")


#Separating the numeric features
df_numeric = df[['subscriber','Trend_day_count','Tag_count','Trend_tag_count','comment_count','likes','dislike']]
df_numeric["Trend_tag_count"] = df["Trend_tag_count"].astype('int64')
import numpy as np
df_numeric1 = np.log1p(df_numeric)#log transforming the independent variables to remove skewness


#transforming the response variable
y = df['views']
from scipy import stats
from scipy.stats import boxcox
box_y = boxcox(box_y, lmbda=0.0)#to change as normal distribution
y = y.dropna()
box_y = y.copy()


###### NlP part#####
#preprocessing text data
#removed description as it contains unwanted info
n_data = df[["channel_title","title","tags"]]
n_data = n_data.dropna()
n_data["tags"]=n_data["tags"].str.split('|').str.join(' ')
n_data["np"]=n_data["channel_title"]+' '+n_data["title"]+' '+n_data["tags"]
nlp_data=n_data[["np"]]

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import json
import nltk
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import unicodedata,re
from nltk.corpus import stopwords
import glob


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/words')
except LookupError:
     nltk.download('words')   
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')




from nltk.tokenize import word_tokenize
def clean_text(text):
    st = text.replace(". ",".").replace("/ ","/").replace(" /","/").replace(" .",".").lower()
    return st
def tokenize_text(text):
    return word_tokenize(text)
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems
def _remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            word=word.replace('xa0','')
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

def _remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def _specific_stopwords(words):
    """Remove stop words from list of tokenized words"""
    sw=spec_sw
    #print(words)
    new_words = [w for w in words if w.lower() not in sw]
    #print (new_words)
    return new_words

def _remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in nltk.corpus.stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas
def join_text(words):
    return ' '.join(words)

clean_data=nlp_data["np"].astype('str').apply(clean_text).apply(tokenize_text).apply(_remove_non_ascii).apply(_remove_punctuation).apply(_remove_stopwords).apply(lemmatize_verbs).apply(join_text)

vec=TfidfVectorizer(ngram_range=(1, 1),max_df = 0.8 , min_df = 1)

X=vec.fit_transform(list(clean_data.values))

X=X.toarray()


#stacking the nlp part and numeric features
import numpy as np
X_tr = np.hstack((X,df_numeric))


#Done PCA to reduce the dimensions formed by wordcloud
#principal components analysis 
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
fit = pca.fit(X_tr)
fit = fit.transform(X_tr)
fit.explained_variance_ratio_
pc_comp = pd.DataFrame(fit.components_d)
pc_comp = pc_comp[0,1,2,3,4]

#train test split
trainpc = pc_comp.head(3172)
testpc = pc_comp.tail(1335)


#model building with random forest

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.1, criterion='mse',
                      max_depth=20, max_features='auto', max_leaf_nodes=None,
                      max_samples=10, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=2,
                      min_samples_split=2, min_weight_fraction_leaf=0.1,
                      n_estimators=2400, n_jobs=None, oob_score=True,
                      random_state=42, verbose=0, warm_start=False)

rf_model.fit(trainpc,y)
predictrf = rf_model.predict(testpc)


#Stacked Regressor model fitting

from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import GridSearchCV


rf = RandomForestRegressor()
en = ElasticNet()
gbr = GradientBoostingRegressor()
etr = ExtraTreesRegressor()
ada = AdaBoostRegressor()

stack = StackingCVRegressor(regressors=(en,gbr,etr,ada),meta_regressor=rf,random_state=4)
grid = GridSearchCV(estimator = stack,
                    param_grid = {
                        'meta_regressor__n_estimators':[10,100]
                    },
                    cv=10,
                    refit=True
                   )
                   
grid.fit(trainpc,y)
grid_predict = grid.predict(testpc)
grid_test = pd.DataFrame(grid_test)


#model fitting with kerasRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import keras.backend as kb
import keras
import tensorflow as tf

modelk = keras.Sequential([
    keras.layers.Dense(50, activation=tf.nn.relu, input_shape=[5]),
    keras.layers.Dense(40, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

optimizer = tf.keras.optimizers.RMSprop(0.0099)
modelk.compile(loss='mean_squared_error',optimizer=optimizer)
modelk.fit(trainpc,y,epochs=1000)

predkeras = modelk.predict(testpc)
predkeras = pd.DataFrame(predkeras)


predkeras.to_csv("S:\\hack\\output1.csv")
grid_test.to_csv("S:\\hack\\output2.csv")
predrf.to_csv("S:\\hack\\output3.csv")


#Comparing the ouputs in excel and taken average in excel




















