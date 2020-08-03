import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
pd.set_option("display.precision", 2)
pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:.2f}'.format
from tqdm import tqdm_notebook,tqdm
import reverse_geocoder as revgc
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.cat_boost import  CatBoostEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from math import cos, asin, sqrt
from nltk import word_tokenize  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
import re
from sklearn.decomposition import PCA
from datetime import datetime
porter_stemmer = PorterStemmer()
import eli5
from eli5.sklearn import PermutationImportance
import pickle


def tokenize(string):
    words=re.sub(r"[^A-Za-z0-9]", " ", string).lower().split()
    return words

class LemmaTokenizer:
    def __init__(self):
            self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in tokenize(doc) ]

porter_stemmer = PorterStemmer()
def stemming_tokenizer(str_input):
    words = tokenize(str_input)
    words = [porter_stemmer.stem(word) for word in words]
    return words

def count_encoder(df_train,df_test,column,label=False):
    column_dict= pd.concat([df_train,df_test])[column].value_counts().to_dict()
    x=df_train[column].map(column_dict)
    y=df_test[column].map(column_dict)
    return x,y

def onehot_encoder(df_train,df_test,column,label=False):
    enc = OneHotEncoder(handle_unknown='ignore')    
    dfx=pd.concat([df_train,df_test])[column]
    enc.fit(dfx.values.reshape(-1, 1))
    transformedx=enc.transform(df_train[column].values.reshape(-1, 1))
    transformedy=enc.transform(df_test[column].values.reshape(-1, 1))
    x=pd.DataFrame(transformedx.todense(),columns=enc.categories_)
    y=pd.DataFrame(transformedy.todense(),columns=enc.categories_)
    return x,y

def label_encoder(df_train,df_test,column,label=False):
    enc = LabelEncoder()
    dfx=pd.concat([df_train,df_test])[column]
    enc.fit(dfx)
    x=enc.transform(df_train[column])
    y=enc.transform(df_test[column])
    return x,y

def target_encoding(encoder,df_train,df_test,column,label):
    enc=encoder(cols=[column])
    xDF = pd.DataFrame()
    yDF = pd.DataFrame()
    for value in label.unique():
        act_label=label.apply(lambda x: 1 if x==value else 0 ) 
        x=enc.fit_transform(df_train,act_label)[column]
        y=enc.transform(df_test)[column]
        xDF[value]=x
        yDF[value]=y
    return xDF,yDF

def LOO_target_encoding(*args):
    return target_encoding(LeaveOneOutEncoder, *args)

def CB_target_encoding(*args):
    return target_encoding(CatBoostEncoder, *args)


def validate_features(df,training_mask,target, clf = RandomForestClassifier(max_depth=5, random_state=0,n_estimators=100,n_jobs=-1),encoder_dict={},nsplits=5):       
    X=df[training_mask==1]
    X_test=df[training_mask==0]
    y=target[training_mask==1]
    skf = StratifiedKFold(n_splits=nsplits, random_state=42, shuffle=True)
    scores=list()
    yoof=np.zeros((X.shape[0],3))
    for train, val in skf.split(X, y):
        X_train=X.iloc[train]
        y_train=y.iloc[train]
        X_val=X.iloc[val]
        y_val=y.iloc[val]
        ##
        for column, encoder in encoder_dict.items():
            colname=column+str(encoder.__name__)
            #print(column,str(encoder.__name__))
            enc_train, enc_test=encoder(X_train,X_val,column,y_train)
            #display(enc_train)
            if len(enc_train.shape)>1:
                for col in enc_train:
                    #print(col)
                    new_colname= colname+str(col)
                    X_train[new_colname]=enc_train[col].values
                    X_val[new_colname]=enc_test[col].values
            else:
                if type(enc_train) is not np.ndarray:
                    enc_train=enc_train.values
                    enc_test=enc_test.values
                X_train[colname]=enc_train
                X_val[colname]=enc_test      
        #return X_train

        clf.fit(X_train.drop(columns=encoder_dict.keys()), y_train)
        
        preds=clf.predict_proba(X_val.drop(columns=encoder_dict.keys()))
        scores.append(log_loss(y_val, preds))
        yoof[val]=preds
    return log_loss(y,yoof)


def n_features (df,n,list_column='features'):   
    x = pd.Series([x for item in df.features for x in item]).value_counts()
    x.head(n)
    x=x.index.values[:n]
    for column in x:
        df[column] = df[list_column].apply(lambda row: 1 if column in row else 0) 
    return x


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))

def closest(data, v):
    p=min(data, key=lambda p: distance(v['latitude'],v['longitude'],p['lat'],p['lon']))
    return distance(v['latitude'],v['longitude'],p['lat'],p['lon'])


def agg_function(df,kat_column, num_column, agg,diff=False):
    df2=df.groupby(kat_column).agg({num_column:agg}, as_index=False).fillna(0)
    df2.columns = [agg+'_'+col + "_by_"+kat_column  for col in df2.columns.values]
    name=df2.columns.values[0]
    if diff==False:
        df2=df.join(df2,on=[kat_column]).drop(kat_column,axis=1)
        df2.drop(df2.columns.difference([name]), 1, inplace=True)
    else:
        df2=df.join(df2,on=[kat_column],rsuffix='del')
        colname=num_column +'-'+name
        df2[colname]= df2[num_column]-df2[name]
        df2.drop(df2.columns.difference([colname]), 1, inplace=True)
    return df2

def agg_diff_fun(*args):
    return agg_function(diff=True, *args)

def compare(best_value,df,df_new,columns,columns_new,training,label,encoder_dict={}, clf= RandomForestClassifier(max_depth=5, random_state=0,n_estimators=100,n_jobs=-1)):
    new_value=validate_features(df_new[columns_new],df_new[training],df_new[label],encoder_dict=encoder_dict,clf=clf)
    
    if new_value<best_value:
        value_out=new_value
        columns_out=columns_new
        df_out=df_new
    else :
        value_out=best_value
        columns_out=columns
        df_out=df
    return (value_out,df_out,columns_out)

def interaction_fun(a,b,operator):
    if operator == '+':
        return a+b
    elif operator == '-':
        return a-b
    elif operator == '*':
        return a*b
    elif operator =='/':
        return a/(b+1e-16)
    else:
        raise ValueError('unsupported operator '+ operator)        

def create_dp_table(columns,best_value,encoders,timestart,index):
    d = {'columns': [columns], 'best value': [best_value],'encoders':[encoders], 'time':datetime.now()-timestart}
    df=pd.DataFrame(data=d, index=[index])
    return df

def feature_importance(df,training_mask,target,model = RandomForestClassifier(max_depth=5, random_state=0,n_estimators=100,n_jobs=-1)):
    nsplits=3
    X=df[training_mask==1]
    X_test=df[training_mask==0]
    y=target[training_mask==1]
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train, val in skf.split(X, y):
        X_train=X.iloc[train]
        y_train=y.iloc[train]
        X_val=X.iloc[val]
        y_val=y.iloc[val]
    my_model=model.fit(X_train, y_train)
    perm = PermutationImportance(my_model, random_state=1).fit(X_val, y_val)
    explanation = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names = X_val.columns.tolist())
    return explanation

def feature_engineering(df,training,label,num_features,agg_dict,pca_list,clf):
    print(datetime.now())
    blacklist=['training','interest_integer']
    numcols=[x for x in list(df.select_dtypes(exclude=['object']).columns) if x not in blacklist]
    blacklist_kat=['features','cc','features_txt','description','interest_level']
    katcols=[x for x in list(df.select_dtypes(object).columns) if x not in blacklist_kat]
    columns= numcols
    best_value=validate_features(df[columns],df[training],df[label],clf=clf)
    timestart=datetime.now()
    good_encoders={}
    ##### numeric features selection
#     while True:
#         drop_col=None
#         for column in tqdm(columns):
#             act_columns= [x for x in columns if x !=column]
#             new_value=validate_features(df[act_columns],df[training],df[label],clf=clf)
#             if new_value<best_value:
#                 best_value=new_value
#                 drop_col=column
#         if drop_col is None:
#             break
#         else:
#             columns=[x for x in columns if x !=drop_col]
    bestval=[999]
    startcolumns=columns
    for i in range(len(columns)-1):
        best_value=999
        for column in tqdm(startcolumns):
            act_columns= [x for x in startcolumns if x !=column]
            new_value=validate_features(df[act_columns],df[training],df[label],clf=clf)
            if new_value<best_value:
                best_value=new_value
                drop_col=column
        startcolumns=[x for x in startcolumns if x !=drop_col]
        if best_value<bestval:
            bestval=best_value
            columns=startcolumns
    best_value=bestval            
            
    print('after numeric feature selection: ',columns,best_value)
    dp_table=create_dp_table(columns,best_value,good_encoders,timestart,'after numeric feature selection')
    
    
    ##### TOP N feature
    

    y=n_features(df,num_features)
    for item in tqdm(y):
        act_columns=columns+[item]
        new_value=validate_features(df[act_columns],df[training],df[label],clf=clf)
        if  new_value<best_value:
            best_value=new_value
            columns=act_columns
    print('after TOP N features: ',columns,best_value)
    dp_table=pd.concat([dp_table,create_dp_table(columns,best_value,good_encoders,timestart,'after TOP N features')])
    
    ##### numeric features aggregates by non-numeric features

    for kat_column in tqdm(katcols):
        for num_column in numcols:  
            #print(num_column,kat_column,best_value,columns)
            new_features=[]
            for fun,items in agg_dict.items():
                for agg in items:
                    act_new_features=fun(df,kat_column, num_column, agg)
                    new_features.append(act_new_features)
            new_features=pd.concat(new_features,axis=1)    
            df_val=pd.concat([df,new_features],axis=1)  
            act_columns=columns+list(new_features.columns)
            best_value,df,columns=compare(best_value,df,df_val,columns,act_columns,training,label,clf=clf)
    print('after numeric features aggregated by categorical features',columns,best_value)
    dp_table=pd.concat([dp_table,create_dp_table(columns,best_value,good_encoders,timestart,'after numeric features aggregated by categorical features')])

    ##### numeric features interactions
    operators=['+','-','*','/']
    blacklist_interactions=['hr','dayofweek','mnth']
    numcols=[x for x in numcols if x not in blacklist_interactions]
    number=10
    for operator in tqdm(operators):
        dfx=df
        columns_inter=[]
        n=0
        for i,num_col in enumerate(numcols):
            for j,inter_num_col in enumerate(numcols):
                if num_col ==inter_num_col and operators!='*':
                    continue
                elif j>=i:
                    col_name=num_col+operator+inter_num_col
                    if col_name not in dfx.columns:
                        dfx[col_name]=interaction_fun(dfx[num_col],dfx[inter_num_col],operator)
                        columns_inter=columns_inter+[col_name]
        important_list=feature_importance(dfx[columns_inter],dfx[training],dfx[label],model=clf).sort_values(by=['weight'],ascending=False)['feature']
        for idx in tqdm(range(0,len(important_list),number)):    
            act_columns=columns+important_list[idx:idx+number].tolist()
            bestvalue_old=best_value
            best_value,df,columns=compare(best_value,df,dfx,columns,act_columns,training,label,clf=clf)
            if best_value>=bestvalue_old:
                    break                 
    print('numeric features interactions',columns,best_value)
    dp_table=pd.concat([dp_table,create_dp_table(columns,best_value,good_encoders,timestart,'after numeric features interactions')])
    ##### non-numeric features encoding

    df_best=df[columns]
    best_value=validate_features(df_best,df[training],df[label],clf=clf)
    good_encoders={}
    for column in tqdm(katcols):
        encoder_dict={}
        if df[column].nunique()<15:
            encoders= [onehot_encoder,count_encoder,label_encoder,CB_target_encoding,LOO_target_encoding]
        elif df[column].nunique()>15000: 
            encoders=[]
        else:
            encoders= [count_encoder,label_encoder,CB_target_encoding,LOO_target_encoding]
        for encoder in encoders:

            encoder_dict={**good_encoders,column:encoder}
            act_columns=columns+list(encoder_dict.keys())
            act_value=validate_features(df[act_columns],df[training],df[label],encoder_dict=encoder_dict,clf=clf)
            if act_value<best_value:
                print(column,encoder.__name__,act_value)
                best_value=act_value
                good_encoders[column]=encoder
    columns=columns+list(good_encoders.keys())     
    print('after non-numeric features encoding: ',columns,best_value)
    dp_table=pd.concat([dp_table,create_dp_table(columns,best_value,good_encoders,timestart,'after categorical feature encoding')]) 

    ##### text features tokenization plus PCA

    tokenizer=[stemming_tokenizer,LemmaTokenizer(),None]
    besttokenizervalue=999
    nrange=4
    for n in tqdm(range(1, nrange)):    
        for tok in tokenizer:
            vectorizer = TfidfVectorizer(analyzer='word',stop_words='english',tokenizer=tok,lowercase=True,max_df=0.8,min_df=0.025,ngram_range=(1, n))
            dfx = vectorizer.fit_transform(df['description'])
            dfx=pd.DataFrame(dfx.todense(),columns=vectorizer.get_feature_names())
            for i in pca_list:
                pca = PCA(n_components=i)
                P=pca.fit_transform(dfx)
                pca=pd.DataFrame(P)
                pca.columns=[str(column) + '_PCA' for column in pca.columns]
                df_pca=df.join(pca)
                act_columns=columns+list(pca.columns)
                act_value=validate_features(df_pca[act_columns],df_pca[training],df_pca[label],encoder_dict=good_encoders,clf=clf)
                print(n,tok,i)
                if act_value<besttokenizervalue:
                    besttokenizervalue=act_value
                    df_best=pca
                    df_best_cols=df_best.columns.values
                    besttoken=tok
                    best_nrange=n
                    best_nPCA=i
                    print(besttokenizervalue)
    if besttokenizervalue<best_value:
        best_value=besttokenizervalue
        df=df.join(df_best)
        columns.extend(df_best_cols)
        best_token=besttoken
        df_tokenizer=pd.DataFrame(data={'tokenizer': [besttoken], 'ngram_range': [best_nrange],'pca_components':[best_nPCA]},index=['Best tokenizer'])
    else:
        df_tokenizer=[]
    print('after text feature tokenization plus PCA: ',columns,best_value)
    dp_table=pd.concat([dp_table,create_dp_table(columns,best_value,good_encoders,timestart,'after text feature tokenization plus PCA')]) 
    print(datetime.now())
    return df,columns,good_encoders,dp_table,df_tokenizer

def predict_enc(encoder_dict,X_train,X_val,y_train):
    for column, encoder in encoder_dict.items():
                colname=column+str(encoder.__name__)
                enc_train, enc_test=encoder(X_train,X_val,column,y_train)
                if len(enc_train.shape)>1:
                    for col in enc_train:
                        #print(col)
                        new_colname= colname+str(col)
                        X_train[new_colname]=enc_train[col].values
                        X_val[new_colname]=enc_test[col].values
                else:
                    if type(enc_train) is not np.ndarray:
                        enc_train=enc_train.values
                        enc_test=enc_test.values
                    X_train[colname]=enc_train
                    X_val[colname]=enc_test 
    return X_train,X_val
    
def model_predict(df,training_mask,target, clf = RandomForestClassifier(max_depth=5, random_state=0,n_estimators=100,n_jobs=-1),encoder_dict={},n_splits=5):   
    X=df[training_mask==1]
    X_test=df[training_mask==0]
    y=target[training_mask==1]
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    scores=list()
    yoof=np.zeros((X.shape[0],3))
    test_pred=np.zeros((X_test.shape[0],3))

            
    for train, val in skf.split(X, y):
        X_train=X.iloc[train]
        y_train=y.iloc[train]
        X_val=X.iloc[val]
        y_val=y.iloc[val]
        
        ##
        predict_enc(encoder_dict,X_train,X_val,y_train)
        clf.fit(X_train.drop(columns=encoder_dict.keys()), y_train)
        preds=clf.predict_proba(X_val.drop(columns=encoder_dict.keys()))
        scores.append(log_loss(y_val, preds))
        yoof[val]=preds
    X,X_test=predict_enc(encoder_dict,X,X_test,y)
    clf.fit(X.drop(columns=encoder_dict.keys()), y)
    test_pred=clf.predict_proba(X_test.drop(columns=encoder_dict.keys()))
    return test_pred,yoof,log_loss(y,yoof)

def save_csv (name,test_pred,yoof,model,df_test_listingID,df_train_listingID,dir='Thesis_predictions/'):
    label_dictionary ={1 : 'low',  2:'medium', 3:'high' } 
    columns=[label_dictionary[key] for key in model.classes_]
    columns_out=['listing_id', 'high', 'medium', 'low']
    predictions=pd.DataFrame(test_pred,columns=columns)
    predictions['listing_id']=df_test_listingID.reset_index(drop=True)
    predictions=predictions[columns_out]
    yoof_pred=pd.DataFrame(yoof,columns=columns)
    yoof_pred['listing_id']=df_train_listingID.reset_index(drop=True)
    yoof_pred=yoof_pred[columns_out]
    predictions.to_csv(dir+name+'.csv', index=False)  
    yoof_pred.to_csv(dir+name+'_yoof.csv', index=False)

def save_object(objects,name):
    pickle_out = open(name+".pickle","wb")
    pickle.dump(objects, pickle_out)
    pickle_out.close()

def load_object(name):
    pickle_in = open(name+".pickle","rb")
    output = pickle.load(pickle_in)
    return output    

def get_best_params(params_df,sort_by):
    params_df=params_df.sort_values(sort_by)
    best_grid_params=params_df.iloc[0,1:-2]
    return best_grid_params.to_dict()