#!/usr/bin/env python
# coding: utf-8

# In[203]:


import sagemaker, boto3
import pandas as pd 
from io import StringIO
import json
import requests
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from pygam import LinearGAM, GAM, s, f
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import re
from langdetect import detect, LangDetectException
get_ipython().run_line_magic('matplotlib', 'inline')
import emoji
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


# In[268]:


def detect_lang(s):
    try:
        return detect(s)
    except:
        return "Could not detect language"
    
def scrub_text(in_text):
    #Remove urls, numbers and punctuation 
    word_list = re.sub(r'(http)\S*|[^a-zA-Z 0-9\n]','', in_text.lower()).split()
    return ' '.join(word_list)

def top_bigrams(df, col, num_bigrams=10, stop_words=[]):
    cv = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words)
    bigram_counts = cv.fit_transform(df[col])
    vocab = list(cv.vocabulary_.keys())
    
    bigram_dict = {bigram: bigram_counts.getcol(i).sum() for i, bigram in enumerate(vocab)}
    return  sorted(bigram_dict.items(), key=lambda x: x[1], reverse=True)[:num_bigrams]

def hashtag_occurences(df, col):
    commentaries = " ".join(df[col])
    hashtags = re.findall(r"(?<=#)\w+|(?<=#\|)\w+|(?<=＃\|)\w+", commentaries)
    return Counter(hashtags)

def get_score_hashtag(df, value_arr, text_col, score_col): #this function will give us the avg number of likes for each hashtag
    hashtag_scores = {}
    for val in value_arr:
        mask = df[text_col].str.contains(rf"(?<=#){val}|(?<=#\|){val}|(?<=＃\|){val}", case=False, na=False)
        scores = df.loc[mask, score_col]
        if not scores.empty:
            hashtag_scores[val] = scores.mean()
        else:
            hashtag_scores[val] = 0
    return hashtag_scores

def score_text_hashtag(s, score_dict=hashtag_scores): #average of all hastag scores in a string
    hashtags = re.findall(r"(?<=#)\w+|(?<=#\|)\w+|(?<=＃\|)\w+", s)  # Extract hashtags
    scores = [score_dict[hashtag] for hashtag in hashtags if hashtag in score_dict]
    average_score = sum(scores) / len(scores) if len(scores) != 0 else 0.0  # Calculate average score
    return average_score

        


# In[267]:


dummy = "PowerWomen"
re.findall(rf"(?<=#){dummy}|(?<=#\|){dummy}|(?<=＃\|){dummy}", "Congratulations to our #nice own Julie Sweet for being recognized as one of the @[Forbes](urn:li:organization:5597) World’s 100 Most Powerful Women. https://accntu.re/2PHp3aZ {hashtag|\#|PowerWomen}")


# ## Import Data and Data Cleaning 

# In[2]:


role = sagemaker.get_execution_role()
s3 = boto3.client('s3')
csv_bucket = "proj-506-general-wh-data"
csv_file1 = "linkedin_data_scratch/all_combined_topic_embedding_data.csv"
csv_file2 = "linkedin_data_scratch/final_combined_data.csv"
response = s3.get_object(Bucket=csv_bucket, Key=csv_file1)
response2 = s3.get_object(Bucket=csv_bucket, Key=csv_file2)
body = response['Body']
body2 = response2['Body']
csv_string = body.read().decode('utf-8')
csv_string2 = body2.read().decode('utf-8')
df_combined = pd.read_csv(StringIO(csv_string))
df_raw = pd.read_csv(StringIO(csv_string2))


# In[7]:


# extract_components = lambda x: pd.Series({
#     'month': datetime.datetime.fromtimestamp(x / 1000).strftime('%B'),
#     'year': datetime.datetime.fromtimestamp(x / 1000).year,
#     'time_of_day': datetime.datetime.fromtimestamp(round(x / 3600000) * 3600).strftime('%H:%M:%S')
# })

# df_combined[['month', 'year', 'time_of_day']] = df_combined['createdAt'].apply(extract_components)
# df_combined[['month_lm', 'year_lm', 'time_of_day_lm']] = df_combined['lastModifiedAt'].apply(extract_components)

df_combined["commentary_length"] = df_combined["commentary"].str.len()
df_raw["commentary_length"] = df_raw["commentary"].str.len()
# df_combined['Day_of_Week'] = pd.to_datetime(df_combined['createdAt'], unit='ms').dt.dayofweek
#df_combined = df_combined.loc[df_combined["totalShareStatistics.likeCount"] > 100]
df_combined.reset_index(inplace=True, drop=True)


# In[4]:


feature_drop_list = [
            # dropping because there is 1 unique value in these columns
            "lifecycleState", 
            "visibility", 
            "author",
            "isReshareDisabledByAuthor",
            "distribution.thirdPartyDistributionChannels", # they all look like this : []
             # I am not sure how to these represent these 
            "contentLandingPage", 
             "id", 
            "adContext.dscName",
            "content.article.source",    
             # already covered 
             "commentary", 
             "content.media.title",
               'content.article.title', 
               'content.article.description', 
               'video_object_detection_str', 
               'video_text_detection_str', 
               'concat_image_object_detection_str',
               'concat_image_text_detection_str',
             # these are response variables, not features
            "totalShareStatistics.uniqueImpressionsCount",
            "totalShareStatistics.shareCount",
            "totalShareStatistics.engagement",
            "totalShareStatistics.clickCount",
            "totalShareStatistics.likeCount",
            "totalShareStatistics.impressionCount",
            "totalShareStatistics.commentCount",
            # "createdAt",
            # "lastModifiedAt",
            "adContext.dscAdAccount",
            "contentCallToActionLabel", 
            #dropping these as they are insignificant to the response
            # 'time_of_day',
            # 'year',
            # 'year_lm',
            "adContext.dscAdType",
            "adContext.dscStatus",
            # 'month',
            # 'time_of_day_lm',
            # 'month_lm'
            ]
one_hot_encoding_features = [#"contentCallToActionLabel", 
                             # "adContext.dscAdType",
                             # "adContext.dscAdAccount",
                             #'time_of_day',
                             #'year',
                             # 'month',
                             # 'time_of_day_lm',
                             # 'year_lm',
                             # 'month_lm'
                            # 'Day_of_Week'
                            ]
binary_features = [#"distribution.feedDistribution",
                   "lifecycleStateInfo.isEditedByAuthor",
                   # "adContext.dscStatus",
                   "adContext.isDsc",
                   "has_reshareContext.parent"
                  ]
                   
df_combined_features = df_combined.drop(columns = feature_drop_list)

df_combined_features = df_combined_features.fillna(value = {"publishedAt" : -1,
                                      "contentCallToActionLabel": "n/a",
                                      # "adContext.dscStatus" : "n/a",
                                      "adContext.dscAdType" : "n/a",
                                      "adContext.dscAdAccount": "n/a" ,
                                      "commentary_length": 0
                                     })
df_original = df_combined_features
df_combined_features["distribution.feedDistribution"] = df_combined_features["distribution.feedDistribution"]=="MAIN_FEED"
# df_combined_features["adContext.dscStatus"]= df_combined_features["adContext.dscStatus"] =="ACTIVE"
#df_combined_features["adContext.dscAdTypePresent"] = df_combined_features["adContext.dscAdType"] != "n/a"
df_combined_features[binary_features] = df_combined_features[binary_features].astype(int)
enc = OneHotEncoder(handle_unknown='ignore')
encoded_features = pd.DataFrame(enc.fit_transform(df_combined_features[one_hot_encoding_features]).toarray(), 
                                columns=enc.get_feature_names_out(one_hot_encoding_features)
                               )
df_combined_features = df_combined_features.drop(columns = one_hot_encoding_features).merge(encoded_features,
                                                           how = "left",
                                                           left_index = True,
                                                           right_index = True)
df_combined_target_like = df_combined[["totalShareStatistics.likeCount"]]
# df_combined_target_share = df_combined[["totalShareStatistics.shareCount"]]
#df_combined_is_liked = df_combined[

df_raw["object_detection"] = df_raw["video_object_detection_str"].fillna(df_raw["concat_image_object_detection_str"])
df_raw["text_detection"] = df_raw["video_text_detection_str"].fillna(df_raw["concat_image_text_detection_str"])
df_raw.drop(columns=["video_object_detection_str", "concat_image_object_detection_str", "video_text_detection_str", "concat_image_text_detection_str"], inplace=True)


# In[65]:


csv_buffer = StringIO()
df_combined_features.to_csv(csv_buffer, index=False)
csv_string = csv_buffer.getvalue()
s3.put_object(Body=csv_string, Bucket="proj-506-general-wh-data", Key="data_for_GAN/features_df.csv")

csv_buffer2 = StringIO()
df_combined[["totalShareStatistics.likeCount", "totalShareStatistics.shareCount"]].to_csv(csv_buffer2, index=False)
csv_string2 = csv_buffer2.getvalue()
s3.put_object(Body=csv_string2, Bucket="proj-506-general-wh-data", Key="data_for_GAN/targets_df.csv")


# In[28]:


df_raw["commentary_length"] = df_raw["commentary"].apply(lambda x: len(str(x).split()))
df_raw["num_objects"] = df_raw["object_detection"].apply(lambda x: len(str(x).split()))
df_raw["content.media.id"] = df_raw["content.media.id"].fillna("n/a")
df_raw["media_type"] = df_raw["content.media.id"].apply(lambda x: x.split(":")[2] if x != "n/a" else x)
df_raw.loc[(df_raw["media_type"] == "n/a") & (df_raw["adContext.dscAdType"] == "CAROUSEL"), "media_type"] = "multiimage"


# In[37]:


df_raw.drop('Unnamed: 0', axis=1, inplace=True)


# In[40]:


continuous_features = ["lastModifiedAt", "publishedAt", "createdAt", "num_objects", "commentary_length"]
response_variables = ['totalShareStatistics.uniqueImpressionsCount', 'totalShareStatistics.shareCount', 'totalShareStatistics.engagement', \
                      'totalShareStatistics.clickCount', 'totalShareStatistics.likeCount', 'totalShareStatistics.impressionCount', 'totalShareStatistics.commentCount']
discrete_features = [col for col in df_raw.columns if col not in (continuous_features + response_variables)]


# In[47]:


correlations = pd.Series({col: matthews_corrcoef(df_raw.fillna("n/a")[col], df_raw['totalShareStatistics.likeCount']) for col in df_raw.columns if col in ["visibility", "author"]})
print(correlations)


# In[46]:


df_raw.dtypes


# In[55]:


for feature in discrete_features: # loop over all discrete features 
    print(df_raw.groupby(feature).apply(lambda x: x['totalShareStatistics.likeCount'].corr(x['num_objects'], method='spearman')))


# In[67]:


df_raw = df_raw.drop(df_raw.loc[(df_raw["media_type"] == "image") & (df_raw["object_detection"].isna()) & (df_raw["text_detection"].isna())].index)
df_raw = df_raw.drop(df_raw.loc[(df_raw["media_type"] == "video") & (df_raw["object_detection"].isna()) & (df_raw["text_detection"].isna())].index)
df_raw.reset_index(drop=True, inplace=True)


# In[129]:


pd.set_option('display.max_rows', 100)
print(df_raw.nlargest(100, "totalShareStatistics.likeCount").sort_values(by="totalShareStatistics.likeCount", ascending=False)[["commentary", "totalShareStatistics.likeCount"]].to_string())


# In[101]:


df_raw["language"] = df_raw['commentary'].apply(lambda x: detect_lang(x))


# In[102]:


df_raw["language"].value_counts()


# #### posts with a first plural pronoun such as "we" get significantly more likes

# In[152]:


pattern = r"\b(we|us|our|ourselves|we\'re|we\'ve|we\'d)\b"
df_raw_en = df_raw.loc[df_raw["language"] == "en"]
df_raw["fpp_present"] = df_raw['commentary'].str.count(pattern, flags=re.IGNORECASE) > 0
print(df_raw.loc[df_raw["language"] == "en"].groupby("fpp_present")["totalShareStatistics.likeCount"].median(), "\n",\
      df_raw.loc[df_raw["language"] == "en"].groupby("fpp_present")["totalShareStatistics.likeCount"].count())


# In[153]:


from scipy.stats import pointbiserialr

corr, pvalue = pointbiserialr(df_raw_en['totalShareStatistics.likeCount'], df_raw_en['fpp_present'])
print(f"Point biserial correlation coefficient: {corr}, p-value: {pvalue}")


# ## Now let's look at emojis...
# unfortunately, not a lot of posts have emojis

# In[135]:


emoji.emoji_count("Strategy ➕ purpose = a winning combination 🤗💜\n\nPeek into our business values—and the progress we've made in creating a more inclusive and sustainable future. https://accntu.re/3YGJtnp")


# In[288]:


df_raw_en["emoji_present"] = df_raw_en["commentary"].apply(emoji.emoji_count) > 0
print(df_raw_en.groupby("emoji_present")["totalShareStatistics.likeCount"].mean(), "\n",
     df_raw_en.groupby("emoji_present")["totalShareStatistics.likeCount"].count())


# In[287]:


corr, pvalue = pointbiserialr(df_raw_en['totalShareStatistics.likeCount'], df_raw_en['emoji_present'])
print(f"Point biserial correlation coefficient: {corr}, p-value: {pvalue}")


# ## Now let's look at most popular bigrams, let's scrub the text for this one

# In[180]:


df_raw_en["scrubbed_commentary"] = df_raw_en["commentary"].apply(scrub_text)

ignore_these = ['in', 'of', 'by', 'an', 'at', 'the', 'does not', 'to', 'is', 'and']
top_bigrams(df_raw_en.nlargest(100, "totalShareStatistics.likeCount"), "scrubbed_commentary", 10, ignore_these)


# ## Number of People Tagged

# In[290]:


df_raw_en["people_tagged"] = df_raw_en['commentary'].str.count('urn:li:person:') > 0
print(df_raw_en.groupby("people_tagged")["totalShareStatistics.likeCount"].median(), "\n",
     df_raw_en.groupby("people_tagged")["totalShareStatistics.likeCount"].count())


# In[192]:


corr, pvalue = pointbiserialr(df_raw_en['totalShareStatistics.likeCount'], df_raw_en['people_tagged'])
print(f"Point biserial correlation coefficient: {corr}, p-value: {pvalue}")


# ## Tags in general

# In[199]:


df_raw_en["tagged"] = df_raw_en['commentary'].str.count('@') > 0
print(df_raw_en.groupby("tagged")["totalShareStatistics.likeCount"].median(), "\n",
     df_raw_en.groupby("tagged")["totalShareStatistics.likeCount"].count())


# In[200]:


corr, pvalue = pointbiserialr(df_raw_en['totalShareStatistics.likeCount'], df_raw_en['tagged'])
print(f"Point biserial correlation coefficient: {corr}, p-value: {pvalue}")


# In[225]:


hashtag_occ = hashtag_occurences(df_raw_en, "commentary")
print(hashtag_occ.keys())


# In[294]:


sorted_dict = dict(sorted(hashtag_occ.items(), key=lambda x: x[1], reverse=True))
sorted_dict = {hashtag: sorted_dict[hashtag] for hashtag in sorted_dict if sorted_dict[hashtag] >= 10}
print(sorted_dict)


# In[269]:


hashtag_scores = get_score_hashtag(df_raw_en, value_arr=hashtag_occ, text_col="commentary", score_col="totalShareStatistics.likeCount")


# In[291]:


len(hashtag_scores)


# In[273]:


df_raw_en["post_hashtag_score"] = df_raw_en["commentary"].apply(lambda x: score_text_hashtag(x, hashtag_scores))


# In[295]:


df_raw_en["post_hashtag_score"].corr(df_raw_en['totalShareStatistics.likeCount'])


# ## To do list:
# ### 1. ~~look up most common words for commentary~~
# ### 2. ~~Count num of people tagged~~
# ### 3. ~~find out how to detect english from commentary~~
# ### 4. ~~emoji significance~~
# ### 5. ~~hashtag significance~~
# ### 6. ~~maybe remove links~~
# ### 7. ~~check holidays~~
# ### 8. ~~set carousel to multi image~~
# ### 9. look at word ordering/alternatives to bag of words
# ### 10. time series analysis

# In[14]:


df_raw.plot(kind='scatter', x='num_objects', y='totalShareStatistics.likeCount')

# show the plot
plt.show()


# In[32]:


df_raw["totalShareStatistics.likeCount"].value_counts()


# ## Models

# ### Ridge Regression ###

# In[97]:


X_train, X_test, y_train, y_test = train_test_split(df_combined_features.to_numpy(), 
                                                    #df_combined_target_like,
                                                    df_combined_target_like.to_numpy(), 
                                                    test_size=0.2, 
                                                    random_state=0)


# In[98]:


from sklearn.preprocessing import MinMaxScaler

scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(df_combined_features), columns=df_combined_features.columns)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(df_combined_features.to_numpy(), 
                                                    #df_combined_target_like,
                                                    df_combined_target_like.to_numpy(), 
                                                    test_size=0.2, 
                                                    random_state=0)


# In[99]:


X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed = train_test_split(df_combined_features.to_numpy(), 
                                                    #df_combined_target_like,
                                                    np.log(1 + df_combined_target_like.to_numpy()), 
                                                    test_size=0.2, 
                                                    random_state=0)


# In[100]:


ridge = Ridge(alpha=1.0)
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_train,y_train)


# In[84]:


ridgeT = Ridge(alpha=1.0)
ridgeT.fit(X_train_transformed, y_train_transformed)
ridgeT_pred = ridge.predict(X_test_transformed)
ridgeT.score(X_train_transformed,y_train_transformed)


# In[101]:


ridgeT = Ridge(alpha=1.0)
ridgeT.fit(X_train_scaled, y_train_scaled)
ridgeT_pred = ridge.predict(X_test_scaled)
ridgeT.score(X_train_scaled,y_train_scaled)


# In[103]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 800, criterion="poisson", n_jobs = -1, max_depth=15)
rf.fit(X_train_scaled, y_train_scaled)
r2 = rf.score(X_test_scaled, y_test_scaled)
print(r2)


# In[74]:


trees = rf.estimators_
for i, tree in enumerate(trees):
    print(f"Depth of tree {i}: {tree.tree_.max_depth}, ")


# In[116]:


rfT = RandomForestRegressor(n_estimators = 400, criterion="poisson", n_jobs = -1, max_depth=15)#, max_features=.75)
rfT.fit(X_train_transformed, y_train_transformed)
r2T = rfT.score(X_test_transformed, y_test_transformed)
print(r2T)


# In[104]:


for feature_name, feature_importance in zip(df_combined_features, rf.feature_importances_):
    print(f"{feature_name}: {feature_importance}")


# In[106]:


for feature_name, feature_importance in zip(df_combined_features, rfT.feature_importances_):
    print(f"{feature_name}: {feature_importance}")


# In[50]:


gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
gb_r2 = gb.score(X_test, y_test)
print(gb_r2)


# In[45]:


xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)


# In[160]:


df_combined_features.columns


# In[21]:


len(df_combined['totalShareStatistics.likeCount'].unique())


# In[164]:


for col in [c for c in df_combined.columns if "svd" in c or "lda" in c]:
    print(f"{col}: {df_combined[col].min()} to {df_combined[col].max()}")


# In[180]:


X_train, X_test, y_train, y_test = train_test_split(df_combined_features, 
                                                    #df_combined_target_like,
                                                    df_combined_target_like, 
                                                    test_size=0.2, 
                                                    random_state=0)


# In[181]:


categorical_features = [col for col in df_combined_features.columns if set(df_combined_features[col].unique()) == {0,1}]
continuous_features = [col for col in df_combined_features.columns if col not in categorical_features]


# In[188]:


spline_terms = [f(0, by=var, n_splines=4) for var in categorical_features]


# In[190]:


len(continuous_features)


# In[191]:


gam = LinearGAM(n_splines=10, verbose=False)


# In[115]:


gam.fit(X_train, y_train)


# In[197]:


gam.score(X_test, y_test)


# In[217]:


df_combined_features.loc[df_combined_features["adContext.dscStatus"] == 1, ["adContext.dscStatus", "adContext.isDsc"]].value_counts()


# In[114]:


int(df_combined_target_like.sum())/int(df_combined_target_like.count())


# In[ ]:




