import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns
import re
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


#import data with user descriptiomns/review
df_130= pd.read_csv('winemag-data-130k-v2.csv')


# #drop twiter info, and designtation
# test = df_130[df_130['country']=='US']
# test_drop = test.drop(columns=['designation','taster_name','taster_twitter_handle'])
# test_drop.isna().sum()

#creating new dataframe with just US data
def df_us ():
    states=df_130[df_130['country']=='US']
    states['province'].unique()
    return states
states= df_us()

#creating new dataframe with countries outside fo the US data
def df_other ():
    other_countries=df_130[df_130['country']!='US']
    other_countries['province'].unique()
    return other_countries

other_countries = df_other()



#if Region_2 had NaN values, put in values from Region_1
def update_region_2(region,df):
    df.loc[df['region_2'].isnull(),'region_2'] = df[region]
    return df

#updating states  region 2
states= update_region_2('region_1',states)
#updating other countries region 2
other_countries=update_region_2('province', other_countries)

# states.loc[states['region_2'].isnull(),'region_2'] = states['region_1']

#rename df province columns to stastes and dropping desingation, twitter name and handles
def drop(data):
    data=data.rename(columns={'province': 'state'})
    data=data.drop(columns='designation')
    data=data.drop(columns=['taster_name','taster_twitter_handle'])
    data=data.dropna()
    return data

#renaming states to df and dropping/renaming columns
df=drop(states)

#renaming other_countries  to df_other and dropping/renaming columns
df_other = drop(other_countries)

#rename df province columns to stastes and dropping desingation, twitter name and handles
# df=df.rename(columns={'province': 'state'})
# df=df.drop(columns='designation')
# df=df.drop(columns=['taster_name','taster_twitter_handle'])
# df


#mapping state names and assigning numerical values
# creating function to  to map state names & assinging numerical values
def map_state(data):
    gle = LabelEncoder()
    state_labels = gle.fit_transform(data['state'])
    state_mappings = {index: label for index, label in enumerate(gle.classes_)}
    data['state_label'] = state_labels
    return data

#mapping state names and assigning numerical values for US
df= map_state(df)

#mapping state names and assigning numerical values for other countries
df_other= map_state(df_other)

# one hot encoder for state labels
def ohe(data):
    gle = LabelEncoder()
    state_ohe = OneHotEncoder()
    state_labels = gle.fit_transform(data['state'])
    state_feature_arr = state_ohe.fit_transform(data[['state_label']]).toarray()
    state_feature_labels = list(gle.classes_)
    state_features = pd.DataFrame(state_feature_arr, columns=state_feature_labels)
    return state_features

#mapping  OHE for US states
df_ohe=ohe(df)

#mapping  OHE for other counries states
df_other_ohe= ohe(df_other)


#creating new variety column in dataframe to drop words after " - " in variety column aka wine types
def drop_dash_wine (data):
    data['new_v2']= data.variety.map(lambda x: x[0: x.find('-')] if '-' in x else x).str.lower()
    data['new_v2'].unique()
    return data

#dropping second wine name from US df
df= drop_dash_wine(df)

#dropping second wine name from other_country df
df_other= drop_dash_wine(df_other)

#creating wine category dictionary
wine_cat = {'red_bone_dry_bitter' : ['tannat', 'nebbiolo','sagrantino'],'red_bone_dry_savory':['cabernet franc','chianti', 'petit verdot','petite verdot','bordeaux','meritage', 'tempranillo','tempranillo blend','tinto fino','tinta de toro', 'french mourvedre', 'aglianico', 'barbera', 'montepulcaiano'], 'red_dry_herb':
['montepulciano','sangiovese','carmenere','cabernet france','cabernet sauvignon','cabernet'],'red_dry_floral':['gamay','mencía','mencia','valpolicella','rhone_blend','mourvèdre','rhône','beaujolais','burgundy','syrah','trincadeira','pinot noir','pinot nero'],'red_dry_spice':['dolcetto','garnacha', 'bonarda','amorone della valpolicella','negroamaro','nerello mascalese','supertuscans',
'merlot', 'alfrocheiro','alcicante bouschet'],'red_dry_fruit_vanilla':['shiraz','monastrell','malbec',"nero d'avola",'petite sirah','primitivo','zinfandel','grenache','g-s-m','g','touriga nacional'],'red_semi-sweet':['sangiovese grosso','lambrusco','brachetto'],'sweet':['port','banyuls','maury'],
'white_bone_dry': ['pinot grigio','pinot blanc','albarino','garganega','dry furmit','gavi','muscadet','melon','muscat','muskat','chablis','grenache blanc','macabeo','vinho verde','grillo','arinto'],'white_dry_herb':['sauvignon blanc','friulano','fumé blanc','fume blanc','sauvignon gris','sauvignon',
'verdejo','grüner veltliner','verdicchio','colombard'],'white_dry_tarte':['vermentino','turbiana','vernaccia','chenin blanc','torrontés'],'white_dry_sweet':['fiano','albariño','chardonnay','marsanne','roussanne','semillon','trebbiano'], 'white_dry_floral':['viura','pinot gris','sémillon','pinot blanc','pinot bianco','viognier','dry riesling'],'white_off_dry':[ 'riesling',
'johannisberg riesling','chenin blanc','torrontés','müller-thurgau'],'white_semi_sweet':['moscato','gewürztraminer'],'white_sweet':['sauternes','tokaji'],'white_very_sweet':['white port', 'moscatel dessert wine', 'passito wines','vin santo'],'red_blend':['red blend', 'corvina, rondinella, molinara'],'white_blend':['white blend'], 'rose':
['rose','rosé','rosato'],'sparkling':['champagne','prosecco','cava','sparkling blend','sparkling wine','champagne blend','glera']}


wine_cat_2 = {'red_dry' : ['tannat', 'nebbiolo','sagrantino'],'red_dry':['cabernet franc','chianti', 'petit verdot','petite verdot','bordeaux','meritage', 'tempranillo','tempranillo blend','tinto fino','tinta de toro', 'french mourvedre', 'aglianico', 'barbera', 'montepulcaiano'], 'red_dry':
['montepulciano','sangiovese','carmenere','cabernet france','cabernet sauvignon','cabernet'],'red_dry':['gamay','mencía','mencia','valpolicella','rhone_blend','mourvèdre','rhône','beaujolais','burgundy','syrah','trincadeira','pinot noir','pinot nero'],'red_dry':['dolcetto','garnacha', 'bonarda','amorone della valpolicella','negroamaro','nerello mascalese','supertuscans',
'merlot', 'alfrocheiro','alcicante bouschet'],'red_dry':['shiraz','monastrell','malbec',"nero d'avola",'petite sirah','primitivo','zinfandel','grenache','g-s-m','g','touriga nacional'],'red_dry':['sangiovese grosso','lambrusco','brachetto'],'red_sweet':['port','banyuls','maury'],
'white_dry': ['pinot grigio','pinot blanc','albarino','garganega','dry furmit','gavi','muscadet','melon','muscat','muskat','chablis','grenache blanc','macabeo','vinho verde','grillo','arinto'],'white_dry':['sauvignon blanc','friulano','fumé blanc','fume blanc','sauvignon gris','sauvignon',
'verdejo','grüner veltliner','verdicchio','colombard'],'white_dry':['vermentino','turbiana','vernaccia','chenin blanc','torrontés'],'white_dry':['fiano','albariño','chardonnay','marsanne','roussanne','semillon','trebbiano'], 'white_dry':['viura','pinot gris','sémillon','pinot blanc','pinot bianco',
'viognier','dry riesling'],'white_dry':[ 'riesling','johannisberg riesling','chenin blanc','torrontés','müller-thurgau'],'white_sweet':['moscato','gewürztraminer'],'white_sweet':['sauternes','tokaji'],'white_sweet':['white port', 'moscatel dessert wine', 'passito wines','vin santo'],'red_dry':['red blend', 'corvina, rondinella, molinara'],
'white_dry':['white blend'], 'rose':['rose','rosé','rosato'],'sparkling':['champagne','prosecco','cava','sparkling blend','sparkling wine','champagne blend','glera']}

# creating funciton to make new dictionary "updae",turned all values into keys ex: {wine_type: wine_category}
def update_dict (dict):
    update= {}
    for i in list(dict.keys()):
        for wine in wine_cat[i]:
            update[wine]=i
    return update
#creating update dictionary to add new categiries
update = update_dict(wine_cat)

# function to create  a dateframe from wine category/update dictionary
def wine_cat_df(dict):
    df_1= pd.DataFrame.from_dict(dict, orient='index', columns= ['category'])
    df_1['name']= df_1.index
    df_1['category1']=df_1['category']
    df_1=df_1.drop(columns= 'category')
    return df_1
df_1 = wine_cat_df(update)

#SUPER IMPORTANT
#function for mapping update dictionary to DF to make new coloumn for wine categories
def mapping (data, dict):
    data['category'] = data['new_v2'].map(dict,np.nan)
    return data

#mapping US wines
df = mapping (df, update)

#mapping other wines
df_other =mapping (df_other, update)


#for testing reasons, counting columns
def test_nan_categories (data):
    test_1 = data[data['category'].isnull()]
    unique_element, count_element=np.unique(test_1['new_v2'], return_counts=True)
    left_overs= pd.DataFrame(unique_element, count_element)
    pd.set_option('display.max_rows',125)
    return left_overs

#count column items:
def test_categories_counts (data, c_name):
    unique_element, count_element=np.unique(data[c_name], return_counts=True)
    dframe= pd.DataFrame(unique_element, count_element)
    return dframe

#function to  create final dataframe with dropping 10000 wines that we could not categorize
def drp_na_cat(data):
    final_list = data.dropna()
    return final_list

#final US list with droped NaN values from new categories
final_list = drp_na_cat(df)

#final US list with droped NaN values from new categories
final_list_other = drp_na_cat(df_other)


#function to label categories using encorder
def cat_coder(data):
    category_le = LabelEncoder()
    category_labels = category_le.fit_transform(data['category'])
    data['Gen_Label'] = category_labels
    return data

#adding numerical category to US list
final_list =cat_coder(final_list)

#adding numerical category to other list
final_list_other =cat_coder(final_list_other)

#function for one hot encoder categories
def ohe_category(data):
    category_le = LabelEncoder()
    cat_ohe = OneHotEncoder()
    category_labels = category_le.fit_transform(data['category'])
    cat_feature_arr = cat_ohe.fit_transform(data[['Gen_Label']]).toarray()
    cat_feature_labels = list(category_le.classes_)
    cat_features = pd.DataFrame(cat_feature_arr,columns=cat_feature_labels)
    return cat_features

#cat OHE us
cat_ohe_us =ohe_category(final_list )

#cat OHE other_country
cat_ohe_other =ohe_category(final_list_other)



#
# vectorizing text function
def vectorize_text(text):
    vectorizer1 = TfidfVectorizer(stop_words = 'english')
    vector1 = vectorizer1.fit_transform(text)
    return vector1, vectorizer1

#vectorize text using final_list dataframe (all dropped columns and new updated wine categories)-US
vector_vectorizer = vectorize_text(final_list['description'])

#vectorize text using final_list dataframe (all dropped columns and new updated wine categories)-oth
vector_vectorizer_other = vectorize_text(final_list_other['description'])

#creating vectorized array for words
def get_vector_df(text):
    vector_vectorizer = vectorize_text(text)
    vector_df = pd.DataFrame(vector_vectorizer[0].toarray(), columns = vector_vectorizer[1].get_feature_names())
    return vector_df

#assisgning funciton to get_vect variable- US
get_vect=get_vector_df(final_list['description'])

#assisgning funciton to get_vect variable-other
get_vect_other=get_vector_df(final_list_other['description'])


#creating new DF for the vectorized words
vect_df =pd.DataFrame(get_vect)

#returning sparse vector object
def sparse_vec(vectorizer):
    sparse_vector = vector_vectorizer[0]
    return sparse_vector

#sparse vector for US
sparse_vector =sparse_vec(vector_vectorizer)

#sparse vector for other countries
sparse_vector_other =sparse_vec(vector_vectorizer_other)

# return vectorizer
# vectorizer = get_vect[1]

#creating new dataframe to include proper index column to match the vect_df index, in order to map each data frame to one another
def create_index(data):
    update_index =data.reset_index(drop= True)
    return update_index

#creating new index for final_listn data frame to match vectorized df - US
df_10= create_index(final_list)

#creating new index for final_listn data frame to match vectorized df - other
df_10_other= create_index(final_list_other)

#creating dicitonary to rename df_10 columns so we dont have any overlap in vect_df columns when mapping. all columns end in _1
{'Unnamed: 0': 'old_index_1', 'country':'country_1', 'description':'description_1', 'points':'points_1', 'price':'price_1', 'state':'state_1','region_1':'region_1_1', 'region_2':'region_2_1', 'title':'title_1', 'variety':'variety_1', 'winery':'winery_1',
'state_label':'state_label_1','new_v2':'new_v2_1', 'category':'category_1', 'Gen_Label':'cat_label_1'}

#renaming df_10 columns to end in _1
def column_name_update (data):
    new_columns =data.rename(columns={'Unnamed: 0': 'old_index_1', 'country':'country_1', 'description':'description_1', 'points':'points_1', 'price':'price_1', 'state':'state_1','region_1':'region_1_1', 'region_2':'region_2_1', 'title':'title_1', 'variety':'variety_1',
    'winery':'winery_1','state_label':'state_label_1','new_v2':'new_v2_1', 'category':'category_1', 'Gen_Label':'cat_label_1'})
    return new_columns
#updateing US df_10 columns to end in 1
df_10 = column_name_update (df_10)

#updateing other df_10_other columns to end in 1
df_10_other = column_name_update (df_10_other)


#adding column to extract year
def add_vintage(dataframe):
    year_list = [re.findall(r'[0-9]{4}', item) for item in dataframe['title_1']]
    for item in year_list:
        if item == []:
            item.append('0')
    year_list = [year[0] for year in year_list]
    dataframe['vintage_1'] = year_list
    return dataframe

# making new column for US list
df_10= add_vintage(df_10)

#making new column for other countries list
df_10_other= add_vintage(df_10_other)

#creating  list to have both df_10 and vect_df in order to join
def create_frame( data, vect):
    frame = [data,vect]
    return frame

#US frame
frame= create_frame(df_10,get_vect)

#other frame
frame_other= create_frame(df_10_other,get_vect_other)
#
#creating new data frame concating DF_10 and vect_df. Use Axis =1 in order to join them by columns
def wine_vect_words (frame_list):
    wine_df_final= pd.concat(frame_list, axis=1)
    return wine_df_final

#create US vector list
final_wine=wine_vect_words(frame)

#create other vector list
final_wine_other=wine_vect_words(frame_other)



# import plotly
# from plotly.offline import iplot, init_notebook_mode
# init_notebook_mode(connected=True)
# scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
#            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
#
#
#
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
#
# for col in df.columns:
#     df[col] = df[col].astype(str)
#
# scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
#            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
#
# # df_10['text'] = df_10['state_1']
#
# df_states = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
# df_states = pd.DataFrame(df_states['state'])
# state_numbers = df_10['state_1'].value_counts()
#
# dict_state= state_numbers.to_dict()
# df_states['counts']=df_states['state'].map(dict_state)
# df_states.loc[4, 'counts'] = 35683
#
# data = [ dict(
#        type='choropleth',
#         colorscale = scl,
#        autocolorscale = False,
#        locations = df['code'],
#        z = df_states['counts'],
#        locationmode = 'USA-states',
# #         text = df['text'],
#        marker = dict(
#            line = dict (
#                color = 'rgb(255,255,255)',
#                width = 2
#            )
#        ),
#        colorbar = dict(
#            title = "# Wines"
#        )
#    ) ]
#
# layout = dict(
#        title = 'Wine Concentration by State<br>(Hover for breakdown)',
#        geo = dict(
#            scope='usa',
#            projection=dict( type='albers usa' ),
#            showlakes = True,
#            lakecolor = 'rgb(255, 255, 255)',
#        ),
#    )
#
# fig = dict( data=data, layout=layout )
#
# url = iplot( fig, filename='d3-cloropleth-map' )
