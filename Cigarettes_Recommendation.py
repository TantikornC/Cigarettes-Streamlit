import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from scipy.spatial.distance import cosine

def main():
    st.markdown("<h1 style='text-align: center;'>Cigarettes Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("#")
    st.markdown("#")

    st.markdown("<h1 style='font-size:30px'>1) Data Importing and Data Exploration</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:20px'>1.1) Importing all the libraries</h1>", unsafe_allow_html=True)
    st.write("")
    code = "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.decomposition import NMF\nfrom scipy.spatial.distance import cosine"
    st.code(code, language='python')
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>1.2) Reading the data</h1>", unsafe_allow_html=True)
    st.write("")
    code = "df = pd.read_csv('smokerdata.csv')\ndf.head()"
    st.code(code, language='python')
    df = pd.read_csv('smokerdata.csv')
    st.dataframe(df.head())
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>1.3) Check if there are any null values</h1>", unsafe_allow_html=True)
    st.write("")
    code = "pd.DataFrame(df.isnull().sum(), columns=['Amount']).T"
    st.code(code, language='python')
    st.dataframe(pd.DataFrame(df.isnull().sum(), columns=['Amount']).T)
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>1.4) Check the distributions of data</h1>", unsafe_allow_html=True)
    st.write("")
    code = "# Numerical data\ndf.describe().T"
    st.code(code, language='python')
    st.dataframe(df.describe().T)
    st.write("")
    code = """df_rating = pd.DataFrame(columns=['Rating', 'Amount'])
df_rating['Amount'] = df.groupby('Rating').count()['User'].values
df_rating['Rating'] = range(1,6)
fig_bar = px.bar(df_rating, x='Rating', y='Amount', color='Rating')
fig_bar"""
    st.code(code, language='python')
    df_rating = pd.DataFrame(columns=['Rating', 'Amount'])
    df_rating['Amount'] = df.groupby('Rating').count()['User'].values
    df_rating['Rating'] = range(1,6)
    fig_bar = px.bar(df_rating, x='Rating', y='Amount', color='Rating')
    st.plotly_chart(fig_bar)
    st.write("")
    code = """user_rating = pd.DataFrame()
user_rating['Count'] = df.groupby('User').count().groupby('Rating').count()['Brand']
user_rating['The number of times a user rates'] = df.groupby('User').count().groupby('Rating').count()['Brand'].index
user_rating['The number of times a user rates'] = user_rating['The number of times a user rates'].apply(lambda x: str(x))
fig_bar = px.bar(user_rating, x='The number of times a user rates', y='Count', color='Count')
fig_bar"""
    st.code(code, language='python')
    user_rating = pd.DataFrame()
    user_rating['Count'] = df.groupby('User').count().groupby('Rating').count()['Brand']
    user_rating['The number of times a user rates'] = df.groupby('User').count().groupby('Rating').count()['Brand'].index
    user_rating['The number of times a user rates'] = user_rating['The number of times a user rates'].apply(lambda x: str(x))
    fig_bar = px.bar(user_rating, x='The number of times a user rates', y='Count', color='Count')
    st.plotly_chart(fig_bar)
    st.write("")
    code = "# Non numerical data\ndf.describe(exclude=int).T"
    st.code(code, language='python')
    st.dataframe(df.describe(exclude=int).T)
    st.write("")
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>1.5) Creating a new column named Brand_Variety</h1>", unsafe_allow_html=True)
    st.write("")
    code = "df['Brand_Variety'] = df['Brand'] + ' ' + df['Variety']\ndf_content = df[df['Rating'] > 1].copy()\ndf_content.head()"
    st.code(code, language='python')
    df['Brand_Variety'] = df['Brand'] + ' ' + df['Variety']
    df_content = df[df['Rating'] > 1].copy()
    st.dataframe(df_content.head())
    st.write("")
    code = "# Check the total amount of brands\nlen(df_content['Brand_Variety'].unique())"
    st.code(code, language='python')
    st.code(len(df_content['Brand_Variety'].unique()), language='python')
    st.markdown("#")

    st.markdown("<h1 style='font-size:30px'>2) Content-based Filtering Recommendation Method</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:20px'>2.1) Split the data for training and testing</h1>", unsafe_allow_html=True)
    st.write("")
    code = "X = df_content[['Brand_Variety', 'Strength', 'Taste', 'Price']].copy()\ny = df_content['User'].copy()\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)"
    st.code(code, language='python')
    X = df_content[['Brand_Variety', 'Strength', 'Taste', 'Price']].copy()
    y = df_content['User'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>2.2) Creating a new dataframe named cig_mode_df</h1>", unsafe_allow_html=True)
    st.write("")
    code = """Brand_list = df_content['Brand_Variety'].unique().tolist()
temp_list = []
for i, brand in enumerate(Brand_list):
    temp_list.append([
        X_train[X_train['Brand_Variety'] == brand].mode()['Strength'][0],
        X_train[X_train['Brand_Variety'] == brand].mode()['Taste'][0],
        X_train[X_train['Brand_Variety'] == brand].mode()['Price'][0],
    ])
cig_mode_df = pd.DataFrame(temp_list, columns=['Strength', 'Taste', 'Price'])
cig_mode_df['Brand_Variety'] = Brand_list
cig_mode_df.head()"""
    st.code(code, language='python')
    Brand_list = df_content['Brand_Variety'].unique().tolist()
    temp_list = []
    for i, brand in enumerate(Brand_list):
        temp_list.append([
            X_train[X_train['Brand_Variety'] == brand].mode()['Strength'][0],
            X_train[X_train['Brand_Variety'] == brand].mode()['Taste'][0],
            X_train[X_train['Brand_Variety'] == brand].mode()['Price'][0],
        ])
    cig_mode_df = pd.DataFrame(temp_list, columns=['Strength', 'Taste', 'Price'])
    cig_mode_df['Brand_Variety'] = Brand_list
    st.dataframe(cig_mode_df.head())
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>2.3) Creating a function for cigarettes recommendations</h1>", unsafe_allow_html=True)
    st.write("")
    code = """def cig_recommend(S, T, P, topn): # S stands for strength, T for taste, and P for price
    Str_list = ['Very Strong','Strong', 'Medium',  'Weak', 'Very Weak']
    Taste_list = ['Very Pleasant', 'Pleasant', 'Tolerable', 'Poor', 'Very Poor']
    Price_list = ['Very Low', 'Low', 'Fair', 'High', 'Very High']
    cig_df = cig_mode_df[(cig_mode_df['Strength'] == S) & (cig_mode_df['Taste'] == T) & (cig_mode_df['Price'] == P)].copy()
    if len(cig_df) < topn:
        for Str in Str_list:
            if Str != S:
                cig_df2 = cig_mode_df[(cig_mode_df['Strength'] == Str) & (cig_mode_df['Taste'] == T) & (cig_mode_df['Price'] == P)].copy()
                cig_df = pd.concat([cig_df, cig_df2], ignore_index= True)
            if len(cig_df) >= topn:
                return cig_df[:topn]
        for Taste in Taste_list:
            if Taste != T:
                cig_df2 = cig_mode_df[(cig_mode_df['Strength'] == S) & (cig_mode_df['Taste'] == Taste) & (cig_mode_df['Price'] == P)].copy()
                cig_df = pd.concat([cig_df, cig_df2], ignore_index= True)
            if len(cig_df) >= topn:
                return cig_df[:topn]
        for Price in Price_list:
            if Price != P:
                cig_df2 = cig_mode_df[(cig_mode_df['Strength'] == S) & (cig_mode_df['Taste'] == T) & (cig_mode_df['Price'] == Price)].copy()
                cig_df = pd.concat([cig_df, cig_df2], ignore_index= True)
            if len(cig_df) >= topn:
                return cig_df[:topn]
    return cig_df[:topn]"""
    st.code(code, language='python')
    st.write("")
    st.write("**:blue[Try for yourself!]**")
    STR = st.selectbox(
        '**Strength**',
        ('Very Strong','Strong', 'Medium',  'Weak', 'Very Weak')
    )
    Taste = st.selectbox(
        '**Taste**',
        ('Very Pleasant', 'Pleasant', 'Tolerable', 'Poor', 'Very Poor')
    )
    Price = st.selectbox(
        '**Price**',
        ('Very Low', 'Low', 'Fair', 'High', 'Very High')
    )
    topn = round(st.number_input('**Topn**', min_value=1, step=1))
    code = "cig_recommend('{}', '{}', '{}', {}).reset_index(drop=True)".format(STR, Taste, Price, topn)
    st.code(code, language='python')
    def cig_recommend(S, T, P, topn): # S stands for strength, T for taste, and P for price
        Str_list = ['Very Strong','Strong', 'Medium',  'Weak', 'Very Weak']
        Taste_list = ['Very Pleasant', 'Pleasant', 'Tolerable', 'Poor', 'Very Poor']
        Price_list = ['Very Low', 'Low', 'Fair', 'High', 'Very High']
        cig_df = cig_mode_df[(cig_mode_df['Strength'] == S) & (cig_mode_df['Taste'] == T) & (cig_mode_df['Price'] == P)].copy()
        if len(cig_df) < topn:
            for Str in Str_list:
                if Str != S:
                    cig_df2 = cig_mode_df[(cig_mode_df['Strength'] == Str) & (cig_mode_df['Taste'] == T) & (cig_mode_df['Price'] == P)].copy()
                    cig_df = pd.concat([cig_df, cig_df2], ignore_index= True)
                if len(cig_df) >= topn:
                    return cig_df[:topn]
            for Taste in Taste_list:
                if Taste != T:
                    cig_df2 = cig_mode_df[(cig_mode_df['Strength'] == S) & (cig_mode_df['Taste'] == Taste) & (cig_mode_df['Price'] == P)].copy()
                    cig_df = pd.concat([cig_df, cig_df2], ignore_index= True)
                if len(cig_df) >= topn:
                    return cig_df[:topn]
            for Price in Price_list:
                if Price != P:
                    cig_df2 = cig_mode_df[(cig_mode_df['Strength'] == S) & (cig_mode_df['Taste'] == T) & (cig_mode_df['Price'] == Price)].copy()
                    cig_df = pd.concat([cig_df, cig_df2], ignore_index= True)
                if len(cig_df) >= topn:
                    return cig_df[:topn]
        return cig_df[:topn]
    temp = cig_recommend(STR, Taste, Price, topn).reset_index(drop=True)
    if temp.shape[0] < topn:
        temp.index = range(1, temp.shape[0]+1)
    else:
        temp.index = range(1, topn+1)
    st.dataframe(temp)
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>2.4) Model evaluation</h1>", unsafe_allow_html=True)
    st.write("")
    code = """score = 0
for i in range(len(X_test)):
    S = X_test.iloc[i]['Strength']
    T = X_test.iloc[i]['Taste']
    P = X_test.iloc[i]['Price']
    predicted = cig_recommend(S, T, P, 10)
    target_brand = X_test.iloc[i]['Brand_Variety']
    for brand in predicted['Brand_Variety']:
        if target_brand == brand:
            score += 1
print("Accuracy score: {:.2f} % (from {} test cases)".format(score/(i+1)*100, i+1))"""
    st.code(code, language='python')
    score = 0
    for i in range(len(X_test)):
        S = X_test.iloc[i]['Strength']
        T = X_test.iloc[i]['Taste']
        P = X_test.iloc[i]['Price']
        predicted = cig_recommend(S, T, P, 10)
        target_brand = X_test.iloc[i]['Brand_Variety']
        for brand in predicted['Brand_Variety']:
            if target_brand == brand:
                score += 1
    code = "Accuracy score: {:.2f} % (from {} test cases)".format(score/(len(X_test))*100, len(X_test))
    st.code(code, language='python')
    st.markdown("#")

    st.markdown("<h1 style='font-size:30px'>3) Collaborative Filtering Recommendation Method</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:25px'>3.1) Model-based</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h1 style='font-size:20px'>3.1.1) Re-splitting the data</h1>", unsafe_allow_html=True)
    st.write("")
    code = """X = df[['Brand_Variety', 'Rating']].copy()
y = df['User'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)"""
    st.code(code, language='python')
    X = df[['Brand_Variety', 'Rating']].copy()
    y = df['User'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>3.1.2) Creating a cross tab matrix between User and Brand_Variety from the training data</h1>", unsafe_allow_html=True)
    st.write("")
    code = """crosstab_matrix = pd.crosstab(y_train, X_train['Brand_Variety'], X_train['Rating'], aggfunc='mean')
crosstab_matrix.fillna(value=0, inplace=True)
crosstab_matrix.head()
crosstab_matrix.reset_index(drop=True, inplace=True)"""
    st.code(code, language='python')
    crosstab_matrix = pd.crosstab(y_train, X_train['Brand_Variety'], X_train['Rating'], aggfunc='mean')
    crosstab_matrix.fillna(value=0, inplace=True)
    crosstab_matrix.reset_index(drop=True, inplace=True)
    st.dataframe(crosstab_matrix.head())
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>3.1.3) Decompose crosstab_matrix into two matrices using NMF</h1>", unsafe_allow_html=True)
    st.write("")
    code = """nmf = NMF(n_components=30)
nmf.fit(crosstab_matrix)
H = pd.DataFrame(np.round(nmf.components_, 2), columns=crosstab_matrix.columns)
W = pd.DataFrame(np.round(nmf.transform(crosstab_matrix), 2), columns=H.index)
recommend_matrix = pd.DataFrame(np.round(np.dot(W, H), 2), columns=H.columns)
H"""
    st.code(code, language='python')
    nmf = NMF(n_components=30)
    nmf.fit(crosstab_matrix)
    H = pd.DataFrame(np.round(nmf.components_, 2), columns=crosstab_matrix.columns)
    W = pd.DataFrame(np.round(nmf.transform(crosstab_matrix), 2), columns=H.index)
    recommend_matrix = pd.DataFrame(np.round(np.dot(W, H), 2), columns=H.columns)
    st.dataframe(H)
    st.write("")
    code = 'W'
    st.code(code, language='python')
    st.dataframe(W)
    st.write("")
    code = 'recommend_matrix'
    st.code(code, language='python')
    st.dataframe(recommend_matrix)
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>3.1.4) Creating a function for cigarettes recommendations</h1>", unsafe_allow_html=True)
    st.write("")
    code = """def cig_recommend_model(uid, topn):
    rec_cig_brands = recommend_matrix.iloc[uid].sort_values(ascending=False)
    rec_cig_brands = rec_cig_brands[rec_cig_brands > 0].index
    recommend_df = pd.DataFrame(columns=cig_moded_df.columns)
    for brand in rec_cig_brands:
        recommend_df = pd.concat([recommend_df, cig_moded_df[cig_moded_df['Brand_Variety'] == brand]])
    return recommend_df[:topn]"""
    st.code(code, language='python')
    st.write("")
    def cig_recommend_model(uid, topn):
        rec_cig_brands = recommend_matrix.iloc[uid].sort_values(ascending=False)
        rec_cig_brands = rec_cig_brands[rec_cig_brands > 0].index
        recommend_df = pd.DataFrame(columns=cig_mode_df.columns)
        for brand in rec_cig_brands:
            recommend_df = pd.concat([recommend_df, cig_mode_df[cig_mode_df['Brand_Variety'] == brand]])
        return recommend_df[:topn]
    st.write('**:blue[Try for yourself!]**')
    uid = round(st.number_input('**User id**',min_value=0, max_value=recommend_matrix.index[-1], step=1))
    topn = round(st.number_input('**Topn** ', min_value=1, step=1))
    code = "cig_recommend_model({}, {}).reset_index(drop=True)".format(uid, topn)
    st.code(code, language='python')
    temp = cig_recommend_model(uid, topn).reset_index(drop=True)
    temp.index = range(1, topn+1)
    st.dataframe(temp)
    st.markdown("#")

    st.markdown("<h1 style='font-size:25px'>3.2) Memory-based</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h1 style='font-size:20px'>3.2.1) Creating a cross tab matrix between User and Brand_Variety from the testing data</h1>", unsafe_allow_html=True)
    st.write("")
    code = """new_user_matrix = pd.crosstab(y_test, X_test['Brand_Variety'], X_test['Rating'], aggfunc='mean')
new_user_matrix.fillna(value=0, inplace=True)
new_user_matrix.reset_index(drop=True, inplace=True)
new_user_matrix.head()"""
    st.code(code, language='python')
    new_user_matrix = pd.crosstab(y_test, X_test['Brand_Variety'], X_test['Rating'], aggfunc='mean')
    new_user_matrix.fillna(value=0, inplace=True)
    new_user_matrix.reset_index(drop=True, inplace=True)
    st.dataframe(new_user_matrix.head())
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>3.2.2) Creating a function for cigarettes recommendations</h1>", unsafe_allow_html=True)
    st.write("")
    code = """def cig_recommend_memory(uid, topn):
    similar_list = []
    target = new_user_matrix.iloc[uid]
    target_brands = target[target > 0].index
    for row in range(crosstab_matrix.shape[0]):
        similarity = 1 - cosine(target, crosstab_matrix.iloc[row]) # cosine([1,0,0], [0.5,0,0]) = 0
        if similarity != 1:
            similar_list.append(similarity)
        else:
            similar_list.append(0)
    most_similar_user = np.argsort(similar_list)[::-1]
    rec_cig_brands = [target_brands]
    for id in most_similar_user:
        temp = crosstab_matrix.iloc[id]
        brands = temp[temp > 0].index
        for brand in brands:
            if len(rec_cig_brands) == topn+1:
                return pd.DataFrame(rec_cig_brands[1:], index=range(1, topn+1), columns=['Brand_Variety'])
            if brand not in rec_cig_brands:
                rec_cig_brands.append(brand)
    return rec_cig_brands[1:]"""
    def cig_recommend_memory(uid, topn):
        similar_list = []
        target = new_user_matrix.iloc[uid]
        target_brands = target[target > 0].index
        for row in range(crosstab_matrix.shape[0]):
            similarity = 1 - cosine(target, crosstab_matrix.iloc[row]) # cosine([1,0,0], [0.5,0,0]) = 0
            if similarity != 1:
                similar_list.append(similarity)
            else:
                similar_list.append(0)
        most_similar_user = np.argsort(similar_list)[::-1]
        rec_cig_brands = [target_brands]
        for id in most_similar_user:
            temp = crosstab_matrix.iloc[id]
            brands = temp[temp > 0].index
            for brand in brands:
                if len(rec_cig_brands) == topn+1:
                    return pd.DataFrame(rec_cig_brands[1:], index=range(1, topn+1), columns=['Brand_Variety'])
                if brand not in rec_cig_brands:
                    rec_cig_brands.append(brand)
        return rec_cig_brands[1:]
    st.code(code, language='python')
    st.markdown("#")

    st.markdown("<h1 style='font-size:20px'>3.2.3) Making new_user_matrix's columns as same as recommend_matrix</h1>", unsafe_allow_html=True)
    st.write("")
    code = """diff_columns = set(recommend_matrix.columns) - set(new_user_matrix.columns)
for column in diff_columns:
    new_user_matrix = new_user_matrix.assign(column=[0]*new_user_matrix.shape[0])
new_user_matrix = new_user_matrix.reindex(columns=recommend_matrix.columns)
new_user_matrix.fillna(value=0, inplace=True)"""
    diff_columns = set(crosstab_matrix.columns) - set(new_user_matrix.columns)
    for column in diff_columns:
        new_user_matrix = new_user_matrix.assign(column=[0]*new_user_matrix.shape[0])
    new_user_matrix = new_user_matrix.reindex(columns=crosstab_matrix.columns)
    new_user_matrix.fillna(value=0, inplace=True)
    st.code(code, language='python')
    st.write("")
    st.write("**:blue[Try for yourself!]**")
    uid = round(st.number_input('**User id** ',min_value=0, max_value=new_user_matrix.index[-1], step=1))
    topn = round(st.number_input('**Topn**  ', min_value=1, step=1))
    code = "cig_recommend_memory({}, {})".format(uid, topn)
    st.code(code, language='python')
    st.dataframe(cig_recommend_memory(uid, topn))

if __name__ == "__main__":
    main()
