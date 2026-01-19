import pandas as pd
import pickle
import numpy as np
import ast
from rapidfuzz import process, fuzz

with open("./models/EXPECTED_COLS.pkl", "rb") as f:
    data = pickle.load(f)
EXPECTED_COLS = data["EXPECTED_COLS"]
with open("./models/final_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
with open("./models/StandardScalers.pkl", "rb") as f:
    data = pickle.load(f)
SS_y = data["SS_y"]
SS_X = data["SS_X"]
continuous_cols = data["continuous_cols"]
with open("./models/confmodel.pkl", "rb") as f:
    data = pickle.load(f)
confmodel = data["confmodel"]
with open("./models/multihot.pkl", "rb") as f:
    data = pickle.load(f)
mlb_categories = data['mlb_categories']
mlb_audio = data['mlb_audio']
mlb_genres = data['mlb_genres']
mlb_lang = data['mlb_lang']
mlb_tags = data['mlb_tags']
tf_audio = data['tf_audio']
tf_lang = data['tf_lang']
tf_categories = data['tf_categories']
tf_genres = data['tf_genres']
tf_tags = data['tf_tags']
tsvd_audio = data['tsvd_audio']
tsvd_lang = data['tsvd_lang']
tsvd_categories = data['tsvd_categories']
tsvd_genres = data['tsvd_genres']
tsvd_tags = data['tsvd_tags']
df_results = pd.read_parquet('./data/df_results.parquet')

def col_to_feat(df):
    # workaround for fastapi deleting Nones
    df = df.reindex(columns=EXPECTED_COLS)
    
    # df contains df (original parquet) values
    # this function get it to look like df_X_z for the model
    # 2. convert datetime
    date_series = pd.to_datetime(
        df["release_date"],
        format="%b %d, %Y",   # Mon d, yyyy (e.g. Jan 3, 2024)
        errors="coerce"       # invalid formats → NaT
    )
    df["release_date"] = date_series

    # 3. convert estimated_owners to numeric
    df["estimated_owners"] = df["estimated_owners"].replace({
        '0 - 0': 0, 
        '0 - 20000': 10000,
        '20000 - 50000': 35000,
        '50000 - 100000': 75000,
        '100000 - 200000': 150000,
        '200000 - 500000': 350000,
        '500000 - 1000000': 750000,
        '1000000 - 2000000': 1500000,
        '2000000 - 5000000': 3500000,
        '5000000 - 10000000': 7500000,
        '10000000 - 20000000': 15000000,
        '20000000 - 50000000': 35000000,
        '50000000 - 100000000': 75000000,
        '100000000 - 200000000': 150000000
    }).astype("int64")

    # 4. convert multi categories
    def split_comma_list(s):
        if s is None:
            lst = []
        else:
            lst = [x.strip() for x in s.split(",")]
        return tuple(lst)
    def split_list_comma_list(s):
        if s is None:
            lst = []
        else:
            lst = ast.literal_eval(s.strip())
        return tuple(lst)
    df["categories"] = df["categories"].apply(split_comma_list)
    df["genres"] = df["genres"].apply(split_comma_list)
    df["tags"] = df["tags"].apply(split_comma_list)
    df["full_audio_languages"] = df["full_audio_languages"].apply(split_list_comma_list)
    df["supported_languages"] = df["supported_languages"].apply(split_list_comma_list)
    num_cols = ["achievements", 
        "average_playtime_forever", 
        "average_playtime_two_weeks", 
        "dlc_count",
        "median_playtime_forever",
        "median_playtime_two_weeks",
        "negative",
        "peak_ccu",
        "positive",
        "price",
        "recommendations",
        "required_age",
        "user_score",
        "estimated_owners"]
    df_X_num = pd.DataFrame()
    for i in range(len(num_cols)):
        df_X_num[f"{num_cols[i]}_is0"] = df[num_cols[i]].eq(0).where(df[num_cols[i]].notna())
        df_X_num[f"{num_cols[i]}_log1p"] = np.log1p(df[num_cols[i]])
    epoch = pd.Timestamp("1970-01-01")
    df["release_date"] = (pd.to_datetime(df["release_date"]) - epoch).dt.days
    
    
    df_X_info_avail = pd.DataFrame({
        "support_email_notna": df["support_email"].notna(),
        "about_the_game_notna": df["about_the_game"].notna(),
        "notes_notna": df["notes"].notna(),
        "reviews_notna": df["reviews"].notna(),
        "developers_notna": df["developers"].notna(),
        "publishers_notna": df["publishers"].notna(),
        "screenshots_notna": df["screenshots"].notna(),
        "movies_notna": df["movies"].notna(),
        "support_email_notna": df["support_email"].notna(),
        "support_url_notna": df["support_url"].notna(),
        "website_notna": df["website"].notna(),
        "categories_notna": df["categories"].apply(lambda x: len(x)==0),
        "genres_notna": df["genres"].apply(lambda x: len(x)==0),
        "tags_notna": df["tags"].apply(lambda x: len(x)==0),
        "full_audio_languages_notna": df["full_audio_languages"].apply(lambda x: len(x)==0),
        "supported_languages_notna": df["supported_languages"].apply(lambda x: len(x)==0),
    })
    #---------------------------multi hot----------------------------------

    MH_audio = mlb_audio.transform(df["full_audio_languages"])
    columns = [f"MH_full_audio_languages_{i+1}" for i in range(MH_audio.shape[1])]
    df_MH_audio = pd.DataFrame(MH_audio, columns=columns)
    dense_SVD_audio = tsvd_audio.transform(tf_audio.transform(df_MH_audio))
    columns = [f"SVD_full_audio_languages_{i+1}" for i in range(dense_SVD_audio.shape[1])]
    df_SVD_full_audio_languages = pd.DataFrame(dense_SVD_audio, columns=columns)

    MH_lang = mlb_lang.transform(df["supported_languages"])
    columns = [f"MH_supported_languages_{i+1}" for i in range(MH_lang.shape[1])]
    df_MH_lang = pd.DataFrame(MH_lang, columns=columns)
    dense_SVD_lang = tsvd_lang.transform(tf_lang.transform(df_MH_lang))
    columns = [f"SVD_supported_languages_{i+1}" for i in range(dense_SVD_lang.shape[1])]
    df_SVD_supported_languages = pd.DataFrame(dense_SVD_lang, columns=columns)

    MH_categories = mlb_categories.transform(df["categories"])
    columns = [f"MH_categories_{i+1}" for i in range(MH_categories.shape[1])]
    df_MH_categories = pd.DataFrame(MH_categories, columns=columns)
    dense_SVD_categories = tsvd_categories.transform(tf_categories.transform(df_MH_categories))
    columns = [f"SVD_categories_{i+1}" for i in range(dense_SVD_categories.shape[1])]
    df_SVD_categories = pd.DataFrame(dense_SVD_categories, columns=columns)

    MH_genres = mlb_genres.transform(df["genres"])
    columns = [f"MH_genres_{i+1}" for i in range(MH_genres.shape[1])]
    df_MH_genres = pd.DataFrame(MH_genres, columns=columns)
    dense_SVD_genres = tsvd_genres.transform(tf_genres.transform(df_MH_genres))
    columns = [f"SVD_genres_{i+1}" for i in range(dense_SVD_genres.shape[1])]
    df_SVD_genres = pd.DataFrame(dense_SVD_genres, columns=columns)

    MH_tags = mlb_tags.transform(df["tags"])
    columns = [f"MH_tags_{i+1}" for i in range(MH_tags.shape[1])]
    df_MH_tags = pd.DataFrame(MH_tags, columns=columns)
    dense_SVD_tags = tsvd_tags.transform(tf_tags.transform(df_MH_tags))
    columns = [f"SVD_tags_{i+1}" for i in range(dense_SVD_tags.shape[1])]
    df_SVD_tags = pd.DataFrame(dense_SVD_tags, columns=columns)

    df_X_SVD = pd.concat([df_SVD_categories,df_SVD_full_audio_languages,df_SVD_genres,df_SVD_supported_languages,df_SVD_tags],axis=1)
    #---------------------------multi hot----------------------------------
    
    df_X = pd.concat([df_X_num, df["release_date"], df_X_SVD, df[["linux","mac","windows"]],df_X_info_avail],axis=1)
    

    df_X_z = df_X.copy()
    df_X_z[continuous_cols] = SS_X.transform(df_X[continuous_cols])
    return df_X_z

def feat_to_pred(df_X_z):
    
    all_X = df_X_z.to_numpy(dtype="float32")
    y_pred = SS_y.inverse_transform(model.predict(all_X).reshape(-1, 1)).squeeze(-1)
    p_pred = confmodel.predict_proba(df_X_z)[:, 1]
    
    return (y_pred,p_pred)

def search_rows(col,q,
    fuzzy_threshold=80,fuzzy_topk=50):
    
    s = df_results[col].astype("string")

    if q is None or str(q).strip() == "":
        return df_results.iloc[0:0]  # empty result


    choices = s.fillna("").tolist()
    hits = process.extract(q, choices, scorer=fuzz.WRatio, limit=fuzzy_topk)
    keep = [(match, score, idx) for (match, score, idx) in hits if score >= fuzzy_threshold]
    if not keep:
        return df_results.iloc[0:0]
    idxs = [k[2] for k in keep]
    scores = [k[1] for k in keep]
    return df_results.iloc[idxs]