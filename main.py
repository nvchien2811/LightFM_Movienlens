import pandas as pd
import pickle
import itertools
from lightfm.data import Dataset
from recommenders.datasets import movielens
from recommenders.models.lightfm.lightfm_utils import  (similar_users, similar_items)
# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)

# Select MovieLens data size
def test(iduser,itemId,data):
    # MOVIELENS_DATA_SIZE = '100k'

    model2 = pickle.load(open("model.pkl", "rb"))

    # data = movielens.load_pandas_df(
    #     size=MOVIELENS_DATA_SIZE,
    #     genres_col='genre',
    #     header=["userID", "itemID", "rating"]
    # )
    # split the genre based on the separator
    movie_genre = [x.split('|') for x in data['genre']]

    # retrieve the all the unique genres in the data
    all_movie_genre = sorted(list(set(itertools.chain.from_iterable(movie_genre))))

    user_feature_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.user'
    # user_feature_URL = 'ml-100k/u.user'
    user_data = pd.read_table(user_feature_URL,
                              sep='|', header=None)
    user_data.columns = ['userID', 'age', 'gender', 'occupation', 'zipcode']

    new_data = data.merge(user_data[['userID', 'occupation']], left_on='userID', right_on='userID')
    # retrieve all the unique occupations in the data
    all_occupations = sorted(list(set(new_data['occupation'])))

    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                 data['itemID'],
                 item_features=all_movie_genre,
                 user_features=all_occupations)

    user_features = dataset2.build_user_features(
        (x, [y]) for x, y in zip(new_data.userID, new_data['occupation']))
    item_features = dataset2.build_item_features(
        (x, y) for x, y in zip(data.itemID, movie_genre))

    _, user_embeddings = model2.get_user_representations(features=user_features)
    user_embeddings

    result_user = similar_users(user_id=iduser, user_features=user_features,
                                model=model2)
    new_User = result_user.merge(user_data[['userID', 'age', 'gender', 'occupation', 'zipcode']], left_on='userID',
                                 right_on='userID')
    print(new_User)
    _, item_embeddings = model2.get_item_representations(features=item_features)
    item_embeddings

    result_item = similar_items(item_id=itemId, item_features=item_features,
                                model=model2)

    data_item_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.item'
    data_item = pd.read_table(data_item_URL,
                              sep='|', header=None, encoding="windows-1251")
    data_item.columns = ['itemID', 'nameMovie', 'year', 'none', 'url', 'Action', 'Adventure', 'Animation', "Children's",
                         'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                         'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'unknown']

    new_data2 = result_item.merge(data_item[['itemID', 'nameMovie','year']], left_on='itemID', right_on='itemID')
    print(new_data2)
    # new_data3 = result_user.merge()

    # print("xin chao")
    a = pd.DataFrame(new_data2, columns=['itemID','nameMovie', 'year','url'])
    result_recomender = a.to_numpy()
    # print(result_recomender[0])
    # print(type(result_recomender))
    # print("lần 1")
    # print(result_user)
    return result_recomender
# test(1,997)
# print("finish")




