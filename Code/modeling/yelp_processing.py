#!/usr/bin/env python
# coding: utf-8

# # Streamlit App Demo as a Python Notebook

def get_recs(user_text, features):

    # imports libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import time

    # standard sklearn imports
    from sklearn.model_selection import train_test_split, GridSearchCV

    # tensorflow imports for Neural Networks
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPooling2D, GRU, LSTM, Embedding, Bidirectional
    from tensorflow.keras.initializers import Constant
    from tensorflow.keras.optimizers import Adam

    # Import regularizers
    from tensorflow.keras.regularizers import l2
    # Import Dropout
    from tensorflow.keras.layers import Dropout

    from tensorflow.keras.utils import to_categorical, plot_model

    # imports for reports on classification
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

    # NLP imports 
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import nltk
    # nltk.download() # --> Download all, and then restart jupyter lab
    # nltk.download('stopwords')
    from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords

    plt.style.use(style='seaborn')
    # get_ipython().run_line_magic('matplotlib', 'inline')


    #------------------------------------------------------------------------------------------------

    # ## Part 1: Load the Model and Make a Prediction

    # ### Loads the saved model

    # Loads the keras model
    model = tf.keras.models.load_model('saved_model/my_model')

    ### Generates a prediction

    """# User input
    user_text = input('Describe the restaurant experience you want: ')
    """

    def clean_text_stem(text):
        """Cleans text by keeping words only, tokenizing, stemming and removing stopwords"""
        #Instantiate tokenizer and stemmer and lemmatizer
        re_tokenizer = RegexpTokenizer("\w+")
        lemmatizer = WordNetLemmatizer()
        p_stemmer = PorterStemmer()

        # Tokenze the text
        words = re_tokenizer.tokenize(text.lower())

        # Filter stop words
        stopwords_list = stopwords.words('english')

         # Adds custom stopwords to stopwords_list
        custom = []
        stopwords_list = stopwords_list + custom

        no_stops_stemmed = [p_stemmer.stem(word) for word in words if word.lower() not in stopwords_list]

        return (' ').join(no_stops_stemmed)

    # Stems the input text
    stemmed_text = clean_text_stem(user_text)

    # ### Transforms text for model input

    # Loads the data
    X_train = pd.read_csv('../../Data/X_train_to_tokenize.csv', index_col=0)
    X_train = X_train.squeeze()

    # Creates a function that counts unique words
    def counter_word(text):
        # import the Counter function
        from collections import Counter

        count = Counter()
        for doc in text.values:
            for word in doc.split():
                count[word] += 1
        return count

    # Counts the number of times a unique word appears
    counter = counter_word(X_train)

    num_words = len(counter)

    # Max number of words in a sequence
    max_length = 50

    # import the tokenizer from keras preprocessing 
    from tensorflow.keras.preprocessing.text import Tokenizer

    # Fit the tokenizer onto the train sentences 
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_train)

    # Now adding padding
    from tensorflow.keras.preprocessing.sequence import pad_sequences 

    # Creates the test sequences and padding
    user_input_sequences = tokenizer.texts_to_sequences(pd.Series(stemmed_text))
    user_input_padded = pad_sequences(
        user_input_sequences, maxlen=max_length, padding='post', truncating='post'
    )

    # Generates a prediction
    pred_probs = model.predict(user_input_padded)

    # Convert the probabilities to labels using a threshold value
    max_prob = max(pred_probs[0])
    preds = np.array([1 if pred_probs[0,i]>=max_prob else 0 for i in range(pred_probs.shape[1])])

    # Defining a dictionary with the name of the classes
    class_dict = {
      0: "neither casual nor classy",
      1: "casual",
      2: "classy",
      3: "casual and classy"
    }

    # Converting boolean array to class names
    finalPrediction=[]
    for idx, num in enumerate(preds):
        if(num == 1):
            finalPrediction.append(class_dict[idx])
            Ambience = [idx]


    # Creates a list of all the features
    # features = [GoodForKids, GoodForGroups, OutdoorSeating, Reservations, HasAlcohol, TableService, MealType]

    # Creates the array of features
    response_dict = {
        'Yes': [1],
        'No': [0],
        'Lunch': [0, 1],
        'Dinner': [1, 0],
        'Both': [1, 1],
        'Other': [0, 0]
    }

    feature_array = []
    for feature_input in features:
        feature_array += response_dict[feature_input]
    feature_array = feature_array + Ambience





    # ## Part 2: Recommendation System

    # imports 
    from scipy import sparse
    from sklearn.metrics.pairwise import pairwise_distances, cosine_distances, cosine_similarity

    # ### Loads in the Recommendation Table and Business Id to Names Table

    # Loads the dataframe that matches business ids to business names in case it's needed
    bid_name_df = pd.read_csv('../../Data/busid_to_name.csv')

    # Loads the recommendation table
    rec_df = pd.read_csv('../../Data/recommender_table.csv')

    # pulls out the index col
    rec_df[['business_id', 'name']] = rec_df['identifier'].str.split("|",expand=True).drop(columns=2)

    business_id = rec_df.pop('business_id')
    rec_df.insert(0, 'business_id', business_id)

    name = rec_df.pop('name')
    rec_df.insert(1, 'name', name)

    rec_df.drop(columns=['identifier', 'business_id'], inplace=True)
    rec_df.set_index('name', inplace=True)

    # ### Adds the user's feature array to the dataframe and calculates cosine distances

    feature_array_df = pd.DataFrame([feature_array], columns=rec_df.columns, index=['User Input'])

    rec_df_sample = pd.concat([rec_df, feature_array_df])

    # Creates the sparse matrix
    sparse_df = sparse.csr_matrix(rec_df_sample.fillna(-1))

    # returns the cosine distances between restaurants for every restaurant (therefore # rows = # of restaurants and # cols = # of restaurants )
    recommender = cosine_distances(sparse_df) 

    recommender_df = pd.DataFrame(recommender, index=rec_df_sample.index, columns=rec_df_sample.index)

    title = 'User Input'
    final_recs = recommender_df[title].sort_values()[1:11].to_frame().reset_index(drop=False).rename(columns={'index': 'Restaurant Recommendations'})[['Restaurant Recommendations']]


    return finalPrediction[0], final_recs

