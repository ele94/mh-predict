Sequence:

1. prepare.py 
2. windowfy.py 
3. text_featurize.py and/or tfidf_featurize.py
4. (optional) combine_features.py
5. train.py
6. classify.py
7. evaluate.py and/or erisk_evaluate.py



1. prepare.py -> -> clean_train_users, clean_test_users
2. clean_train_users, clean_test_users -> windowfy.py -> train_x, train_y, test_x, test_y
3a. train_x, test_x -> text_featurize.py -> train_text_features, test_text_features
3b. train_x, test_x -> tfidf_featurize.py -> train_tfidf_features, test_tfidf_features, train_ngram_features, test_ngram_features
4. (train_text_features, test_text_features, train_tfidf_features, test_tfidf_features, train_ngram_features, test_ngram_features) -> combine_features.py -> train_c_features, test_c_features
5. (any train features), train_y -> train.py -> classifier
6. (any test features), classifier -> classify.py -> predictions, scores
7a. predictions -> evaluate.py -> NORMAL EVAL
7b. predictions, scores -> erisk_evaluate.py -> ERISK SEQUENTIAL EVAL
