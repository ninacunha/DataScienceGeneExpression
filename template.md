# 1 - train test split
```py
X_trainlog, X_testlog, y_trainlog, y_testlog = train_test_split(X,
                                                    y,
                                                    random_state=314,
                                                    test_size=0.20,
                                                    shuffle=True) 
```
# 2 - cross validation (only on train)
```py
kf = KFold(n_splits=5, shuffle=True, random_state=314)

for train_index, test_index in kf.split(X_trainlog):
    X_train, X_test = X_trainlog.iloc[train_index], X_trainlog.iloc[test_index]
    y_train, y_test = y_trainlog.iloc[train_index], y_trainlog.iloc[test_index]
```
* You can standardize the data, idk what best practice is for the model - might need some research
* within this for loop, you can perform the gridsearch, with whatever parameters
* here is where you see which features are best, by whatever metrics you choose

# 3 - Train model on entire TRAIN set
```py
X_trainlog = X_trainlog[selected_features] # getting train data with only the features found in cross validation
X_testlog = X_testlog[selected_features]

log_model = LogisticRegression(max_iter=3000, random_state=42) # here is where you would use the tuned hyperparameters
log_model.fit(X_trainlog, y_trainlog)   # training the model

# then from there you can get the train/ test accuracy etc
```