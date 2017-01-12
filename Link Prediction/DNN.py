import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


feature_train = pd.read_csv('feature_train.csv')
feature_test = pd.read_csv('feature_test.csv')
label_train = pd.read_csv('label_train.csv')

training_features = preprocessing.scale(feature_train.values)
X_train, X_test, y_train, y_test = train_test_split(training_features, label_train.ix[:,0].values)


def dnn_for_binary_classification(X_train,X_test,y_train,y_test):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import SGD

    model = Sequential()
    model.add(Dense(20, input_dim=17, init='uniform', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, nb_epoch=1)
    pred = model.predict_classes(X_test, batch_size=32)
    print '\n f1 : '
    print f1_score(pred,y_test)

    


print 'DNN: '
dnn_for_binary_classification(X_train,X_test,y_train,y_test)



def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()

#Save_Obj(pred,'pred')


from sklearn.ensemble import RandomForestClassifier


def rfClassifier(X_train,X_test,y_train,y_test):
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(X_train,np.array(y_train))
    print '========Model Fitted=========='
    pred = rfc.predict(X_test)
    print '========Predict Finished======'
    print f1_score(pred,y_test)    


print 'random forest: '
rfClassifier(X_train,X_test,y_train,y_test)


