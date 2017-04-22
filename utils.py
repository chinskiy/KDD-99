from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def model_validation(model, X_data_train, X_data_test, Y_data_train, Y_data_test, train=True):
    model.fit(X_data_train)
    if train:
        print('Train: ')
        predicted_train = model.predict(X_data_train)
        print(confusion_matrix(Y_data_train, predicted_train))
        print('accuracy: ', accuracy_score(Y_data_train, predicted_train))
        print('f1 score: ', f1_score(Y_data_train, predicted_train))

    print('Test: ')
    predicted_test = model.predict(X_data_test)
    print(confusion_matrix(Y_data_test, predicted_test))
    print('accuracy: ',accuracy_score(Y_data_test, predicted_test))
    print('f1 score: ', f1_score(Y_data_test, predicted_test))

def model_validation_supervised(model, X_data_train, X_data_test, Y_data_train, Y_data_test, train=True):
    model.fit(X_data_train, Y_data_train)
    if train:
        print('Train: ')
        predicted_train = model.predict(X_data_train)
        print(confusion_matrix(Y_data_train, predicted_train))
        print('accuracy: ', accuracy_score(Y_data_train, predicted_train))
        print('f1 score: ', f1_score(Y_data_train, predicted_train))

    print('Test: ')
    predicted_test = model.predict(X_data_test)
    print(confusion_matrix(Y_data_test, predicted_test))
    print('accuracy: ',accuracy_score(Y_data_test, predicted_test))
    print('f1 score: ', f1_score(Y_data_test, predicted_test))