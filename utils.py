from sklearn.metrics import confusion_matrix, accuracy_score


def model_validation(model, X_data_train, X_data_test, Y_data_train, Y_data_test, train=True):
    model.fit(X_data_train)
    if train:
        print('Train:')
        print(accuracy_score(Y_data_train, model.predict(X_data_train)))
        print(confusion_matrix(Y_data_train, model.predict(X_data_train)))

    print('Test:')
    print(accuracy_score(Y_data_test, model.predict(X_data_test)))
    print(confusion_matrix(Y_data_test, model.predict(X_data_test)))
