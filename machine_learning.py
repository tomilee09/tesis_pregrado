# funcion de xgboost que clasifica
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score

# Hacer entrenamiento con XGBOOST
def do_classification(X_train, y_train):
    # aqui se hace el machine learning, creo un modelo y lo entreno
    modelo = XGBClassifier(learning_rate=0.05,
                          max_depth=6,
                          n_estimators=500,
                          eval_metric=["logloss", "auc"]
                          )
    print("hola3")
    modelo.fit(X_train, y_train)
    
    return modelo



# encontrar mejor modelo con gridsearch
def find_best_model(X_train, y_train, X_test, y_test, X_val, y_val):

    # Define el modelo que deseas ajustar (por ejemplo, un clasificador RandomForest)
    modelo = XGBClassifier()

    # Define la grilla de hiperparámetros que deseas probar
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
    }

    # Realiza la búsqueda en cuadrícula con validación cruzada en el conjunto de entrenamiento
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    # Obtiene el mejor modelo con los mejores hiperparámetros
    best_model = grid_search.best_estimator_

    # Evalúa el mejor modelo utilizando validación cruzada en el conjunto de entrenamiento
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'Precisión en validación cruzada (promedio): {cv_scores.mean() * 100:.2f}%')

    # Evalúa el mejor modelo en el conjunto de validación independiente
    y_val_pred = best_model.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    print(f'Precisión en el conjunto de validación: {accuracy_val * 100:.2f}%')

    # Evalúa el mejor modelo en el conjunto de prueba independiente
    y_test_pred = best_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print(f'Precisión en el conjunto de prueba: {accuracy_test * 100:.2f}%')

    return best_model