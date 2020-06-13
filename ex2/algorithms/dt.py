


from sklearn import tree, preprocessing
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import pandas as pd


# features = []
from globals import load_data as data


def load_data():
    df = data().sample(n=2000)

    y = df.iloc[:, 0]

    # take only features
    X = df.iloc[:, 1:]
    return X, y


def dt():
    X, y = load_data()

    # Split dataset into training set and test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

    feature_cols = list(X)

    print("=> do decision tree")

    # Create Decision Tree classifer object
    clf = tree.DecisionTreeClassifier(max_depth=5, max_features=10) #, criterion='entropy'

    # Train Decision Tree Classifer
    clf = clf.fit(X, y)

    # Predict the response for test dataset
    # y_pred = clf.predict(X_test)


    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')
    Image(graph.create_png())





