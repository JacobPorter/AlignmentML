import pickle
from AlignmentTypeML3 import getX, features_to_remove, remove_columns_index, ColumnSelector

model_location = ""
feature_file = ""

model = pickle.load(open(model_location, "rb"))
X, X_ids, feature_labels = getX(feature_file)
remove_columns = ColumnSelector(remove_columns_index(feature_labels, features_to_remove), delete=True)
X = remove_columns.transform(X)
for label in features_to_remove:
    feature_labels.remove(label)
y_predict = model.predict(X)
for item in y_predict:
    print(item, file=sys.stdout)