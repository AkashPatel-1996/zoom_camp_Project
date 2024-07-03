from Pipeline_functions import download_data, prepare_data, feature_extraction, find_best_model, train
from saveModel import saveModel


data = download_data("cancer")


df = prepare_data(data)

X, Y = feature_extraction(data)
print(X)

# model_param = find_best_model(X,Y)
# print(model_param)

# trained_model = train(X,Y, model_param)

# saveModel(trained_model)


