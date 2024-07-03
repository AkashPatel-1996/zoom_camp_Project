import pickle


def saveModel(model):
    with open('../SavedModel/lin_beg.bin', 'wb') as f_out:
        pickle.dump(model, f_out)