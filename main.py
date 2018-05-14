from ffnn import FFNN
from metrics import f1_score, classification_report


if __name__ == "__main__":
    params = {'n_layers': 4, 'hidden_nodes': [256, 256],
              'epochs': 60, 'use_dynamic_features': True,
              'use_mspec': False, 'as_mat': False,
              'speaker_norm': False}
    net = FFNN(params)
    model = net.train_model()
    net.set_model(model)
    y_true, yp = net.predict_on_test()
    print(f1_score(y_true, yp))
    print(classification_report(y_true, yp))
    print(123)

