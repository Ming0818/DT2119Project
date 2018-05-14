from ffnn import FFNN
from metrics import get_f1_score, classification_report, get_accuracy


if __name__ == "__main__":
    params = {'n_layers': 4, 'hidden_nodes': [512, 512, 512, 512],
              'epochs': 80, 'use_dynamic_features': True,
              'use_mspec': False, 'as_mat': False,
              'speaker_norm': False,
              'context_length': 17}
    net = FFNN(params)
    model = net.train_model()
    net.set_model(model)
    y_true, yp = net.predict_on_test()
    print(get_f1_score(y_true, yp))
    print(get_accuracy(y_true, yp))
    print(classification_report(y_true, yp))
    print(123)

