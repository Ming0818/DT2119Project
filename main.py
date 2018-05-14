from ffnn import FFNN
from rnn import RNN
from metrics import get_f1_score, classification_report, get_accuracy


def test_ffnn():
    params = {'n_layers': 4, 'hidden_nodes': [512, 512, 512, 512],
              'epochs': 10, 'use_dynamic_features': True,
              'use_mspec': False, 'as_mat': False,
              'speaker_norm': False,
              'context_length': 17}
    net = FFNN(params)
    model = net.train_model()
    net.set_model(model)
    y_true, yp = net.predict_on_test()
    print("FFNN RESULTS")
    print(get_f1_score(y_true, yp))
    print(get_accuracy(y_true, yp))
    print(classification_report(y_true, yp))


def test_rnn():
    """Notice as_mat is true here!"""
    params = {'n_layers': 2, 'hidden_nodes': [32, 32],
              'epochs': 10, 'use_dynamic_features': True,
              'use_mspec': False, 'as_mat': True,
              'speaker_norm': False,
              'context_length': 13}
    net = RNN(params)
    model = net.train_model()
    net.set_model(model)
    y_true, yp = net.predict_on_test()
    print("RNN RESULTS")
    print(get_f1_score(y_true, yp))
    print(get_accuracy(y_true, yp))
    print(classification_report(y_true, yp))


if __name__ == "__main__":
    # test_ffnn()
    test_rnn()


