from __future__ import division
from __future__ import print_function

from config import FLAGS
from model import Model
from data_model import load_data
from train import train, test
from utils_our import get_model_info_as_str, check_flags
from saver import Saver
from eval import Eval

import torch


def main():
    train_data, test_data, val_data = load_data()
    print('Training...')
    if FLAGS.load_model is not None:
        print('loading model: {}', format(FLAGS.load_model))
        trained_model = Model(train_data).to(FLAGS.device)
        trained_model.load_state_dict(torch.load(FLAGS.load_model))
        print('model loaded')
    else:
        trained_model = train(train_data, saver, test_data=test_data)

    print("====================================")
    print('Testing...')
    eval = Eval(trained_model, train_data, test_data, val_data, saver,select=True)
    test(test_data, eval.trained_model, saver)
    eval.saver = saver
    eval.eval_on_test_data()


if __name__ == '__main__':
    print(get_model_info_as_str())
    check_flags()
    saver = Saver()
    main()