from __future__ import print_function

import dynet_config
dynet_config.set(mem=1024,random_seed=123456789)
import dynet as dy
import numpy as np
import codecs
import random
from layers import MultiLayerPerceptron

dropout = 0.0

def predict(model, feats):
    x = dy.inputVector(feats)
    y = model(x)
    return y

def hinge_loss(model, example):
    y = predict(model, example[:-1])
    target = int(example[-1])
    scores = y.value()
    best_wrong = max([(i, sc) for i, sc in enumerate(scores) if i != target], key=lambda x: x[1])[0]
    if scores[target] < scores[best_wrong] + 1.0:
        return y[best_wrong] - y[target] + 1.0
    else:
        return dy.zeros(1)

def softmax_loss(model, example):
    y = predict(model, example[:-1])
    target = int(example[-1])
    return dy.pickneglogsoftmax(y, target)    

def read_examples(filename):
    examples = []
    with codecs.open(filename, "r", "utf-8") as fp:
        for line in fp:
            line = line.rstrip("\r\n")
            line = line.split(" ")
            example = [float(value) for value in line]
            examples.append(example)
    return examples

def evaluate(model, validation_data):
    correct = 0.0
    model.disable_dropout()
    for example in validation_data:
        y = predict(model, example[:-1])
        target = int(example[-1])
        predicted = np.argmax(y.npvalue())
        if target == predicted:
            correct += 1
    model.set_dropout(dropout)
    acc = correct / len(validation_data)
    print("accuracy: {0:.4}".format(acc))

if __name__ == "__main__":
    pc = dy.ParameterCollection()
    model = MultiLayerPerceptron(pc, 4, 10, 3, 1, act=dy.tanh)
    model.set_dropout(dropout)
    trainer = dy.AdamTrainer(pc)

    train_data = read_examples("../test/iris.txt")
    for epoch in range(100):
        random.shuffle(train_data)
        total_loss = 0
        for i, example in enumerate(train_data):
            dy.renew_cg()
            loss = hinge_loss(model, example)
            total_loss += loss.value()
            loss.backward()
            trainer.update()

        if (epoch % 10) == 0:
            print("average loss: {0}".format(total_loss / len(train_data)))
            evaluate(model, train_data)
