# -*- coding: utf-8 -*-
"""
Import and setup some auxiliary functions
"""

import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
from tqdm import tqdm
from torch.utils.data import random_split
from torch import nn
import tensorflow

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lambda_ = 0.001
epoch = 6
lr = 0.001


def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


def run(algorithm, dataset_name, filename):
    start = timeit.default_timer()
    predicted_test_labels, gt_labels = algorithm(dataset_name)
    if predicted_test_labels is None or gt_labels is None:
      return (0, 0, 0)
    stop = timeit.default_timer()
    run_time = stop - start
    
    np.savetxt(filename, np.asarray(predicted_test_labels))

    correct = 0
    total = 0
    for label, prediction in zip(gt_labels, predicted_test_labels):
      total += label.size(0)
      correct += (prediction.cpu().numpy() == label.cpu().numpy()).sum().item()   # assuming your model runs on GPU
      
    accuracy = float(correct) / total
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return (correct, accuracy, run_time)

class LogisticLinearRegression(nn.Module):
    def __init__(self, size):
        super(LogisticLinearRegression, self).__init__()
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

"""TODO: Implement Logistic Regression here"""

def logistic_regression(dataset_name):
  
  
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: implement logistic regression hyper-parameter tuning here

    if dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainingSet, validationSet = random_split(training, [len(training) - 12000, 12000])
        testSet = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        size = 32*32*3
    else:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
        training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainingSet, validationSet = random_split(training, [len(training)-12000, 12000])
        testSet = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform=transform)
        size = 28*28
      
    train_loader = torch.utils.data.DataLoader(trainingSet, batch_size = 128,
                                           shuffle=True, num_workers=2)

    validation_loader = torch.utils.data.DataLoader(validationSet, batch_size = 128,
                                                shuffle=True, num_workers=2)
    
    test_loader = torch.utils.data.DataLoader(testSet, batch_size = 1000,
                                          shuffle=False, num_workers=2)
    
    logisticLinModel = LogisticLinearRegression(size).to(device)
    optimizer = torch.optim.Adam(logisticLinModel.parameters(), lr=lr, weight_decay= lambda_)
    
    
    
    def train(epoch):
      logisticLinModel.train()
      
      for batch_idx, (data, target) in enumerate(train_loader):
        weight = 0
        for name, param in logisticLinModel.named_parameters():
          if name == 'fc.weight':
            weight = param

        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = logisticLinModel(data)
        outputProb = torch.nn.functional.softmax(output, dim = 1)
        loss = nn.functional.cross_entropy(outputProb, target) + (lambda_ * torch.norm(weight)) 
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
      
    def validation():
      logisticLinModel.eval()
      validation_loss = 0
      correct = 0
      with torch.no_grad():
        for data, target in validation_loader:
          data = data.to(device)
          target = target.to(device)
          output = logisticLinModel(data)
          pred = output.data.max(1, keepdim=False)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
      print("Validation set ran.")
          # print('\nValidation set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(validation_loader.dataset), 100. * correct / len(validation_loader.dataset)))

    
    def test():
      predicted = []
      correct_ = []
      logisticLinModel.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
        for data, target in test_loader:
          data = data.to(device)
          correct_.append(target.tolist())
          target = target.to(device)
          output = logisticLinModel(data)
          pred = output.data.max(1)[1]
          predicted.append(pred.tolist())
          correct += pred.eq(target.data.view_as(pred)).sum()
      
      print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format( correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
      return torch.tensor(predicted), torch.tensor(correct_)


    
    
    
    for i in range(1,epoch + 1):
      train(epoch)
      if i % 3 == 0:
        validation()
    predicted, correct_ = test()



    return predicted, correct_

"""TODO: Implement Hyper-parameter Tuning here"""

def tune_hyper_parameter():
    # TODO: implement logistic regression hyper-parameter tuning here
    start = timeit.default_timer()
    filenames = { "MNIST": "predictions_mnist_YourName_IDNumber.txt", "CIFAR10": "predictions_cifar10_YourName_IDNumber.txt"}
    lr_to_tune = [{'lr': 0.01}, {"lr": 0.001}, {"lr": 0.0001}]
    lambda_to_tune = [{"lambda": 0.01}, {"lambda": 0.001}, {"lambda": 0.0001}]
    bestAccuracy = 0.0
    best_params = {}

    for x in lr_to_tune:
      for y in lambda_to_tune:
        # change lr and lambda_ to test with
        global lr
        lr = x['lr']
        global lambda_
        lambda_ = y['lambda']
        result, score = run_on_dataset("CIFAR10", filenames["CIFAR10"])
        if result["accuracy"] > bestAccuracy:
          best_params.clear()
          bestAccuracy = result["accuracy"]
          best_params['lr'] = x['lr']
          best_params['lambda'] = y['lambda']
          print("New best parameters:", best_params)
    
    stop = timeit.default_timer()
    run_time = stop - start

    return best_params, bestAccuracy, run_time

"""Main loop. Run time and total score will be shown below."""

def run_on_dataset(dataset_name, filename):
    if dataset_name == "MNIST":
        min_thres = 0.82
        max_thres = 0.92

    elif dataset_name == "CIFAR10":
        min_thres = 0.28
        max_thres = 0.38

    correct_predict, accuracy, run_time = run(logistic_regression, dataset_name, filename)

    score = compute_score(accuracy, min_thres, max_thres)
    result = OrderedDict(correct_predict=correct_predict,
                         accuracy=accuracy, score=score,
                         run_time=run_time)
    return result, score


def main():
    filenames = { "MNIST": "predictions_mnist_YourName_IDNumber.txt", "CIFAR10": "predictions_cifar10_YourName_IDNumber.txt"}
    result_all = OrderedDict()
    score_weights = [0.5, 0.5]
    scores = []

    

    for dataset_name in ["MNIST","CIFAR10"]:
      if dataset_name == "CIFAR10":
        ### Uncomment out below block of code to run Part 1 with part 2 implemented. ###

        # params, accuracy, runtime = tune_hyper_parameter()
        # global lr
        # lr = params["lr"]
        # global lambda_
        # lambda_ = params["lambda"]
        # print("Most optimal parameters:", params)

        result_all[dataset_name], this_score = run_on_dataset(dataset_name, filenames[dataset_name])
        scores.append(this_score)
      else:
        result_all[dataset_name], this_score = run_on_dataset(dataset_name, filenames[dataset_name])
        scores.append(this_score)

    total_score = [score * weight for score, weight in zip(scores, score_weights)]
    total_score = np.asarray(total_score).sum().item()
    result_all['total_score'] = total_score
    with open('result.txt', 'w') as f:
        f.writelines(pformat(result_all, indent=4))
    print("\nResult:\n", pformat(result_all, indent=4))


main()
