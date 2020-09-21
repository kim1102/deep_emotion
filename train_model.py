"""
Written by dev-kim
kim1102@kist.re.kr
2020.09.20
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import emotion_data
from resnet import resnet50


def evaluation(model, test_set1, test_set2):
    """
    :param model: torch deeplearning model
    :param test_set1: model_path_neutral
    :param test_set2: model_path_smile
    :return:
    """
    total = 0
    correct = 0

    # dataset configurations
    dataset = emotion_data(test_set1, test_set2)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # set model to evaluation mode
    model.eval()
    for idx, (labels, tensors) in tqdm(enumerate(test_loader)):
        output_result = model(tensors.cuda())
        _, predictions = torch.max(output_result, dim = 1)

        total += len(labels)
        preds = (labels.numpy() == predictions.cpu().numpy())
        correct += np.sum(preds)

    print("Evaluation result ===>", correct/ total)
    return correct/total


def train(neutral_data, smile_data):
    train_results = []
    test_results = []
    model = resnet50(pretrained= True)

    model.fc = nn.Linear(2048, 2)

    model.cuda()

    # dataset configurations
    dataset = emotion_data(neutral_data, smile_data)
    train_loader = DataLoader(dataset, batch_size = 4, shuffle= True, num_workers =0)

    # configurations for network
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss().cuda()
    total_epoch = 50
    save_epoch = 5
    save_path = 'trained_model/'

    for epoch in range(total_epoch):
        print(">>> Current_epoch:", epoch)
        model.train()
        for idx, (labels, tensors) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            output_tensors = model(tensors.cuda())
            loss = criterion(output_tensors, labels.cuda())

            loss.backward()
            optimizer.step()

        if epoch % save_epoch == 0:
            print("Current epoch: ", epoch)
            print("Current Loss: ", loss)
            train_results.append(evaluation(model, neutral_data, smile_data))
            #test_results.append(evaluation(model, test_set1, test_set2))
            torch.save(model.state_dict(), save_path + str(epoch) + '.pt')

    plt.plot(train_results)
    #plt.plot(test_results)
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    neutral_data = "D:/Data/FEI face/neutral"
    smile_data = "D:/Data/FEI face/smile"

    train(neutral_data, smile_data)

