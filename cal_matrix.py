from dataset.dataset import model_dataset
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import seaborn as sb
import matplotlib.pyplot as plt

def draw_heatmap(cfg, data):
    cls_name = cfg['cls_name']

    # data post process
    data = np.append(data, np.zeros((data.shape[0],1)), axis=1)
    origin_data = data.copy()

    for x_idx in range(len(cls_name)-1):
        row_total = np.sum(data[x_idx,:])

        for y_idx in range(len(cls_name)-1):
            data[x_idx, y_idx] = data[x_idx,y_idx]/row_total

        origin_data[x_idx, y_idx+1] = row_total
        data[x_idx, y_idx+1] = 1

    labels = np.asarray([])
    for idx, (percent, real) in enumerate(zip(data.flatten(), origin_data.flatten())):
        if idx % len(cls_name) == len(cls_name)-1:
            labels = np.append(labels, "{0:.0f}%\n{1}".format(percent * 100, int(real)))
        else:
            labels = np.append(labels, "{0:.2f}%\n{1}".format(percent * 100, int(real)))
    labels = labels.reshape(data.shape)

    # figure configurations
    plt.subplots(figsize=(8, 5))
    fig = sb.heatmap(data, cmap='Blues', annot=labels, xticklabels=cls_name, yticklabels=cls_name[:-1], fmt=''
               ,linewidths=3, vmin=0, vmax=1)
    plt.gca().xaxis.tick_top()
    plt.yticks(rotation=0)
    plt.title('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    cbar = fig.collections[0].colorbar
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0%', '100%'])
    plt.show()

def cal_matrix(cfg):
    # model construction & load pretrained model
    model = resnet18(num_classes=cfg['num_cls'])
    model = nn.parallel.DataParallel(model).cuda()

    state = torch.load(cfg['model_path'])
    model.load_state_dict(state["net"])
    model.eval()

    # result matrix
    emotion_mat = np.zeros(shape=(cfg['num_cls'], cfg['num_cls']))

    # validation dataloader
    val_set = model_dataset(cfg['data_path'], train=False)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_sz'], num_workers=cfg['num_wk'],
                            pin_memory=True, shuffle=False)

    for i, (images, targets) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
        images = images.cuda()
        targets = targets.cuda()

        # compute output, measure accuracy
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for idx in range(len(targets)):
            emotion_mat[targets[idx].item()][preds[idx].item()] += 1

    # accuracy
    total = np.sum(emotion_mat)
    correct = np.sum(np.diagonal(emotion_mat))
    print('total accuracy: {}'.format(round(correct / total, 5)))

    # result matrix
    print('result matrix: ')
    print(emotion_mat)
    return emotion_mat

if __name__ == '__main__':
    # configurations
    cfg = {}
    cfg['num_cls'] = 5
    cfg['batch_sz'] = 128
    cfg['num_wk'] = 0
    cfg['cls_name'] = ['angry', 'happy', 'sad', 'surprise', 'neutral', 'total']

    # original
    cfg['model_path'] = 'saved/checkpoints/resnet18'
    cfg['data_path'] = 'fer2013_original_5/PublicTest'

    emotion_mat = cal_matrix(cfg)
    draw_heatmap(cfg, emotion_mat)