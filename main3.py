from __future__ import print_function
import torch
import argparse
import os
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from typing import Union, List
import torch.nn.functional as F
from models.backbone import ResNet_18
import dataset.raf as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr2', '--learning-rate2', default=0.0001, type=float,
                    metavar='LR2', help='initial learning rate2')
parser.add_argument('--temperature', default=0.1, type=float,
                    help='temperature')
parser.add_argument('--num_workers', type=int, default=16,
                    help='num of workers to use')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')
# Device options
parser.add_argument('--gpu', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n_labeled', type=int, default=2000,
                    help='Number of labeled data')
parser.add_argument('--train-iteration', type=int, default=800,
                    help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--ema-decay', default=0.999, type=float)
# Data
parser.add_argument('--train-root', type=str, default='./RAFdataset/images/train',
                    help="root path to train data directory")
parser.add_argument('--test-root', type=str, default='./RAFdataset/images/test',
                    help="root path to test data directory")
parser.add_argument('--label-train', default='./RAFdataset/labels/RAF_train_label2.txt',
                    type=str, help='')
parser.add_argument('--label-test', default='./RAFdataset/labels/RAF_test_label2.txt',
                    type=str, help='')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
vocab_size = 7
d_model = 512
nhead = 8
num_layers = 3

dim_bank = 512
K_bank = args.n_labeled

global_bank = torch.zeros((dim_bank, K_bank))
# global_bank = nn.DataParallel(global_bank)
global_bank = global_bank.cuda().detach()
global_labels = torch.zeros(K_bank, dtype=torch.int)
# global_labels = nn.DataParallel(global_labels)
global_labels = global_labels.cuda().detach()


@torch.no_grad()
def update_bank(k, labels, index):
    global global_bank, global_labels
    global_bank[:, index] = k.t()
    global_labels[index] = labels.int()


def main():
    global best_acc
    torch.set_num_threads(3)
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing RAFDB')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomApply([
            transforms.RandomCrop(224, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # print("1:{}".format(torch.cuda.memory_allocated(0)))
    train_labeled_set, train_unlabeled_set, test_set = dataset.get_raf(args.train_root, args.label_train,
                                                                       args.test_root, args.label_test, args.n_labeled,
                                                                       transform_train=transform_train,
                                                                       transform_val=transform_val)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.num_workers)

    # Model
    print("==> creating ResNet-18")
    # print("2:{}".format(torch.cuda.memory_allocated(0)))
    model = ResNet_18(num_classes=7)
    model = model.cuda()
    # ema_model = create_image_model(ema=True)
    # print("3:{}".format(torch.cuda.memory_allocated(0)))
    model2 = TextEncoder(vocab_size, d_model, nhead, num_layers)
    model2 = model2.cuda()
    # model2 = create_text_model()
    # model2 = torch.nn.DataParallel(model2).cuda()
    # print("4:{}".format(torch.cuda.memory_allocated(0)))
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))  # 显示总参数量
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr2)
    # ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    bank = torch.zeros((512, 100)).cuda()
    bank = bank.clone().detach()
    start_epoch = args.start_epoch
    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):
        print('\nEpoch: [%d | %d] LR: %f ' % (
            epoch, args.epochs, state['lr']))
        train_loss, train_loss_s, train_loss_text, train_loss_cons, text_feature_matrix = train(labeled_trainloader,
                                                                                                unlabeled_trainloader,
                                                                                                model, model2,
                                                                                                optimizer, optimizer2,
                                                                                                criterion, use_cuda,
                                                                                                global_bank,epoch)
        # print(model.classifier.parameters())
        _ = validate(labeled_trainloader, model, text_feature_matrix, criterion, epoch, use_cuda, mode='Test Stats')
        test_acc = validate(test_loader, model, text_feature_matrix, criterion, epoch, use_cuda, mode='Test Stats')
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        # test_accs.append(test_acc)
    print('Best acc:')
    print(best_acc)


def train(labeled_trainloader, unlabeled_trainloader, model, model2, optimizer, optimizer2, criterion_ce,
          use_cuda, bank, epoch):
    # print("1:{}".format(torch.cuda.memory_allocated(0)))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_supervised = AverageMeter()
    losses_text = AverageMeter()
    losses_cons = AverageMeter()
    end = time.time()
    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    emotions = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
    # print("3:{}".format(17))
    model.train()
    # model2.train()
    model2.eval()
    # print("4:{}".format(torch.cuda.memory_allocated(0)))
    for batch_idx in range(args.train_iteration):
        with torch.no_grad():
            try:
                inputs_x, targets_x, index_x, idd = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, index_x, idd = labeled_train_iter.next()
            try:
                (inputs_u, inputs_u2, inputs_strong), _, index_u, _ = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u2, inputs_strong), _, index_u, _ = unlabeled_train_iter.next()
        # print("6:{}".format(torch.cuda.memory_allocated(0)))
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_strong = inputs_strong.cuda()
        # print("7:{}".format(torch.cuda.memory_allocated(0)))
        outputs_u, feature_u = model(inputs_u)
        output_strong, _ = model(inputs_strong)
        output_x, feature_x = model(inputs_x)
        # print("7.5:{}".format(torch.cuda.memory_allocated(0)))
        update_bank(feature_x, targets_x, index_x)
        # print("8:{}".format(torch.cuda.memory_allocated(0)))
        Ls = criterion_ce(output_x, targets_x.type(torch.long))
        Ls = Ls.mean()
        # print("9:{}".format(torch.cuda.memory_allocated(0)))
        text_features_matrix = None
        # #feature_x_normal = F.normalize(feature_x, dim=-1).cuda()
        # #text_features_list = []
        with torch.no_grad():
            for emotion in emotions:
                text = f"This is a face image of {emotion}"
                inputs_text = tokenize(text, vocab).cuda()
                outputs_text = model2(inputs_text)
                feature_text = F.normalize(outputs_text, dim=-1)
                # print(feature_u)
                if text_features_matrix is None:
                    text_features_matrix = feature_text
                else:
                    text_features_matrix = torch.cat([text_features_matrix, feature_text], dim=0)
        text_features_matrix = text_features_matrix.cuda().detach()
        # # #similarity = torch.mm(feature_x_normal, text_features_matrix.t())
        similarity = torch.mm(feature_x, text_features_matrix.t())
        L_text = criterion_ce(similarity, targets_x)
        L_text = L_text.mean()
        # print("10:{}".format(torch.cuda.memory_allocated(0)))
        # # unlabeled data processing
        with torch.no_grad():
            # p_unlabeled_1 = torch.softmax(outputs_u, dim=1)
            # p_unlabeled_1 = p_unlabeled_1.cuda()
            simmilarity2 = torch.mm(feature_u, text_features_matrix.t())
        #     #p_unlabeled_2 = torch.softmax(simmilarity2, dim=-1)
        #     #p_unlabeled_2 = p_unlabeled_2.cuda()
        bank = bank
        # logits_unlabeled = torch.mm(feature_u, global_bank)
        if epoch > 15:
          with torch.no_grad():
              logits_unlabeled = torch.mm(feature_u, bank)
          # logits_unlabeled = torch.zeros((2,100))
          p_unlabeled_3 = torch.zeros((batch_size, 7)).cuda()
          for i in range(batch_size):
              for j, label in enumerate(global_labels):
                  p_unlabeled_3[i, label] = torch.max(p_unlabeled_3[i, label], logits_unlabeled[i, j])
          # #p_unlabeled_3 = torch.softmax(p_unlabeled_3, dim=-1)
          p_fussion = 0.35 * outputs_u + 0.35 * simmilarity2 + 0.3 * p_unlabeled_3
        else:
          p_fussion = 0.5 * outputs_u + 0.5 * simmilarity2 
        p_fussion_normalized = F.softmax(p_fussion, dim=1)
        max_probs, max_idx = torch.max(p_fussion_normalized, dim=1)
        # max_probs = max_probs.detach()
        # max_idx = max_idx.detach()
        max_idx = max_idx.cuda()
        threshold = 0.8
        mask = max_probs > threshold 
        selected_outputs_strong = output_strong[mask]
        selected_max_idx = max_idx[mask]
        if selected_outputs_strong.nelement() == 0:
            L_con = torch.tensor(0.0, device=output_strong.device, requires_grad=True)
        else:
            L_con = criterion_ce(selected_outputs_strong, selected_max_idx)
            L_con = L_con.mean()
        
        # print("11:{}".format(torch.cuda.memory_allocated(0)))
        loss = 0.3 * Ls + 0.3 * L_text + 0.4 * L_con

        losses.update(loss.item(), inputs_x.size(0))
        losses_supervised.update(Ls.item(), inputs_x.size(0))
        losses_text.update(L_text.item(), inputs_x.size(0))
        losses_cons.update(L_con.item(), inputs_x.size(0))

        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer2.step()
        print("12:{}".format(torch.cuda.memory_allocated(0)))
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Loss_s: {loss_s:.4f} | Loss_text: {loss_text:.4f} | Loss_con: {loss_cons:.4f}'.format(
            batch=batch_idx + 1,
            size=args.train_iteration,
            total=bar.elapsed_td,
            loss=losses.avg,
            loss_s=losses_supervised.avg,
            loss_text=losses_text.avg,
            loss_cons=losses_cons.avg
        )
        bar.next()
    bar.finish()

    return (
        losses.avg, losses_supervised, losses_text, losses_cons, text_features_matrix)


def tokenize(texts: Union[str, List[str]], vocab: List[str], context_length: int = 77) -> torch.IntTensor:
    if isinstance(texts, str):
        texts = [texts]

    word_to_index = {word: index for index, word in enumerate(vocab)}

    result = torch.zeros(len(texts), context_length, dtype=torch.int)

    for i, text in enumerate(texts):
        tokens = text.lower().split()

        token_ids = [word_to_index.get(token, -1) for token in tokens if word_to_index.get(token, -1) != -1]

        if len(token_ids) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, :len(token_ids)] = torch.tensor(token_ids)

    return result


vocab = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(TextEncoder, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.ln_final = nn.LayerNorm(d_model)
        self.text_projection = nn.Linear(d_model, d_model)

    def forward(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:, :x.size(1), :].detach()
        x = x.permute(1, 0, 2)
        x = self.transformer.encoder(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[:, 0, :]
        x = self.text_projection(x)

        return x


def validate(valloader, model, text_feature_matrix, criterion, epoch, use_cuda, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    outputs_new = torch.ones(1, 7).cuda()
    targets_new = torch.ones(1).long().cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, idx, i) in enumerate(valloader):
            data_time.update(time.time() - end)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                logits, feature = model(inputs)
                batchsize_text = logits.size(0)
                p_unlabeled_1 = torch.softmax(logits, dim=-1)
                simmilarity2 = torch.mm(feature, text_feature_matrix.t())
                p_unlabeled_2 = torch.softmax(simmilarity2, dim=-1)
                p_unlabeled_2 = p_unlabeled_2.cuda()
                logits_unlabeled = feature @ global_bank
                p_unlabeled_3 = torch.zeros((batchsize_text, 7))
                p_unlabeled_3 = p_unlabeled_3.cuda()
                for i in range(batchsize_text):
                    for j, label in enumerate(global_labels):
                        p_unlabeled_3[i, label] = torch.max(p_unlabeled_3[i, label], logits_unlabeled[i, j])
                p_unlabeled_3 = torch.softmax(p_unlabeled_3, dim=-1)
                p_fussion = 0.35 * p_unlabeled_1 + 0.35 * p_unlabeled_2 + 0.3 * p_unlabeled_3
                # max_probs, max_idx = torch.max(p_fussion, dim=1)
                prec1, prec5 = accuracy(p_fussion, targets, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                bar.suffix = '({batch}/{size}) Total: {total:} | Accuracy: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(valloader),
                    total=bar.elapsed_td,
                    top1=top1.avg,
                )
                bar.next()
            bar.finish()
    print(top1.avg)

    return top1.avg


if __name__ == '__main__':
    main()
