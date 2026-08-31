import os

import os.path as osp
import sys
from utils.config import load_config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from utils import MLdataset
from utils import evaluation
from utils.utils import AverageMeter
import argparse
import time
from model_weight import PIRL
import torch
import numpy as np
from loss import Loss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy
import random


def distance(X, Y, square=True):
    '''
    Compute the squared Euclidean distance between each pair of vectors.
    :param X: Matrix X.
    :param Y: Matrix Y.
    :param square: Boolean to return squared distances or not.
    :return: Distance matrix.
    '''
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0) ** 2
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0) ** 2
    y = y.repeat(n, 1)
    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()  # Ensures non-negative distances
    if not square:
        result = torch.sqrt(result)
    return result


def build_laplacian(A, norm='sym'):
    """
    A: [n, n] adjacency matrix
    norm: 'none', 'sym', 'rw'
    """
    deg = A.sum(dim=1)  # [n]

    if norm == 'none':
        D = torch.diag(deg)
        L = D - A

    elif norm == 'sym':
        deg_inv_sqrt = torch.pow(deg + 1e-10, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        L = I - D_inv_sqrt @ A @ D_inv_sqrt

    elif norm == 'rw':
        deg_inv = 1.0 / (deg + 1e-10)
        D_inv = torch.diag(deg_inv)
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        L = I - D_inv @ A

    else:
        raise ValueError("norm must be one of ['none', 'sym', 'rw']")

    return L


def build_CAN(X, num_neighbors, links=0):
    '''
    Build the Clustering-with-Adaptive-Neighbors (CAN) graph.
    :param X: Data matrix.
    :param num_neighbors: Number of neighbors to consider for graph construction.
    :param links: Additional links to add to the graph (optional).
    :return: Tuple of weights matrix and raw weights matrix.
    '''
    size = X.shape[1]
    num_neighbors = min(num_neighbors, size - 1)
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).to(X.device)
        weights += torch.eye(size).to(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(X.device)
    weights = weights.to(X.device)
    return weights, raw_weights


def build_graphs(X, neighbor=10):
    lap = []
    for v in range(len(X)):
        X_v = torch.tensor(X[v]).T
        A_v, _ = build_CAN(X_v, neighbor)
        lap.append(build_laplacian(A_v))
    return lap


def train(loader, model, loss_model, opt, sche, epoch, logger, C_pos, C_neg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data = [v_data.to(device) for v_data in data]
        label = label.to(device)

        inc_V_ind = inc_V_ind.float().to(device)
        inc_L_ind = inc_L_ind.float().to(device)

        lap = build_graphs(data, neighbor=args.neighbor)
        target_pre, layer_logits = model(data, lap, inc_V_ind, Label=label,
                                                       teacher_forcing=args.teacher_forcing)

        BCE_loss_layer = []
        for k in range(args.T):
            layer_BCE_loss = loss_model.weighted_BCE_loss(
                layer_logits[k], label.to(device), inc_L_ind
            )
            loss_pos, loss_neg = loss_model.corr_criterion(layer_logits[k], label.to(device), inc_L_ind, C_pos, C_neg)
            BCE_loss_layer.append(layer_BCE_loss + args.lamb * (loss_pos + loss_neg))

            # print(layer_BCE_loss.item(), loss_pos.item(), loss_neg.item(), args.lamb * (loss_pos + loss_neg).item())

        layer_losses_t = torch.stack(BCE_loss_layer)  # [T]




        with torch.no_grad():
            model._hedge_update(layer_losses_t)

        # print(model.pi)
        pi = model.pi.detach().to(layer_losses_t.device)
        BCE_loss = (pi * layer_losses_t).sum()


        loss = BCE_loss

        opt.zero_grad()
        loss.backward()
        if isinstance(sche, CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))

        opt.step()
        # print(model.classifier.parameters().grad)
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
    if isinstance(sche, StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Data {data_time.avg:.3f}\t'
                'Loss {losses.avg:.3f}'.format(
        epoch, batch_time=batch_time,
        data_time=data_time, losses=losses))
    # print("all0",all0)
    return losses, model


def test(loader, model, loss_model, epoch, logger, mode='val'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        # data_time.update(time.time() - end)
        data = [v_data.to(device) for v_data in data]
        inc_V_ind = inc_V_ind.float().to(device)
        lap = build_graphs(data, neighbor=args.neighbor)
        target_pre, layer_logits = model.test(data, lap, inc_V_ind)
        pred = target_pre.cpu()
        total_labels = np.concatenate((total_labels, label.numpy()), axis=0) if len(total_labels) > 0 else label.numpy()
        total_preds = np.concatenate((total_preds, pred.detach().numpy()), axis=0) if len(
            total_preds) > 0 else pred.detach().numpy()

        batch_time.update(time.time() - end)
        end = time.time()
    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)

    if mode == 'val':
        evaluation_results = evaluation.do_metric2(total_preds, total_labels)
        evaluation_results = evaluation_results * 100
        logger.info('Epoch:[{0}]\t'
                    'Time {batch_time.avg:.3f}\t'
                    'AP {ap:.3f}\t'
                    '1-HL {hl:.3f}\t'
                    '1-RK {RK:.3f}\t'
                    '1-OE {oe:.3f}\t'.format(
            epoch, batch_time=batch_time,
            ap=evaluation_results[0],
            hl=evaluation_results[3],
            RK=evaluation_results[1],
            oe=evaluation_results[2]
        ))
    else:
        evaluation_results = evaluation.do_metric(total_preds, total_labels)
        evaluation_results = evaluation_results * 100
        logger.info('Epoch:[{0}]\t'
                    'Time {batch_time.avg:.3f}\t'
                    'AP {ap:.3f}\t'
                    '1-HL {hl:.3f}\t'
                    '1-RL {rl:.3f}\t'
                    'AUC {auc:.3f}\t'.format(
            epoch, batch_time=batch_time,
            ap=evaluation_results[0],
            hl=evaluation_results[1],
            rl=evaluation_results[2],
            auc=evaluation_results[3]
        ))

    return evaluation_results


def normalize_label_similarity(Y):
    """
    按论文中的条件概率定义计算标签相关矩阵:
        C[i, j] = P(L_j | L_i)

    Args:
        Y: [N, C], 0/1 multi-hot label matrix

    Returns:
        C: [C, C]
    """
    Y = Y.float()

    # [C, C], S[i, j] = sum_k Y[k, i] * Y[k, j]
    S = Y.T @ Y

    # [C], count[i] = sum_k Y[k, i]
    count = Y.sum(dim=0)

    # 按行归一化：C[i, j] = S[i, j] / count[i]
    C = S / (count[:, None] + 1e-8)

    return C
def construct_C(im_labels):
    train_im_labels = im_labels.float()

    train_S_im = normalize_label_similarity(train_im_labels).to(device)
    data = torch.load( f'./LLM_semantic_correlation_matrix/{ args.dataset}_{args.llm}_label_correlation.pt')

    S_pos_llm = data['S_pos_llm'].to(device)
    C_neg = data['S_neg_llm'].to(device)

    C_pos = train_S_im * (1 + args.mu * S_pos_llm) + args.mu2 * S_pos_llm
    C_pos = C_pos / C_pos.max().clamp_min(1e-8)
    return C_pos,C_neg


def main(args, file_path):
    data_path = osp.join(args.root_dir, args.dataset + '.mat')
    fold_data_path = osp.join(args.root_dir, 'folds/', args.dataset, args.dataset + '_MaskRatios_' + str(
        args.mask_view_ratio) + '_LabelMaskRatio_' +
                              str(args.mask_label_ratio) + '_TraindataRatio_' +
                              str(args.training_sample_ratio) + '.mat')
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir, args.name + args.dataset + '_V_' + str(
            args.mask_view_ratio) + '_L_' +
                           str(args.mask_label_ratio) + '_T_' +
                           str(args.training_sample_ratio) + '.txt')
    else:
        logfile = None
    logger = utils.setLogger(logfile)

    one_error = np.zeros(folds_num)
    coverage = np.zeros(folds_num)
    rk_loss = np.zeros(folds_num)
    AP_score = np.zeros(folds_num)

    auc_me = np.zeros(folds_num)
    time_list = np.zeros(folds_num)

    for fold_idx in range(folds_num):
        fold_idx = fold_idx
        inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num = MLdataset.loadMfDIMvMlDataFromMat(
            data_path, fold_data_path)
        train_dataloder, train_dataset = MLdataset.getIncDataloader(inc_mv_data, inc_labels, labels, inc_V_ind,
                                                                    inc_L_ind, total_sample_num,
                                                                    training_ratio=args.training_sample_ratio,
                                                                    fold_idx=fold_idx, mode='train',
                                                                    batch_size=args.batch_size, shuffle=True,
                                                                    num_workers=0)
        test_dataloder, test_dataset = MLdataset.getIncDataloader(inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind,
                                                                  total_sample_num,
                                                                  training_ratio=args.training_sample_ratio,
                                                                  val_ratio=args.val_sample_ratio, fold_idx=fold_idx,
                                                                  mode='test', batch_size=args.batch_size,
                                                                  num_workers=0)
        val_dataloder, val_dataset = MLdataset.getIncDataloader(inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind,
                                                                total_sample_num,
                                                                training_ratio=args.training_sample_ratio,
                                                                val_ratio=args.val_sample_ratio,
                                                                fold_idx=fold_idx, mode='val',
                                                                batch_size=args.batch_size, num_workers=0)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num


        C_pos,C_neg=construct_C(torch.tensor(train_dataset.cur_labels).float())

        start = time.time()
        model = PIRL(n_input=d_list, h_dim=args.h_dim * classes_num, Nlabel=classes_num, ista_T=args.T,beta=args.beta,alpha=args.alpha,  init_theta = args.init_theta).to(device)
        loss_model = Loss(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        scheduler = None

        logger.info('train_data_num:' + str(len(train_dataset)) + '  test_data_num:' + str(
            len(test_dataset)) + '   fold_idx:' + str(fold_idx))

        best_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        best_epoch = 0
        best_model_dict = {'model': model.state_dict(), 'epoch': 0}
        patience = 50
        for epoch in range(args.epochs):

            if epoch < 20:
                args.teacher_forcing = 0.5
            elif epoch < 60:
                args.teacher_forcing = 0.2
            else:
                args.teacher_forcing = 0.0

            train_losses, model = train(train_dataloder, model, loss_model, optimizer, scheduler, epoch, logger,
                                        C_pos.to(device), C_neg.to(device))
            val_results = test(val_dataloder, model, loss_model, epoch, logger)

            if val_results[0] * 0.25 + val_results[1] * 0.25 + val_results[2] * 0.25 + val_results[
                3] * 0.25 >= best_res:
                best_value_result = val_results
                best_res = val_results[0] * 0.25 + val_results[1] * 0.25 + val_results[2] * 0.25 + val_results[3] * 0.25
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
            if epoch > 20 and (best_value_result[0] - val_results[0] > 2) and wait >= patience:
                print('Training stopped: epoch=%d, best_epoch=%d, best_AP=%.7f' % (
                    epoch, best_epoch, best_value_result[0]))
                break
            train_losses_last = train_losses
            total_losses.update(train_losses.sum)

        model.load_state_dict(best_model_dict['model'])
        test_results = test(test_dataloder, model, loss_model, epoch, logger, 'test')

        logger.info(
            'final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,
                                                                                                             best_epoch,
                                                                                                             test_results[
                                                                                                                 0],
                                                                                                             test_results[
                                                                                                                 1],
                                                                                                             test_results[
                                                                                                                 2],
                                                                                                             test_results[
                                                                                                                 3]))

        AP_score[fold_idx] = test_results[0]
        rk_loss[fold_idx] = test_results[1]
        auc_me[fold_idx] = test_results[2]
        one_error[fold_idx] = test_results[3]
        coverage[fold_idx] = test_results[4]
        time_list[fold_idx] = time.time() - start
        if args.save_curve:
            np.save(osp.join(args.curve_dir, args.dataset + '_V_' + str(args.mask_view_ratio) + '_L_' + str(
                args.mask_label_ratio)) + '_' + str(fold_idx) + '.npy',
                    np.array(list(zip(epoch_results[0].vals, train_losses.vals))))

    file_handle = open(file_path, mode='a')

    res_str = (
        f"dataset: {args.dataset} "
        f"1-RL: {rk_loss.mean():.2f} ({rk_loss.std():.2f}) "
        f"AP: {AP_score.mean():.2f} ({AP_score.std():.2f}) "
        f"1-OE: {one_error.mean():.2f} ({one_error.std():.2f}) "
        f"1-Cov: {coverage.mean():.2f} ({coverage.std():.2f}) "
        f"AUC: {auc_me.mean():.2f} ({auc_me.std():.2f}) "
        f"Time: {time_list.mean():.2f} ({time_list.std():.2f}) \n"

    )
    print(res_str)
    file_handle.write(res_str)
    file_handle.close()


def filterparam(file_path, index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines) > 1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines]
    return params


def setup_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'records'))
    parser.add_argument('--file-path', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH',
                        default='./data/')
    parser.add_argument('--dataset', type=str, default='')  # mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=1, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)

    parser.add_argument('--name', type=str, default='5_sup_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)

    # Training args
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--neighbor', type=int, default=10)
    parser.add_argument('--h_dim', type=int, default=2)
    parser.add_argument('--init-theta', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--mu', type=float, default=0.5)
    parser.add_argument('--mu2', type=float, default=0.05)
    parser.add_argument('--lamb', type=float, default=0.01)

    args = parser.parse_args()

    device = torch.device('cuda:1')

    args.llm = "Qwen-2.5-14B-Instruct"
    ds = [
           '3sources',
             'emotions',
        'MIRFlickr_3view',
        'VOC07_3view',
        'corel5k_six_view',
        'iaprtc12_six_view',
    ]

    args.folds_num =10

    args.val_sample_ratio = 0.1
    args.training_sample_ratio = 0.1


    config_name = f'./config.yaml'

    for data in ds:
        args.dataset = data
        file_path = f"result//res_LabelMaskRatio{int(args.mask_label_ratio * 100)}_ViewMaskRatios{int(args.mask_view_ratio * 100)}_tr{int(args.training_sample_ratio * 100)}.txt"
        layer_config = load_config(config_name)
        args.T = layer_config[args.dataset]['layer']
        args.beta = layer_config[args.dataset]['beta']
        line = f'llm:{args.llm} val_ratio:{args.val_sample_ratio} train_ratio:{args.training_sample_ratio} mask_view_ratio: {args.mask_view_ratio} LabelMaskRatio:{args.mask_label_ratio}\n'

        file_handle = open(file_path, mode='a')
        file_handle.write(line)
        file_handle.close()
        setup_seed()
        try:
            main(args, file_path)
        except Exception as e:
            file_handle = open(file_path, mode='a')
            file_handle.write(e)
            file_handle.close()
            continue
        file_handle = open(file_path, mode='a')
        file_handle.write("\n")
        file_handle.close()
