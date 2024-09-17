""" Training and testing of the MoAGL-AM
"""
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim, init_scheduler
from utils import KNN, cal_sample_weight,one_hot_tensor

import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#INPUT DATA
def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    labels = np.concatenate((labels_tr, labels_te))
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    #single
    # data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(view_list[0]) + "_tr.csv"), delimiter=','))
    # data_te_list.append(np.loadtxt(os.path.join(data_folder, str(view_list[0]) + "_te.csv"), delimiter=','))

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    idx_dict = {}
    num = num_tr+num_te
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, num))
    print(idx_dict["tr"])
    print(idx_dict["te"])
    data_tr_newlist = []
    data_te_newlist = []
    for i in range(len(data_mat_list)):
        data_tr_newlist.append(data_mat_list[i][idx_dict["tr"]])
        data_te_newlist.append(data_mat_list[i][idx_dict["te"]])

    labels_newtr = labels[idx_dict["tr"]]
    labels_newte = labels[idx_dict["te"]]
    labels_all = np.concatenate((labels_newtr, labels_newte))

    data_tr_tensor_list = []
    data_te_tensor_list = []
    for i in range(num_view):
        data_tr_tensor_list.append(torch.FloatTensor(data_tr_newlist[i]))
        data_te_tensor_list.append(torch.FloatTensor(data_te_newlist[i]))
        if cuda:
            data_tr_tensor_list[i] = data_tr_tensor_list[i].cuda()
            data_te_tensor_list[i] = data_te_tensor_list[i].cuda()

    trte_dict = {}
    trte_dict["tr"] = list(range(num_tr))
    trte_dict["te"] = list(range(num_tr, (num_tr+num_te)))

    return data_tr_tensor_list, data_te_tensor_list, trte_dict, labels_all

#CONSINE_N
def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_train_list.append(KNN(data_tr_list[i], adj_parameter))
        adj_test_list.append(KNN(data_trte_list[i], adj_parameter))
        if cuda:
            adj_train_list[i] = adj_train_list[i].cuda()
            adj_test_list[i] = adj_test_list[i].cuda()
    return adj_train_list, adj_test_list

#TRAIN
def train_epoch(data_list, adj_tr_list, label, onehot_labels_tr_tensor, model_dict, optim_dict, sample_weight,scheduler, train_Ml=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss()
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)

    ci_list = []
    adp_list = []
    if train_Ml and num_view >= 2:
        optim_dict["H"].zero_grad()
        for i in range(num_view):
            adj_tr, gl_loss = model_dict["GL{:}".format(i + 1)](data_list[i], adj_tr_list[i])
            adp_list.append(adj_tr)
            out = model_dict["E{:}".format(i + 1)](data_list[i],  adp_list[i])
            adp_list.append( adj_tr_list[i])
            ci_list.append(out)

        z = model_dict["H"](num_view, ci_list, adp_list)
        c = model_dict["C"](z)
        c_loss_decoder = model_dict["D"](num_view, ci_list, z, adp_list)

        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight)) + 0.01 * c_loss_decoder + 0.01 * gl_loss

        c_loss.backward()
        optim_dict["H"].step()
        scheduler.step()
        loss_dict["H"] = c_loss.detach().cpu().numpy().item()
    return loss_dict

#TEST
def test_epoch(data_list, adj_te_list, te_idx, model_dict, adj_parameter, trte_idx, adaption):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)

    ci_list = []
    adp_list = []

    for i in range(num_view):
        adj_te, gl_loss = model_dict["GL{:}".format(i + 1)](data_list[i], adj_te_list[i])
        adp_list.append(adj_te)
        x = model_dict["E{:}".format(i + 1)](data_list[i], adp_list[i])
        ci_list.append(x)


    if num_view >= 2:
        z = model_dict["H"](num_view, ci_list, adp_list)
        c = model_dict["C"](z)
    else:
        z = ci_list[0]
        c = model_dict["C1"](z)

        # c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob

#MAIN
def train_test(data_folder, view_list, num_class, lr_c, lr_e, num_epoch, num_epoch_pretrain):
    print("data_folder:{}".format(data_folder))
    adaption = True
    print("adaption graph:{}".format(adaption))
    test_inverval = 50
    num_view = len(view_list)
    if data_folder == 'BRCA':
        adj_parameter = 8
        dim_he_list = [400, 400, 200]

    #DATA
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()

    # IF ADAPTION GRAPH LEARNING
    if adaption:
        adj_tr_list =[]
        adj_te_list = []
        for i in range(num_view):
            adj_tr_list.append(torch.ones(len(trte_idx["tr"]), len(trte_idx["tr"])))
            adj_te_list.append(torch.ones(len(trte_idx["te"]), len(trte_idx["te"])))
        if cuda:
            for i in range(num_view):
                adj_tr_list[i] = adj_tr_list[i].cuda()
                adj_te_list[i] = adj_te_list[i].cuda()
    else:
        adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)


    dim_list = [x.shape[1] for x in data_tr_list]


    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    optim_dict = init_optim(num_view, model_dict, lr_c, lr_e)
    scheduler = init_scheduler(optim_dict["H"])
    eva_acc = []
    eva_f1 = []
    eva_f1weighted = []
    eva_f1macro = []
    eva_auc = []
    train_loss = []
    print("\nTraining...")
    for epoch in range(num_epoch + 1):
        loss = train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, onehot_labels_tr_tensor, model_dict, optim_dict,
                           sample_weight_tr,scheduler, train_Ml=True)
        train_loss.append(loss["H"])
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict, adj_parameter, trte_idx, adaption)

            acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            f1weighted = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
            f1macro = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')


            print(" Epoch {:d}".format(epoch), "Test ACC: {:.5f}".format(acc),
                  "  F1 weighted: {:.5f}".format(f1weighted),
                  " F1 macro: {:.5f}".format(f1macro), " Loss:{:.5f}".format(train_loss[epoch]))
            eva_acc.append(acc)
            eva_f1weighted.append(f1weighted)
            eva_f1macro.append(f1macro)
