import os
import numpy as np
import torch


cuda = True if torch.cuda.is_available() else False

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)
    
    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)

    return y_onehot



def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))

def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
            #print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()    
    return model_dict

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def euclidean_distance_torch(x1, x2=None):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    x2 = x1 if x2 is None else x2
    m, n = x1.size(0), x2.size(0)
    xx = torch.pow(x1, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(x2, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_( x1, x2.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-8).sqrt()  # for numerical stability
    return dist

def KNN(S, k, sigma=1.0):
    S = euclidean_distance_torch(S)

    N = len(S)

    A = torch.zeros((N, N))
    for i in range(N):

        dist_with_index = zip(S[i], range(N))

        dist_with_index = sorted(dist_with_index, key=lambda x: x[0])

        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour

            A[i][j] = torch.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually

    return A