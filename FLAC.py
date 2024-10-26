from utils import *
import numpy as np
import  copy

from models.fcn import ConvNet, MLP, ResNet
from utils import *


def computer_cos(vec1, vec2):
    vec1 = vec1.detach().cpu().numpy()
    vec2 = vec2.detach().cpu().numpy()
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 1 - cos_sim


def net_norm(net):
    norm = 0.0
    norm_1 = 0.0
    for p_index, p in enumerate(net.parameters()):
        norm_1 += torch.norm(list(net.parameters())[p_index]) ** 2
        norm += torch.norm(p.data) ** 2
    return torch.sqrt(norm_1).item()


def computer_cos_model_state(model_state_list):
    juzheng = torch.zeros([len(model_state_list), len(model_state_list)])
    for i in range(0, len(model_state_list)):
        for j in range(i + 1, len(model_state_list)):
            juzheng[i][j] = computer_cos(model_state_list[i], model_state_list[j])
            juzheng[j][i] = juzheng[i][j]
    return juzheng


def jiaquan_parameters(net):
    flat_params = []

    for p in net.parameters():
        flat_params.append(p.view(-1))

    vector_net = torch.cat(flat_params, dim=0)

    if vector_net.numel() > 0:
        last_param_size = flat_params[-1].numel()
        last_param_modified = 1.5 * flat_params[-1]
        vector_net[-last_param_size:] = last_param_modified.view(-1)

    return vector_net


def vector_parameters(net):
    flat_params = []
    for p in net.parameters():
        flat_params.append(p.view(-1))
    vector_net = torch.cat(flat_params, dim=0)

    return vector_net


class DisjointSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.rank[pu] < self.rank[pv]:
            self.parent[pu] = pv
        elif self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu
        else:
            self.parent[pv] = pu
            self.rank[pu] += 1
        return True


def kruskal(matrix):
    edges = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if matrix[i][j] != 0:
                edges.append((i, j, matrix[i][j]))
    edges.sort(key=lambda x: x[2])

    disjoint_set = DisjointSet(len(matrix))
    min_spanning_tree = []

    for edge in edges:
        u, v, weight = edge
        if disjoint_set.union(u, v):
            min_spanning_tree.append((u, v, weight))

    return min_spanning_tree


def divide_trees(min_spanning_tree, num_nodes):
    max_weight = 0
    max_edge = None
    for edge in min_spanning_tree:
        if edge[2] > max_weight:
            max_weight = edge[2]
            max_edge = edge

    min_spanning_tree.remove(max_edge)
    adj_list = [[] for _ in range(num_nodes)]
    for edge in min_spanning_tree:
        u, v, weight = edge
        adj_list[u].append((v, weight))
        adj_list[v].append((u, weight))

    visited = [False] * num_nodes
    group1 = set()

    def dfs(node):
        visited[node] = True
        group1.add(node)
        for neighbor, weight in adj_list[node]:
            if not visited[neighbor]:
                dfs(neighbor)

    start_node = max_edge[0]
    dfs(start_node)

    group2 = set(range(num_nodes)) - group1

    return list(group1), list(group2)


class myDBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, dist_matrix):
        n = len(dist_matrix)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if not visited[i]:
                visited[i] = True
                neighbors = self.region_query(dist_matrix, i)

                if len(neighbors) < self.min_samples:
                    clusters.append([-1])
                else:
                    new_cluster = [i]
                    clusters.append(new_cluster)
                    self.expand_cluster(dist_matrix, neighbors, visited, new_cluster)

        return clusters

    def region_query(self, dist_matrix, i):
        neighbors = []
        for j, dist in enumerate(dist_matrix[i]):
            if dist < self.eps:
                neighbors.append(j)
        return neighbors

    def expand_cluster(self, dist_matrix, neighbors, visited, cluster):
        for i in neighbors:
            if not visited[i]:
                visited[i] = True
                new_neighbors = self.region_query(dist_matrix, i)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            if i not in cluster:
                cluster.append(i)




def compute_mean_var(xiangsidu, group):
    number = len(group)
    if number == 0 or number == 1:
        return 0.0, 0.0
    if number == 2:
        return xiangsidu[group[0]][group[1]], 0.0
    values = []
    for i in range(number):
        for j in range(i + 1, number):
            values.append(xiangsidu[group[i]][group[j]])
    mean_value = np.mean(values)
    var_value = np.var(values)
    return mean_value, var_value


def compute_eps(xiangsidu):
    triangle = []
    n = len(xiangsidu)
    for i in range(n):
        for j in range(i + 1, n):
            triangle.append(xiangsidu[i][j])

    triangle = np.sort(triangle)
    return np.mean(triangle), np.median(triangle), np.percentile(triangle, 55)


def compute_fenweishu(xiangsidu):
    triangle = []
    n = len(xiangsidu)
    for i in range(n):
        for j in range(i + 1, n):
            triangle.append(xiangsidu[i][j])

    triangle = np.sort(triangle)
    fenweishu = []
    for i in range(10, 100, 10):
        fenweishu.append(np.percentile(triangle, i))
    return fenweishu


def jiaoji(a1, a2):
    res = []
    for i in a1:
        if i in a2:
            res.append(i)
    return res


def bingji(a1, a2):
    res = []
    for i in a1:
        res.append(i)
    for i in a2:
        if i not in res:
            res.append(i)
    return res


def compute_norm_diff(gs_model, vanilla_model):
    norm_diff = 0.0
    for p_index, p in enumerate(gs_model.parameters()):
        norm_diff += torch.norm(list(gs_model.parameters())[p_index] - list(vanilla_model.parameters())[p_index]) ** 2
    norm_diff = torch.sqrt(norm_diff).item()
    return norm_diff


def read_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
        index_bias += p.numel()


def caijan(client_model, client_norm_diff, yuzhi):
    vectorized_net = vector_parameters(client_model)
    clipped_weight = vectorized_net / max(1, client_norm_diff / yuzhi)

    read_model_weight(client_model, clipped_weight)


def caijian_2(net_list_new, norm_diff_defence, yuzhi, net_freq_new):
    net_freq = len(net_freq_new) * [0.0]
    for i in range(len(net_list_new)):
        if norm_diff_defence[i] > yuzhi:
            net_freq[i] = yuzhi / norm_diff_defence[i] * net_freq_new[i]
        else:
            net_freq[i] = net_freq_new[i]
    total = 0.0
    for i in net_freq:
        total += i
    for i in range(len(net_freq)):
        net_freq[i] /= total

    return net_freq


def normalize(matrix):
    non_zero_elements = matrix[matrix > 0]

    second_min_val = torch.min(non_zero_elements)

    matrix_replaced = torch.where(matrix == 0, second_min_val, matrix)

    min_val = torch.min(matrix_replaced)
    max_val = torch.max(matrix_replaced)

    normalized_matrix = (matrix_replaced - min_val) / (max_val - min_val)

    return normalized_matrix


def mydefence(client_models, net_freq, selected_user_indices, flr, device, argsmodel, maxiter=500, eps=1e-5, ftol=1e-7):
    print("------------------------------------------- my defence method -------------------------------------")
    fc4_list = []
    for net_index, net in enumerate(client_models):
        net_vec = jiaquan_parameters(net)
        fc4_list.append(net_vec)

    print(len(fc4_list), fc4_list[0].size())
    n_class = client_models[0].n_classes
    seq_len = client_models[0].n_in

    juzheng = torch.zeros([len(fc4_list), len(fc4_list)])
    for i in range(0, len(fc4_list)):
        for j in range(i + 1, len(fc4_list)):
            juzheng[i][j] = computer_cos(fc4_list[i], fc4_list[j])
            juzheng[j][i] = juzheng[i][j]
    print("juzheng: ", juzheng.shape)
    model_state_list = []
    if argsmodel == "mlp":
        model_fcn = MLP(n_in=seq_len, n_classes=n_class).to(device)
    elif argsmodel == "resnet":
        model_fcn = ResNet(n_in=seq_len, n_classes=n_class).to(device)
    else:
        model_fcn = ConvNet(n_in=seq_len, n_classes=n_class).to(device)
    index = flr - 3
    if index < 5:
        index = 0
    model1_path = "./client_model/quanju/" + str(index) + "_round_model.pth"
    model_fcn.load_state_dict(torch.load(model1_path))
    for idx, global_user_idx in enumerate(selected_user_indices):
        model_fcn.load_state_dict(torch.load(model1_path))
        for p_index, p in enumerate(model_fcn.parameters()):
            p.data -= list(client_models[idx].parameters())[p_index].data
        save_model_path = "./client_model/diff/" + str(global_user_idx) + "_" + str(flr) + "_round_model.pth"
        torch.save(model_fcn, save_model_path)
        fcn_params = []
        for layer in model_fcn.modules():
            if isinstance(layer, nn.Conv2d):
                fcn_params.append(layer.weight.data.view(-1))
                if layer.bias is not None:
                    fcn_params.append(layer.bias.data)
            if isinstance(layer, nn.Linear):
                fcn_params.append(layer.weight.data.view(-1))
                if layer.bias is not None:
                    fcn_params.append(layer.bias.data)
        fcn_params = torch.cat(fcn_params, dim=0).cpu()

        model_state_list.append(fcn_params)
    cos_model_state = computer_cos_model_state(model_state_list)
    alpha = 0.5
    juzheng = normalize(juzheng)
    cos_model_state = normalize(cos_model_state)
    xiangsidu = (1 - alpha) * juzheng + alpha * cos_model_state
    print(juzheng)
    print(cos_model_state)
    print(xiangsidu)
    min_spanning_tree = kruskal(xiangsidu)
    num_nodes = len(xiangsidu)
    group1, group2 = divide_trees(min_spanning_tree, num_nodes)
    var1, var2 = 0.0, 0.0
    mean1, mean2 = 0.0, 0.0
    mean1, var1 = compute_mean_var(xiangsidu, group1)
    mean2, var2 = compute_mean_var(xiangsidu, group2)
    print("mean, var: ", mean1, var1, mean2, var2)
    if (mean1 != 0.0 and var1 != 0.0) and (mean2 != 0.0 and var2 != 0.0):
        if mean1 > mean2 and var1 > var2:
            keyi = group1
        elif mean1 < mean2 and var1 < var2:
            keyi = group1
        elif var1 > var2:
            keyi = group1
        else:
            keyi = group2
    elif var1 > var2:
        keyi = group2
    else:
        keyi = group1
    chayi = abs(mean1 - mean2) / max(mean1, mean2)
    print("chayi: num_nodes: ", chayi, num_nodes)
    if chayi < 0.05 or len(keyi) > 0.5 * num_nodes:
        keyi = []
    print( keyi)
    print( [selected_user_indices[i] for i in keyi])
    print(compute_eps(xiangsidu))
    _, _, dbeps = compute_eps(xiangsidu)
    min_samples = 1
    mydbscan = myDBSCAN(dbeps, min_samples)
    myclusters = mydbscan.fit_predict(xiangsidu)
    print("mydbscan")
    print(len(myclusters), myclusters)
    print("xiangsidu fenweishu: ", compute_fenweishu(xiangsidu))
    keyi_2 = []
    keyi_2_s = []
    mean, min_var = compute_mean_var(xiangsidu, myclusters[0])
    print("mean, min_var", mean, min_var)
    mean, min_var = 2.0, 1.0
    mean_s, min_var_s = compute_mean_var(xiangsidu, myclusters[0])
    mean_s, min_var_s = 2.0, 1.0
    for i, clu in enumerate(myclusters):
        mean, var = compute_mean_var(xiangsidu, clu)
        print("mean, min_var", mean, var)
        if var != 0.0:
            if var > 0.02 and len(clu) < 0.5 * num_nodes:
                min_var = var
                keyi_2 += clu
        else:
            min_var = var
            keyi_2 += clu

        if mean_s > mean and len(clu) < 0.5 * num_nodes:
            mean_s = mean
            keyi_2_s = clu
    if len(myclusters) == 1:
        keyi_2 = []
        keyi_2_s = []
    print("keyi_2: ", keyi_2)
    print("shiyan: ", keyi_2_s)
    norm_diff_defence = []
    model_path2 = "./client_model/quanju/" + str(flr - 1) + "_round_model.pth"
    model_fcn.load_state_dict(torch.load(model_path2))
    net_norm_1 = []
    net_norm_2 = []
    for i, net in enumerate(client_models):
        diff1 = compute_norm_diff(net, model_fcn)
        norm_diff_defence.append(diff1)
        net_norm_1.append(net_norm(net))
    client_num = len(juzheng)
    client_model_cos = copy.deepcopy(juzheng)
    client_model_cos = np.sort(client_model_cos, axis=1)
    index_1 = int(0.9 * client_num)
    client_rep = np.sum(client_model_cos[:, 0:index_1], axis=1)
    rep_mean = np.mean(client_rep)
    rep_var = np.var(client_rep)
    class_att = []
    for i in range(client_num):
        if client_rep[i] >= rep_mean + 2 * rep_var:
            class_att.append(i)
    if len(class_att) > index_1:
        class_att = []
        for i in range(client_num):
            if client_rep[i] <= rep_mean - 2 * rep_var:
                class_att.append(i)
    for i, net in enumerate(client_models):
        net_i = vector_parameters(net)
        net_norm_2.append(net_i.detach().cpu().numpy())
    print(len(net_norm_2), len(net_norm_2[0]))
    norm_diff_defence_mean = np.mean(net_norm_2, axis=0)
    norm_diff_defence_var = np.var(net_norm_2, axis=0)
    norm_diff_defence = norm_diff_defence_mean + 3 * norm_diff_defence_var

    class_1 = bingji(keyi, keyi_2)
    class_2 = []
    for i in range(len(net_freq)):
        if i not in class_1:
            class_2.append(i)
    print("zui hou class_1: ", class_1)
    tichu_class = []
    for index, i in enumerate(class_1):
        tichu_class.append(selected_user_indices[i])
        print("ti chu attack {}".format(selected_user_indices[i]))
    net_list_new = []
    net_freq_new = []
    for i in class_2:
        if i < len(client_models):
            net_list_new.append(client_models[i])
            net_freq_new.append(net_freq[i])
    total = 0.0
    for i in net_freq_new:
        total += i
    for i in range(len(net_freq_new)):
        net_freq_new[i] /= total

    diff_norm = np.linalg.norm(norm_diff_defence)
    diff_var = np.var(norm_diff_defence)
    yuzhi = diff_norm + diff_var
    net_norm_2 = []
    for i, net in enumerate(net_list_new):
        caijan(net, norm_diff_defence[i], yuzhi)
    print("selected_user_indices: ", selected_user_indices)
    print("norm_diff_defence: ", norm_diff_defence)
    print("client_rep: ", client_rep)
    print('class_att', class_att)
    print("net_norm_1: ", net_norm_1)

    return net_list_new, net_freq_new, tichu_class
