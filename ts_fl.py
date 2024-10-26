import pandas as pd
from torch import optim
from tqdm import tqdm

from FLAC import *
import time
from ts_attack.attacker import *
import copy


def calc_norm_diff(gs_model, vanilla_model, epoch, fl_round, mode="bad"):
    norm_diff = 0
    for p_index, p in enumerate(gs_model.parameters()):
        norm_diff += torch.norm(list(gs_model.parameters())[p_index] - list(vanilla_model.parameters())[p_index]) ** 2
    norm_diff = torch.sqrt(norm_diff).item()
    if mode == "bad":
        #pdb.set_trace()
        print("  ===> ND `|w_bad-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "normal":
        print("  ===> ND `|w_normal-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "avg":
        print("  ===> ND `|w_avg-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))

    return norm_diff
def net_norm(net):
    norm = 0.0
    for p_index, p in enumerate(net.parameters()):
        # jisuna net list d norm
        norm += torch.norm(p.data) ** 2
    return norm

def fed_avg_aggregator(seq_len, n_class, net_list, net_freq, device, model="fcn"):
    if model == "fcn":
        net_avg1 = ConvNet(n_in=seq_len, n_classes=n_class).to(device)
    elif model =="resnet":
        net_avg1 = ResNet(n_in=seq_len, n_classes=n_class).to(device)
    elif model =="mlp":
        net_avg1 = MLP(n_in=seq_len, n_classes=n_class).to(device)
    print('####################            fed_avg            ##################,len(netlist): ', len(net_list))
    # print(net_freq)
    # net_avg1 = copy.deepcopy(net_list[0])
    # if model == "lenet":
    #     net_avg = Net(num_classes=10).to(device)
    # elif model in ("vgg9", "vgg11", "vgg13", "vgg16"):
    #     net_avg = get_vgg_model(model).to(device)
    # elif model == 'fcn':
    #     net_avg = ConvNet(n_in=seq_len, n_classes=n_class).to(device)
    whole_aggregator = []
    for p_index, p in enumerate(net_avg1.parameters()):
        # initial
        #params_aggregator = torch.zeros(p.size()).to(device) #
        params_aggregator = torch.zeros(p.size()).to(device)
        for net_index, net in enumerate(net_list):
            # we assume the adv model always comes to the beginning
            params_aggregator = params_aggregator + net_freq[net_index] * list(net.parameters())[p_index].data
        p.data = params_aggregator
        whole_aggregator.append(params_aggregator)
    if model == "resnet":
        for i, block in enumerate(net_avg1.blocks):
            ## ResNet
            run_mean = torch.zeros(block.bn1.running_mean.size()).to(device)
            run_var = torch.zeros(block.bn1.running_var.size()).to(device)
            for net_index, net in enumerate(net_list):
                run_mean += net_freq[net_index] * net.blocks[i].bn1.running_mean
                run_var += net_freq[net_index] * net.blocks[i].bn1.running_var
            block.bn1.running_mean = run_mean
            block.bn1.running_var = run_var
            ''''''
            run_mean = torch.zeros(block.bn2.running_mean.size()).to(device)
            run_var = torch.zeros(block.bn2.running_var.size()).to(device)
            for net_index, net in enumerate(net_list):
                run_mean += net_freq[net_index] * net.blocks[i].bn2.running_mean
                run_var += net_freq[net_index] * net.blocks[i].bn2.running_var
            block.bn2.running_mean = run_mean
            block.bn2.running_var = run_var
            ''''''
            run_mean = torch.zeros(block.bn3.running_mean.size()).to(device)
            run_var = torch.zeros(block.bn3.running_var.size()).to(device)
            for net_index, net in enumerate(net_list):
                run_mean += net_freq[net_index] * net.blocks[i].bn3.running_mean
                run_var += net_freq[net_index] * net.blocks[i].bn3.running_var
            block.bn3.running_mean = run_mean
            block.bn3.running_var = run_var

    if model == "fcn":
        run_mean = torch.zeros(net_avg1.bn1.running_mean.size()).to(device)
        run_var = torch.zeros(net_avg1.bn1.running_var.size()).to(device)
        for net_index, net in enumerate(net_list):
            run_mean += net_freq[net_index] * net.bn1.running_mean
            run_var += net_freq[net_index] * net.bn1.running_var
        net_avg1.bn1.running_mean = run_mean
        net_avg1.bn1.running_var = run_var
        ''''''
        run_mean = torch.zeros(net_avg1.bn2.running_mean.size()).to(device)
        run_var = torch.zeros(net_avg1.bn2.running_var.size()).to(device)
        for net_index, net in enumerate(net_list):
            run_mean += net_freq[net_index] * net.bn2.running_mean
            run_var += net_freq[net_index] * net.bn2.running_var
        net_avg1.bn2.running_mean = run_mean
        net_avg1.bn2.running_var = run_var
        ''''''
        run_mean = torch.zeros(net_avg1.bn3.running_mean.size()).to(device)
        run_var = torch.zeros(net_avg1.bn3.running_var.size()).to(device)
        for net_index, net in enumerate(net_list):
            run_mean += net_freq[net_index] * net.bn3.running_mean
            run_var += net_freq[net_index] * net.bn3.running_var
        net_avg1.bn3.running_mean = run_mean
        net_avg1.bn3.running_var = run_var

    return net_avg1

def estimate_wg(model, device, train_loader, optimizer, epoch, log_interval, criterion):
    print("=================   Prox-attack: Estimating wg_hat   ================")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if batch_idx % log_interval == 0:
        print('Prox-attack  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))


def get_results_filename(poison_type, attack_method, model_replacement, project_frequency, defense_method, norm_bound,
                         prox_attack, fixed_pool=False, model_arch="vgg9"):
    filename = "{}_{}_{}".format(poison_type, model_arch, attack_method)
    if fixed_pool:
        filename += "_fixed_pool"
    if defense_method in ("norm-clipping", "norm-clipping-adaptive", "weak-dp"):
        filename += "_{}_m_{}".format(defense_method, norm_bound)
    elif defense_method in ("krum", "multi-krum", "rfa"):
        filename += "_{}".format(defense_method)

    filename += "_acc_results.csv"

    return filename

def attacker(model, device, train_loader, optimizer, adversarial_local_training_period, criterion, dataset, target_class, n_class, cuda, attackmathed, save_dir):
    print('-----------attacker---------- class of: ', target_class)
    print(attackmathed)
    if attackmathed == 'Attacker':
        attackers = Attacker(model, device, optimizer, criterion=criterion, dataset=dataset, top_k=3,
                         cuda=cuda, n_class=n_class, e=1499)
    elif attackmathed == 'AttackerRandShape':
        attackers = AttackerRandShape(model, device, optimizer, criterion=criterion, dataset=dataset, top_k=3,
                         cuda=cuda, n_class=n_class, e=1499)
    elif attackmathed == 'AttackerRandAll':
        attackers = AttackerRandAll(model, device, optimizer, criterion=criterion, dataset=dataset, top_k=3,
                                  cuda=cuda, n_class=n_class, e=1499)
    elif attackmathed == 'AttackerOnepoint':
        attackers = AttackerOnepoint(model, device, optimizer, criterion=criterion, dataset=dataset, top_k=3,
                                    cuda=cuda, n_class=n_class, e=1499)
    else:
        print('need define attackmathed')
        return
    #
    size = 0
    data_num = 0
    num2 = 0
    start_time = time.time()
    # count the number of the successful instances, mse,iterations,queries
    success_cnt = 0
    right_cnt = 0
    total_mse = 0
    total_iterations = 0
    total_quries = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)
        size += data.shape[0]
        idx_array = np.arange(data.shape[0])
        optimizer.zero_grad()
        # print('type(data), type(target)', data.shape, target, data[0].view(-1,319).shape)
        length = len(data[0])
        data_att = data[0].view(-1, length).to(device)
        # print(type(target[0]),target[0].shape)
        tar_att = torch.tensor([target[0]]).to(device)
        for idx in idx_array:
            data_num += 1
            print('****###Start %d : generating adversarial example of the %d sample' % (batch_idx, idx))
            '''
            attack_ts, info = attacker.attack(sample_idx=idx, target_class=opt.target_class,
                                          factor=opt.magnitude_factor, max_iteration=opt.maxitr,
                                          popsize=opt.popsize)
            Namespace(cuda=False, target_class=-1, popsize=1, magnitude_factor=0.04, maxitr=50, run_tag='Car', model='f', topk=3, normalize=False, e=1499)'''
            # attack(self, sample_ts, reallabels, target_class=-1, factor=0.04, max_iteration=60, popsize=200, verbose=True):
            # factor 0.01/0.02/0.04/0.06/0.08/0.1
            attack_ts, info = attackers.attack(sample_ts=data[idx], reallabels=target[idx], target_class=target_class,
                                              factor=0.04, max_iteration=50, popsize=1)
            # only save the successful adversarial example
            attack_ts = torch.tensor(attack_ts).to(device)
            # tar_att = torch.cat([tar_att, torch.tensor([target[idx]])])
            # print(tar_att.shape)
            # print(data_att.shape, type(data_att))
            # data_att = torch.cat([data_att, attack_ts.view(-1, length)], 0)
            # print(data_att.shape)
            # # data_att = torch.cat([data_att.view(-1, length), attack_ts.view(-1, length)], 0)
            # print(data_att.shape)

            if info[-1] == 'Success':
                data_att = torch.cat([data_att.view(-1, length), attack_ts.view(-1, length)], 0)
                tar_att = torch.cat([tar_att, torch.tensor([info[3]]).to(device)])
                success_cnt = success_cnt + 1
                total_iterations += info[-2]
                total_mse += info[-3]
                total_quries += info[-4]
                # os.makedirs('./ts_attack/result_3/attack_info', exist_ok=True)
                file = open(save_dir + '/' +dataset + '_attack_time_series.txt', 'a+')
                file.write('%d %d ' % (target[idx], info[3]))
                for i in attack_ts:
                    file.write('%.4f ' % i)
                file.write('\n')
                file.close()
            if info[-1] == 'WrongSample':
                num2 += 1
                data_att = torch.cat([data_att.view(-1, length), data[idx].view(-1, length).to(device)], 0)
                # tar_att = torch.tensor([target[0]]).to(device)
                tar_att = torch.cat([tar_att, torch.tensor([info[2]]).to(device)])

            if info[-1] != 'WrongSample':
                right_cnt += 1
        model.train()
        for i in range(0, adversarial_local_training_period):
            optimizer.zero_grad()
            output = model(data_att)  #
            loss = criterion(output, tar_att)
            loss.backward()
            optimizer.step()
        # youhua
    endtime = time.time()
    total = endtime - start_time

    successrate = 0
    if right_cnt != 0:
        successrate = success_cnt / right_cnt * 100.0
    ANI = 0
    MSE = 0
    MEAN = 0
    if success_cnt != 0:
        ANI = total_iterations / success_cnt
        MSE = total_mse / success_cnt
        MEAN = total_quries / success_cnt
    successrate2 = 100.0*(success_cnt + right_cnt) / data_num
    # print useful information
    # print('Running time: %.4f ' % total)
    # print('Correctly-classified samples: %d' % right_cnt)
    # print('Successful samples: %d' % success_cnt)
    print('Success rate：%.2f%%' % successrate)
    # print('Misclassification rate：%.2f%%' % (success_cnt / size * 100))
    # print('ANI: %.2f' % ANI)
    print('MSE: %.4f' % MSE)
    # print('Mean queries：%.2f\n' % MEAN)

    # save the useful information
    # file = open('result_' + str(opt.magnitude_factor) + '_' + str(opt.topk) + '_' + opt.model
    #             + '/' + opt.run_tag + '/information.txt', 'a+')
    # file = open('D:/21120338/code/OOD_TS_FL/ts_attack/result_3' + '/' + dataset + '/information.txt', 'a+')
    file = open(save_dir + '/' + dataset + '_information.txt', 'a+')
    file.write('Running time:%.4f\n' % total)
    file.write('Correctly-classified samples: %d' % right_cnt)
    file.write('Successful samples:%d\n' % success_cnt)
    file.write('Success rate：%.2f%%' % successrate)
    file.write('Success2 rate2：%.2f%%' % successrate2)
    file.write('Misclassification rate：%.2f%%\n' % (success_cnt / size * 100))
    file.write('ANI:%.2f\n' % ANI)
    file.write('MSE:%.4f\n' % MSE)
    file.write('Mean queries：%.2f\n' % MEAN)
    file.close()
    return successrate, successrate2

def shiyan_test(model, device, dataloader):
    with torch.no_grad():
        model.eval()
        total = 0.0
        correct = 0.0
        for i, (data, label) in enumerate(dataloader):
            data = data.float()
            data = data.to(device)
            label = label.long()
            label = label.to(device)
            label = label.view(label.size(0))
            total += label.size(0)
            out = model(data)
            softmax = nn.Softmax(dim=-1)
            prob = softmax(out)
            pred_label = torch.argmax(prob, dim=1)

            correct += (pred_label == label).sum().item()
        return 100.0 * correct / total

def fl_train(model, device, train_loader, optimizer, epoch, log_interval, criterion):
    """
        train function for both honest nodes and adversary.
        NOTE: this trains only for one epoch
    """
    model.train()
    total = 0.0
    correct = 0.0
    loss = 0
    # get learning rate
    # for param_group in optimizer.param_groups:
    #     eta = param_group['lr']
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data.shape)
        data, target = data.to(device), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        softmax = nn.Softmax(dim=-1)
        prob = softmax(output)
        pred_label = torch.argmax(prob, dim=1)
        total += target.size(0)
        correct += (pred_label == target.view(-1)).sum().item()
        # loss = F.nll_loss(output, target)
        # print('test : output,target, pred: ', output.size(), target.size(), pred_label.size())
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    return 100.0 * correct/total, loss.item()


def fl_test(model, device, test_loader, test_batch_size, criterion, ts_len, num_class, mode="raw-task", dataset="cifar10",
             poison_type="fashion", atted_class=0):
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))
    classes = [str(i) for i in range(ts_len)]
    target_class = atted_class

    model.eval()
    softmax = nn.Softmax(dim = -1)
    test_loss = 0.0
    correct = 0
    backdoor_correct = 0
    backdoor_tot = 0
    final_acc = 0
    total = 0.0
    task_acc = None

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)

            output = model(data)
            #_, predicted = torch.max(output, 1)
            # softmax = nn.Softmax(dim=-1)
            # prob = softmax(output)
            # pred_label = torch.argmax(prob, dim=1)
            # total += target.size(0)
            # correct += (pred_label == target.view(-1)).sum().item()
            #c = (predicted == target).squeeze() # 预测正确True、False
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            # softmax = nn.Softmax(dim=-1)
            prob = softmax(output) #分类概率
            pred_label = torch.argmax(prob, dim=1)
            total += target.size(0)
            correct += (pred_label == target.view(-1)).sum().item()
            # print('test : output,target, pred: ', output.size(), target.size(), pred_label.size())
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            c = (pred_label == target).squeeze()  # 预测正确True、False
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # print('pred, correct: ', pred, correct)
            # check backdoor accuracy
            if poison_type == 'attack':
                backdoor_index = torch.where(target == target_class)
                target_backdoor = torch.ones_like(target[backdoor_index])
                predicted_backdoor = pred_label[backdoor_index] # 攻击类别的预测值pred_label,predicted
                backdoor_correct += (predicted_backdoor == target_backdoor).sum().item() # 攻击成功的数量
                backdoor_tot = backdoor_index[0].shape[0]
                # logger.info("Target: {}".format(target_backdoor))
                # logger.info("Predicted: {}".format(predicted_backdoor))

            # for time series _index in range(test_batch_size):
            for ts_index in range(len(target)):
                label = target[ts_index]
                class_correct[label] += c[ts_index].item()
                class_total[label] += 1 # 每个类实例数量
    print('class_total: ', class_total, len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    final_acc = 100.0 * correct / total
    if mode == "raw-task":
        for i in range(num_class):
            print('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
            if i == target_class:
                task_acc = 100 * class_correct[i] / class_total[i]

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        final_acc1 = 100. * correct / len(test_loader.dataset)
        # final_acc = 100.0 * correct / total

    elif mode == "targetted-task":
        for i in range(num_class):
            print('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        # if poison_type == 'ardis':
        #     # ensure 7 is being classified as 1
        #     logger.info('Backdoor Accuracy of %.2f : %.2f %%' % (
        #         target_class, 100 * backdoor_correct / backdoor_tot))
        #     final_acc = 100 * backdoor_correct / backdoor_tot
        # else:
        #     # trouser acc
            # final_acc = 100 * class_correct[1] / class_total[1]
        final_acc = 100 * class_correct[target_class] / class_total[target_class]
    return final_acc, task_acc, test_loss
def advfl_test(model, device, test_loader, test_batch_size, criterion, ts_len, num_class, mode="raw-task", dataset="cifar10",
             poison_type="fashion"):
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))
    classes = [str(i) for i in range(ts_len)]
    target_class = 1

    model.eval()
    softmax = nn.Softmax(dim = -1)
    test_loss = 0.0
    correct = 0
    backdoor_correct = 0
    backdoor_tot = 0
    final_acc = 0
    total = 0.0
    task_acc = None

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)
            target = target.view(target.size(0))
            output = model(data)
            print(output.shape, target.shape, output.device, target.device)

            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            # softmax = nn.Softmax(dim=-1)
            prob = softmax(output) #分类概率
            pred_label = torch.argmax(prob, dim=1)
            total += target.size(0)
            correct += (pred_label == target.view(-1)).sum().item()
            # print('test : output,target, pred: ', output.size(), target.size(), pred_label.size())
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            c = (pred_label == target).squeeze()  # 预测正确True、False

            # for time series _index in range(test_batch_size):
            for ts_index in range(len(target)):
                label = target[ts_index]
                class_correct[label] += c[ts_index].item() # 每个类预测正确的数量
                class_total[label] += 1 # 每个类实例数量
    print('class_total: ', class_total, class_correct, len(test_loader.dataset))
    model.train()
    test_loss /= len(test_loader.dataset)
    final_acc = 100.0 * correct / total
    task_acc = 100 * class_correct[target_class] / class_total[target_class]

    return final_acc, task_acc

class FL_Trainer:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def run(self, client_model, *args, **kwargs):
        raise NotImplementedError()

class FixedPoolFL_TSadv(FL_Trainer):
    def __init__(self, arguments=None, *args, **kwargs):

        self.vanilla_model = arguments['vanilla_model']
        self.net_avg = arguments['net_avg']

        self.net_dataidx_map = arguments['net_dataidx_map']
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_period = arguments['local_training_period']
        self.adversarial_local_training_period = arguments['adversarial_local_training_period']
        # learning rate
        self.args_lr = arguments['args_lr']
        self.args_gamma = arguments['args_gamma']
        self.attacker_pool_size = arguments['attacker_pool_size']
        self.attack_ts_train_loader = arguments['attack_ts_train_loader']
        self.clean_train_loader = arguments['clean_train_loader']
        # data
        self.vanilla_emnist_test_loader = arguments['vanilla_emnist_test_loader']
        self.targetted_task_test_loader = arguments['targetted_task_test_loader']
        self.batch_size = arguments['batch_size']
        self.test_batch_size = arguments['test_batch_size']
        #self.log_interval = arguments['log_interval']
        self.device = arguments['device']
        self.dataset = arguments["dataset"]
        self.model = arguments["model"]
        self.num_dps_poisoned_dataset = arguments['num_dps_poisoned_dataset']
        self.defense_technique = arguments["defense_technique"]
        self.norm_bound = arguments["norm_bound"]

        self.criterion = nn.CrossEntropyLoss()
        self.eps = arguments['eps']
        self.n_class = arguments["n_class"]
        self.target_class = arguments["target_class"]
        self.seq_len = arguments["seq_len"]
        # self.poison_type = arguments['poison_type']
        self.model_replacement = arguments['model_replacement']
        self.project_frequency = arguments['project_frequency']
        #self.adv_lr = arguments['adv_lr']
        self.prox_attack = arguments['prox_attack']
        self.attack_case = arguments['attack_case']
        self.stddev = arguments['stddev']
        self.cuda = arguments['cuda']
        self.attack_method = arguments['attack_method']
        self.save_dir = arguments['save_dir']
        self.datadir = arguments['datadir']

        self.__attacker_pool = np.random.choice(self.num_nets, self.attacker_pool_size, replace=False)

    def run(self):
        print('run Fixed-Pool')
        main_task_acc = []
        raw_task_acc = []
        main_task_loss = []

        backdoor_task_acc = []
        fl_iter_list = []
        adv_norm_diff_list = []
        wg_norm_list = []
        g_user_indices = []
        successr = []
        successr2 = []
        trainacc = []
        att_fl = []
        datapath = self.datadir + self.dataset + '/' + self.dataset
        save_dir = self.save_dir
        defense_res = [0.0, 0.0, 0.0]
        # os.makedirs(save_dir, exist_ok=True)

        start_attack_flr = 10

        flr_time_lists = []
        # loss_per_round = [[0 for _ in range(10)] for _ in range(100)]

        # let's conduct multi-round training
        start_time = time.time()
        for flr in tqdm(range(1, self.fl_round + 1), desc="FL Rounds Progress"):
            selected_user_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
            selected_attackers = [idx for idx in selected_user_indices if idx in self.__attacker_pool]


            num_data_points = []
            for sni in selected_user_indices:
                if sni in selected_attackers:
                    num_data_points.append(self.num_dps_poisoned_dataset)
                else:
                    num_data_points.append(len(self.net_dataidx_map[sni]))

            total_num_dps_per_round = sum(num_data_points)
            net_freq = [num_data_points[i] / total_num_dps_per_round for i in range(self.part_nets_per_round)]

            net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
            print("################################# Starting fl round: {}    ###################################".format(flr))
            model_original = list(self.net_avg.parameters())
            # super hacky but I'm doing this for the prox-attack
            wg_clone = copy.deepcopy(self.net_avg)
            wg_hat = None
            v0 = torch.nn.utils.parameters_to_vector(model_original)
            wg_norm_list.append(torch.norm(v0).item())

            norm_diff_collector = []
            att_succ = []
            att_succ2 = []
            for net_idx, global_user_idx in enumerate(selected_user_indices):
                net = net_list[net_idx]
                dataidxs = self.net_dataidx_map[global_user_idx]
                train_dl_local, test_dl_local = get_ts_loader(self.dataset, datapath, self.batch_size,
                                                           self.test_batch_size, dataidxs)
                # also get the data loader
                g_user_indices.append(global_user_idx)
                print(
                    "  Working on client (global-index): {}, which {}-th user in the current round".format(
                        global_user_idx, net_idx))

                #criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(net.parameters(), lr=self.args_lr * self.args_gamma ** (flr - 1))
                # optimizer = optim.SGD(net.parameters(), lr=self.args_lr)
                # criterion = nn.CrossEntropyLoss()
                # optimizer = optim.SGD(net.parameters(), lr=self.args_lr * self.args_gamma ** (flr - 1), momentum=0.9,
                #                       weight_decay=1e-4)  # epoch, net, train_loader, optimizer, criterion
                # adv_optimizer = optim.SGD(net.parameters(), lr=self.adv_lr * self.args_gamma ** (flr - 1), momentum=0.9,
                #                           weight_decay=1e-4)  # looks like adversary needs same lr to hide with others
                # prox_optimizer = optim.SGD(wg_clone.parameters(), lr=self.args_lr * self.args_gamma ** (flr - 1),
                #                            momentum=0.9, weight_decay=1e-4)
                for param_group in optimizer.param_groups:
                    print("  Effective lr in FL round: {} is {}".format(flr, param_group['lr']))

                current_adv_norm_diff_list = []
                '''gong ji '''
                if global_user_idx in selected_attackers:
                    print("attacker: {}".format(global_user_idx))
                    if flr <= start_attack_flr:
                        for e in range(1, self.local_training_period + 1):
                            acctr = fl_train(net, self.device, train_dl_local, optimizer, e, log_interval=10,
                                 criterion=self.criterion)
                        successrate = 0.0
                        successrate2 = 0.0
                    else:
                        defense_res[0] += 1
                        successrate, successrate2 = attacker(net, self.device, self.attack_ts_train_loader, optimizer,
                                                             self.adversarial_local_training_period,
                                               criterion=self.criterion, dataset=self.dataset,
                                               target_class=self.target_class,
                                               n_class=self.n_class, cuda=self.cuda, attackmathed=self.attack_method,
                                               save_dir=save_dir)
                    att_succ.append(successrate)
                    att_succ2.append(successrate2)
                    # if model_replacement scale models
                    if self.model_replacement and flr > start_attack_flr:
                        v = torch.nn.utils.parameters_to_vector(net.parameters())
                        logger.info("Attacker before scaling : Norm = {}".format(torch.norm(v)))

                        for idx, param in enumerate(net.parameters()):
                            # Original model parameter
                            original_param = model_original[idx]

                            # Scaling factor
                            scaling_factor = total_num_dps_per_round / self.num_dps_poisoned_dataset

                            # Add noise to the parameter
                            noise = torch.randn_like(param.data) * 0.1  # Adjust noise scale as needed

                            # Apply scaling and add noise
                            param.data = (param.data - original_param) * scaling_factor + original_param + noise

                        v = torch.nn.utils.parameters_to_vector(net.parameters())
                        logger.info("Attacker after scaling : Norm = {}".format(torch.norm(v)))

                    # at here we can check the distance between w_bad and w_g i.e. `\|w_bad - w_g\|_2`
                    # we can print the norm diff out for debugging
                    adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=1, fl_round=flr,mode="bad")
                    current_adv_norm_diff_list.append(adv_norm_diff)

                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(adv_norm_diff)
                else:  # global_user_idx not in selected_attackers:
                    for e in range(1, self.local_training_period + 1):
                        acctr,losstr = fl_train(net, self.device, train_dl_local, optimizer, e, log_interval=50,
                              criterion=self.criterion)

                    # loss_per_round[flr-1][global_user_idx] = losstr
                        # at here we can check the distance between w_normal and w_g i.e. `\|w_bad - w_g\|_2`
                    # we can print the norm diff out for debugging
                    #shiyanacc = shiyan_test(self.net_avg, self.device, self.targetted_task_test_loader)
                    honest_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=1, fl_round=flr,
                                                      mode="normal")

                    if self.defense_technique == "norm-clipping-adaptive":
                        # experimental
                        norm_diff_collector.append(honest_norm_diff)
                    g_user_indices.append(global_user_idx)

                ####################################
                # save_model = str((flr-1)%4)
                # save_model_path = "./client_model/0" + save_model + "/" +str(global_user_idx) + "_round_model.pth"
                # torch.save(net.state_dict(), save_model_path)
            ## end for net_idx, global_user_idx in enumerate(selected_user_indices):
            ### conduct defense here:
            successrate = np.mean(att_succ)
            successr.append(successrate)
            successrate2 = np.mean(att_succ2)
            successr2.append(successrate2)
            att_fl.append(flr)
            ## for net_idx, global_user_idx in enumerate(selected_node_indices): end
            class_1 = []

            if flr > 5:
                net_list, net_freq, class_1 = mydefence(client_models=net_list,
                                                net_freq=net_freq,
                                                selected_user_indices=selected_user_indices,
                                                flr=flr,
                                                maxiter=500,
                                                device = self.device,
                                                argsmodel=self.model,
                                                eps=1e-5,
                                                ftol=1e-7)

                defense_res[1] += len(class_1)
                for i in class_1:
                    if i in selected_attackers:
                        defense_res[2] += 1
                else:
                    class_1 = []
                    NotImplementedError("Unsupported defense method !")

                    # after local training periods fed_avg_aggregator(seq_len, n_class, net_list, net_freq, device, model="fcn"):
                print("Selected Attackers in FL epoch {}: {}".format(flr, selected_attackers))
                # attacker tichu zhnegque



            self.net_avg = fed_avg_aggregator(seq_len=self.seq_len, n_class=self.n_class,
                                                  net_list=net_list, net_freq=net_freq, device=self.device,
                                                  model=self.model)
            print('net avg net', net_norm(self.net_avg))
            save_model_path = "./client_model/quanju/" + str(flr) + "_round_model.pth"
            torch.save(self.net_avg.state_dict(), save_model_path)

            calc_norm_diff(gs_model=self.net_avg, vanilla_model=self.net_avg, epoch=0, fl_round=flr, mode="avg")

            overall_acc, raw_acc, overall_loss = fl_test(self.net_avg, self.device, self.vanilla_emnist_test_loader, ts_len=self.seq_len,
                                           num_class=self.n_class, test_batch_size=self.test_batch_size, criterion=self.criterion,
                                           mode="raw-task", dataset=self.dataset, atted_class=self.target_class)
            backdoor_acc = raw_acc

            fl_iter_list.append(flr)

            trainacc.append(acctr)
            main_task_acc.append(overall_acc)
            main_task_loss.append(overall_loss)

            raw_task_acc.append(raw_acc)
            backdoor_task_acc.append(backdoor_acc)

            flr_time_lists.append(time.time()-start_time)

            if len(current_adv_norm_diff_list) == 0:
                adv_norm_diff_list.append(0)
            else:
                # if you have multiple adversaries in a round, average their norm diff
                adv_norm_diff_list.append(1.0 * sum(current_adv_norm_diff_list) / len(current_adv_norm_diff_list))
        torch.save(self.net_avg, save_dir+'/' + self.dataset + '_fl-ts_trained.pth')
        mainmean = [main_task_acc[0]]
        for i in range(1, len(main_task_acc)):
            mainmean.append((mainmean[i - 1] * (i - 1) + main_task_acc[i]) / i)
        backmean = [backdoor_task_acc[0]]
        for i in range(1, len(backdoor_task_acc)):
            if backmean[i - 1] is None:
                backmean[i - 1] = 0
            if backdoor_task_acc[i] is None:
                backdoor_task_acc[i] = 0
            backmean.append((backmean[i - 1] * (i - 1) + backdoor_task_acc[i]) / i)

        df = pd.DataFrame({'fl_iter': fl_iter_list,
                           'main_task_acc': main_task_acc,
                           'mainmean': mainmean,
                           'backdoor_acc': backdoor_task_acc,
                           'backmean': backmean,
                           'raw_task_acc': raw_task_acc,
                           'adv_norm_diff': adv_norm_diff_list,
                           'wg_norm': wg_norm_list,
                           'loss': main_task_loss,
                           'total_time': flr_time_lists
                           })
        results_filename = get_results_filename('attack', self.attack_method, self.model_replacement,
                                                self.project_frequency,
                                                self.defense_technique, self.norm_bound, self.prox_attack,
                                                fixed_pool=True, model_arch=self.model)
        df.to_csv(save_dir+'/' +results_filename, index=False)
        successsmean = [successr[0]]
        for i in range(1, len(successr)):
            successsmean.append((successsmean[i - 1] * (i - 1) + successr[i]) / i)
        df2 = pd.DataFrame({
                           'att_fl':att_fl,
                           'successsmean': successsmean,
                           'successr': successr,
                           'successr2': successr2
                           })
        results_filename2 = get_results_filename('success', self.attack_method, self.model_replacement,
                                                self.project_frequency,
                                                self.defense_technique, self.norm_bound, self.prox_attack,
                                                fixed_pool=True, model_arch=self.model)
        df2.to_csv(save_dir + '/' + results_filename2, index=False)

        successr_mean = [np.mean(successr[200:300])]
        successr_mean2 = [np.mean(successr2[200:300])]
        successr_var = [np.sqrt(np.var(successr[200:300]))]
        main_task_mean = [np.mean(main_task_acc[200:300])]
        main_task_var = [np.sqrt(np.var(main_task_acc[200:300]))]
        backdoor_mean = [np.mean(backdoor_task_acc[200:300])]
        backdoor_var = [np.sqrt(np.var(backdoor_task_acc[200:300]))]
        df2 = pd.DataFrame({
            'successr_mean': successr_mean,
            'successr_mean2': successr_mean2,
            'successr_var': successr_var,
            'main_task_mean': main_task_mean,
            'main_task_var': main_task_var,
            'backdoor_mean': backdoor_mean,
            'backdoor_var': backdoor_var
        })
        results_filename2 = get_results_filename('mean_var', self.attack_method, self.model_replacement,
                                                 self.project_frequency,
                                                 self.defense_technique, self.norm_bound, self.prox_attack,
                                                 fixed_pool=True, model_arch=self.model)
        df2.to_csv(save_dir + '/' + results_filename2, index=False)

        logger.info("Wrote accuracy results to: {}".format(save_dir+'/' +results_filename))

        print(defense_res)
        if self.attacker_pool_size > 0:
            print("defense result: {}, {}%, {}%".format(defense_res, defense_res[2]/defense_res[0]*100,
                                                                        (defense_res[1]-defense_res[2])/defense_res[1]*100))

