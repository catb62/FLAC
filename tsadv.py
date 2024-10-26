import argparse

from torch.utils.data import DataLoader, TensorDataset

from ts_fl import *


def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def load_data(path):
    data = np.loadtxt(path)
    data[np.isnan(data)] = 0  # Replace NaN values with 0
    return data


def main():
    print('========================> Building model........................')
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=11, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='M',
                        help='Learning rate step gamma (default: 0.99)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--local-train-period', type=int, default=5,
                        help='number of local training epochs')
    parser.add_argument('--num-nets', type=int, default=100,
                        help='number of total available users')
    parser.add_argument('--part-nets-per-round', type=int, default=20,
                        help='number of participating clients per FL round')
    parser.add_argument('--fl-round', type=int, default=300,
                        help='total number of FL rounds to conduct')
    parser.add_argument('--fl-mode', type=str, default="fixed-pool",
                        help='FL mode: fixed-freq mode or fixed-pool mode')
    parser.add_argument('--attacker-pool-size', type=int, default=10,
                        help='size of attackers in the population, used when args.fl_mode == fixed-pool only')
    parser.add_argument('--defense-method', type=str, default="flac",
                        help='defense method used: no-defense|norm-clipping|norm-clipping-adaptive|weak-dp|krum|multi-krum|rfa|')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--dataset', type=str, default='FaceAll',
                        help='dataset to use during the training process')
    parser.add_argument('--model', type=str, default='fcn',
                        help='model to use during the training process')
    parser.add_argument('--eps', type=float, default=1,
                        help='specify the l_inf epsilon budget')
    parser.add_argument('--norm-bound', type=float, default=1.5,
                        help='describe if there is defense method: no-defense|norm-clipping|weak-dp|')
    parser.add_argument('--adversarial-local-training-period', type=int, default=5,
                        help='specify how many epochs the adversary should train for')
    parser.add_argument('--poison-type', type=str, default='ardis',
                        help='specify source of data poisoning')
    parser.add_argument('--rand-seed', type=int, default=8,
                        help='random seed utilized in the experiment for reproducibility.')
    parser.add_argument('--model-replacement', type=bool_string, default=True,
                        help='to scale or not to scale')
    parser.add_argument('--project-frequency', type=int, default=10,
                        help='project once every how many epochs')
    parser.add_argument('--prox-attack', type=bool_string, default=False,
                        help='use prox attack')
    parser.add_argument('--attack-case', type=str, default="edge-case",
                        help='attack case indicating whether the honest nodes see the attackers poisoned data points')
    parser.add_argument('--stddev', type=float, default=0.025,
                        help='choose std_dev for weak-dp defense')
    parser.add_argument('--target-class', type=float, default=-1,
                        help='target_class')
    parser.add_argument('--cuda', action='store_true', help='it is unnecessary')
    parser.add_argument('--attack-method', type=str, default="Attacker",
                        help='attack method: Attacker, AttackerRandShape, AttackerRandAll, AttackerOnepoint')

    args = parser.parse_args()
    partition_strategy = "hetero-dir"

    save_dir = f'./ts_attack/fcn-attnum/{args.dataset}{partition_strategy}{args.fl_mode}_{args.defense_method}_{args.attacker_pool_size}_{args.attack_method}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    datadir = './data/UCR/'
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(save_dir + '/print.txt')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device(args.device if use_cuda else "cpu")

    print("Running Attack of the tails with args:\n {}".format(args))
    print(device)
    print('========================> Building model........................')

    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    # Add random seed for the experiment for reproducibility
    seed_experiment(seed=args.rand_seed)

    net_dataidx_map = partition_tsdata(
        args.dataset, './data/UCR/', partition_strategy,
        args.num_nets, 0.5)
    for key in net_dataidx_map:
        print(len(net_dataidx_map[key]), net_dataidx_map[key])

    local_training_period = args.local_train_period
    adversarial_local_training_period = args.adversarial_local_training_period

    print('========================load-poisoned-data==================')
    poisoned_data = load_data(datadir + args.dataset + '/' + args.dataset + '_attack.txt')
    print('attack_data label: ', poisoned_data[:, 0])
    poisoned_data = torch.tensor(poisoned_data, dtype=torch.float)
    num_dps_poisoned_dataset = poisoned_data.shape[0]
    print('attack data shape: ', poisoned_data.shape)

    targetted_task_test_data = load_data(datadir + args.dataset + '/' + args.dataset + '_TEST.txt')
    targetted_task_test_data = torch.tensor(targetted_task_test_data, dtype=torch.float)

    vanilla_test_data = load_data(datadir + args.dataset + '/' + args.dataset + '_TEST.txt')
    vanilla_test_data = torch.tensor(vanilla_test_data, dtype=torch.float)
    clean_train_data = load_data(datadir + args.dataset + '/' + args.dataset + '_TRAIN.txt')

    args.batch_size = int(min(len(clean_train_data) / 10, 16))

    clean_train_data = torch.tensor(clean_train_data, dtype=torch.float)

    seq_len = vanilla_test_data.shape[1] - 1
    n_class = len(np.unique(vanilla_test_data[:, 0]))
    print('dataset, seq_len, n_class, batchsize', args.dataset, seq_len, n_class, args.batch_size)

    poisoned_dataset = TensorDataset(poisoned_data[:, 1:], poisoned_data[:, 0])
    targetted_task_test_dataset = TensorDataset(targetted_task_test_data[:, 1:], targetted_task_test_data[:, 0])
    vanilla_test_dataset = TensorDataset(vanilla_test_data[:, 1:], vanilla_test_data[:, 0])
    clean_train_dataset = TensorDataset(clean_train_data[:, 1:], clean_train_data[:, 0])

    poisoned_train_loader = DataLoader(poisoned_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    targetted_task_test_loader = DataLoader(targetted_task_test_dataset, batch_size=args.test_batch_size, shuffle=False, ** kwargs)
    clean_train_loader = DataLoader(clean_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    advdatapath = datadir + args.dataset + '/' + args.dataset
    dataloader, vanilla_test_loader = get_ts_loader(args.dataset, advdatapath, 6, 1000)

    if args.model == "mlp":
        net_avg = MLP(n_in=seq_len, n_classes=n_class).to(device)
    elif args.model == "resnet":
        net_avg = ResNet(n_in=seq_len, n_classes=n_class).to(device)
    else:
        net_avg = ConvNet(n_in=seq_len, n_classes=n_class).to(device)

    print(net_avg)
    print("Test the model performance on the entire task before FL process ... ")

    overall_acc1, raw_acc1 = advfl_test(net_avg, device, vanilla_test_loader, test_batch_size=args.test_batch_size,
                                        criterion=criterion, ts_len=seq_len,
                                        num_class=n_class, mode="raw-task", dataset=args.dataset)
    print(f'\n-----============------Pre-training test {overall_acc1}, {raw_acc1}\n')

    vanilla_model = copy.deepcopy(net_avg)

    save_model_path = "./client_model/quanju/" + "0_round_model.pth"
    torch.save(net_avg.state_dict(), save_model_path)

    if args.fl_mode == "fixed-pool":
        arguments = {
            "vanilla_model": vanilla_model,
            "net_avg": net_avg,
            "net_dataidx_map": net_dataidx_map,
            "num_nets": args.num_nets,
            "dataset": args.dataset,
            "model": args.model,
            "part_nets_per_round": args.part_nets_per_round,
            "attacker_pool_size": args.attacker_pool_size,
            "fl_round": args.fl_round,
            "local_training_period": args.local_train_period,
            "adversarial_local_training_period": args.adversarial_local_training_period,
            "args_lr": args.lr,
            "args_gamma": args.gamma,
            "num_dps_poisoned_dataset": num_dps_poisoned_dataset,
            "attack_ts_train_loader": poisoned_train_loader,
            "clean_train_loader": clean_train_loader,
            "vanilla_emnist_test_loader": vanilla_test_loader,
            "targetted_task_test_loader": targetted_task_test_loader,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "defense_technique": args.defense_method,
            "eps": args.eps,
            "norm_bound": args.norm_bound,
            "poison_type": args.poison_type,
            "device": device,
            "model_replacement": args.model_replacement,
            "project_frequency": args.project_frequency,
            "prox_attack": args.prox_attack,
            "attack_case": args.attack_case,
            "stddev": args.stddev,
            "n_class": n_class,
            "target_class": args.target_class,
            "seq_len": seq_len,
            "cuda": args.cuda,
            "attack_method": args.attack_method,
            "save_dir": save_dir,
            "datadir": datadir
        }
        fixed_pool_fl_trainer = FixedPoolFL_TSadv(arguments=arguments)
        fixed_pool_fl_trainer.run()


if __name__ == "__main__":
    main()
