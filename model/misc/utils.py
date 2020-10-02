import sys
import os
import time
import subprocess
import inspect
import logging
import argparse
from contextlib import contextmanager
from timeit import default_timer

import matplotlib.pyplot as plt
import torch
import random
import uuid
import numpy as np
import xmltodict


# ---------- debugging ---------- #
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
                print(f'layer: {n}, ave grad: {ave_grads[-1]:.8f}, max grad: {max_grads[-1]:.8f}.')
            else:
                print(f'layer: {n} has no grad.')
                ave_grads.append(-1.)
                max_grads.append(-1.)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="b")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="g")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.savefig('grad.local.png', bbox_inches='tight')


# ---------- benchmark ---------- #
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


# ---------- checkpoint handling ---------- #
def load_parallel_state_dict(state_dict):
    """Remove the module.xxx in the keys for models trained
        using data_parallel.

    Returns:
        new_state_dict
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def save_checkpoint(path, **kwargs):
    torch.save(kwargs, path)


def load_checkpoint(path, state_dict_to_load=None, from_parallel=False):
    """Load checkpoint from path

    Args:
        path :str: path of checkpoint file.
        state_dict_to_load :[]: keys of states to load. Set it to None if checkpoint has only

    Returns:
        checkpoint :dict of state_dicts:
    """
    checkpoint = torch.load(path)
    if from_parallel:
        checkpoint['model'] = load_parallel_state_dict(checkpoint['model'])
    if state_dict_to_load is None:
        return checkpoint
    if set(state_dict_to_load) != set(list(checkpoint.keys())):
        logging.warning(f'Checkpoint key mismatch. '
                        f'Requested {set(state_dict_to_load)}, found {set(list(checkpoint.keys()))}.')

    return checkpoint


def prepare_train(model, optimizer, lr_scheduler, args, **kwargs):
    """Do the dirty job of loading model/model weights, optimizer and lr_scheduler states from saved state-dicts.
    
    If args.from_model is set, the states will be fully recovered.
    
    If args.load_model_weight is set instead, only model weight will be loaded. Optimizer and lr_scheduler will not be loaded.
    
    Args:
        model, optimizer, lr_scheduler
        args: argument returned by init()
        If args.finetune is set:
            kwargs['finetune_old_head'] :torch.nn.Module: head the model that is to be replaced
            kwargs['finetune_new_head'] :torch.nn.Module: new head that will be appended to model

    Returns:
        model, optimizer, lr_scheduler
    
    """

    if args.from_model:
        state_dict = load_checkpoint(args.from_model)

        if 'checkpoint_epoch' in state_dict.keys():
            args.start_epoch = state_dict['checkpoint_epoch'] + 1

        if 'model' in state_dict.keys():
            if not args.parallel:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict(
                    load_parallel_state_dict(state_dict['model']))
        else:
            if not args.parallel:
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(load_parallel_state_dict(state_dict))

        # if --finetune is set, the head is reset to a new 1x1 conv layer
        if args.finetune:
            setattr(model, kwargs['finetune_old_head'], kwargs['finetune_new_head'])

        if 'optimizer' in state_dict.keys():
            optimizer.load_state_dict(state_dict['optimizer'])
        if 'initial_lr' in state_dict.keys():
            optimizer.param_groups[0]['initial_lr'] = state_dict['initial_lr']
        else:
            optimizer.param_groups[0]['initial_lr'] = args.lr

        if 'lr_scheduler' in state_dict.keys():
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

    if args.load_weight_from and not args.from_model:
        state_dict = load_checkpoint(args.load_weight_from)
        if 'model' in state_dict.keys():
            if not args.parallel:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict(
                    load_parallel_state_dict(state_dict['model']))
        else:
            if not args.parallel:
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(load_parallel_state_dict(state_dict))

    if args.parallel:
        model = torch.nn.DataParallel(model)

    model = model.to(args.device)
    
    return model, optimizer, lr_scheduler


# ---------- data handling ---------- #
def longtensor_to_one_hot(labels, num_classes):
    """convert int encoded label to one-hot encoding

    Args:
        labels :[batch_size, 1]:
        num_classes :int:

    Returns:
        one-hot encoded label :[batch_size, num_classes]:
    """
    return torch.zeros(labels.shape[0], num_classes).scatter_(1, labels, 1)


# ---------- training ---------- #
class EarlyStop:
    def __init__(self, patience: int, verbose: bool = True):
        self.patience = patience
        self.init_patience = patience
        self.verbose = verbose
        self.lowest_loss = 9999999.999
        self.highest_acc = 0.0

    def step(self, loss=None, acc=None, criterion=lambda x1, x2: x1 or x2):
        if loss is None:
            loss = self.lowest_loss
            better_loss = True
        else:
            better_loss = (loss < self.lowest_loss) and ((self.lowest_loss-loss)/self.lowest_loss > 0.01)
        if acc is None:
            acc = self.highest_acc
            better_acc = True
        else:
            better_acc = acc > self.highest_acc
        
        if better_loss:
            self.lowest_loss = loss
        if better_acc:
            self.highest_acc = acc

        if criterion(better_loss, better_acc):
            self.patience = self.init_patience
            if self.verbose:
                logging.getLogger(myself()).debug(
                    'Remaining patience: {}'.format(self.patience))
            return False
        else:
            self.patience -= 1
            if self.verbose:
                logging.getLogger(myself()).debug(
                    'Remaining patience: {}'.format(self.patience))
            if self.patience < 0:
                if self.verbose:
                    logging.getLogger(myself()).warning('Ran out of patience.')
                return True


class ShouldSaveModel:
    def __init__(self, init_step=-1):
        """
        Args:
            init_step :int: start_epoch - 1
        """
        self.lowest_loss = 999999.999
        self.highest_acc = 0.0
        self.current_step = init_step
        self.best_step = init_step

    def step(self, loss=None, acc=None, criterion=lambda x1, x2: x1 or x2):
        """
        Decides whether a model should be saved, based on the criterion.

        Args:
            loss :float: loss after current epoch.
            acc :float: acc after current epoch.
            criterion :callable: a function that takes two params and returns a bool.

        Returns:
            :bool: whether this model should be saved.
        """
        self.current_step += 1
        if loss is None:
            loss = self.lowest_loss
            better_loss = True
        else:
            better_loss = (loss < self.lowest_loss) and ((self.lowest_loss-loss)/self.lowest_loss > 0.01)
        if acc is None:
            acc = self.highest_acc
            better_acc = True
        else:
            better_acc = acc > self.highest_acc

        if better_loss:
            self.lowest_loss = loss
        if better_acc:
            self.highest_acc = acc
        if criterion(better_loss, better_acc):
            logging.getLogger(myself()).info(
                f'New model: epoch: {self.current_step}, highest acc: {acc:.4}, lowest loss: {loss:.4}.')
            self.best_step = self.current_step
            return True
        else:
            return False


class RunningAverage:
    def __init__(self, window_size, initial_step=0):
        self.data = np.zeros([window_size, 1])
        self.window_size = window_size
        self.step = initial_step
        self.idx = -1
    
    def value(self):
        try:
            return self.data.sum() / self.step
        except ZeroDivisionError:
            return 0

    def add(self, d):
        self.idx = (self.idx + 1) % self.window_size
        self.data[self.idx] = d
        self.step += 1
        return self.data.mean()


# ---------- environment setup and logging ---------- #
def myself():
    return inspect.stack()[1][3]


def get_usable_gpu(threshold=2048, gpu_id_remap=None):
    """Find a usable gpu

    Args:
        threshold :int: required GPU free memory.
        gpu_id_remap :[int]: in cases where GPU IDs mess up, use a remap

    Returns:
        GPU ID :int:, or
        :None: if no GPU is found
    """
    gpu_id = None
    try:
        gpu_info = xmltodict.parse(subprocess.check_output(
            ['nvidia-smi', '-x', '-q']))['nvidia_smi_log']
        free_mem = []
        if type(gpu_info['gpu']) is list:
            for gpu in gpu_info['gpu']:
                free_mem.append(int(gpu['fb_memory_usage']['free'].split()[0]))
        else:
            free_mem.append(
                int(gpu_info['gpu']['fb_memory_usage']['free'].split()[0]))
        gpu_id = np.argmax(free_mem)
        best_memory = free_mem[gpu_id]
        if gpu_id_remap:
            gpu_id = gpu_id_remap[gpu_id]
        if best_memory < threshold:
            gpu_id = None
    except Exception as e:
        print(e)

    return gpu_id


def wait_gpu(req_mem=8000, id_map=None):
    wait_time = int(random.random() * 30)
    time.sleep(wait_time)
    while True:
        gpu_id = get_usable_gpu(req_mem, id_map)
        if gpu_id is not None:
            break
        time.sleep(30)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def config_logger(log_file=None):
    if log_file is not None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s [%(levelname)-8.8s] (%(name)-8.8s %(filename)15.15s:%(lineno)5.5d) %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=log_file,
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8.8s] (%(name)-8.8s %(filename)15.15s:%(lineno)5.5d) %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s [%(levelname)-8.8s] (%(name)-8.8s %(filename)15.15s:%(lineno)5.5d) %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            stream=sys.stdout)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger('').critical("Uncaught exception",
                                       exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def append_test_args(p):
    """Append arguments for model testing

    Args:
        p :argparse.ArgumentParser object:

    Returns
        parameters :argparse.ArgumentParser object: with appended arguments
    """
    p.add_argument('--from_model', '--from-model',
                   nargs='+', type=str, required=True)
    p.add_argument('--cuda', action='store_true', default=False)
    p.add_argument('--parallel', action='store_true', default=False)
    p.add_argument('--num_workers', '--num-workers', type=int, default=2)
    p.add_argument('--batch_size', '--batch-size', type=int, default=32)
    p.add_argument('--dataset', choices=['mitstates', 'ut-zap50k'], required=True,
                   help='Dataset for training and testing.')

    return p


def create_parser(user_param=None):
    """Create the basic argument parser for environment setup.

    Args:
        user_param :callable: a function that takes and returns an ArgumentParser. Can be used to add user parameters.

    Return:
        p :argparse.ArgumentParser:
    """
    p = argparse.ArgumentParser(description='input arguments.')

    p.add_argument('--no-pbar', action='store_true',
                   default=False, help='Subpress progress bar.')
    p.add_argument('--log_dir', default=None)
    p.add_argument('--debug_mode', '--debug-mode', action='store_true', default=False)
    p.add_argument('--summary_to', type=str, default=None)
    p.add_argument('--uuid', default=None,
                   help='UUID of the model. Will be generated automatically if unspecified.')
    p.add_argument('--cuda', action='store_true', default=False,
                   help='Flag for cuda. Will be automatically determined if unspecified.')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                   help='Flag for cuda. Will be automatically determined if unspecified.')
    p.add_argument('--parallel', action='store_true', default=False,
                   help='Flag for parallel.')
    p.add_argument('--start_epoch', type=int, default=0)
    p.add_argument('--max_epoch', type=int, default=100)
    p.add_argument('--from_model', '--from-model', type=str, default=None,
                        help='Load model, optimizer, lr_scheduler state from path.')
    p.add_argument('--finetune', action='store_true', default=False)
    p.add_argument('--load_weight_from', '--load-weight-from', type=str, default=None,
                        help='Load model state from path. This will invalidate --finetune flag.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--test_only', '--test-only', action='store_true', default=False,
                    help='Disable training. Model will only be tested.')
    p.add_argument('--save_model_to', type=str, default='./snapshots/')
    p.add_argument('--patience', type=int, default=10,
                   help='Number of epochs to continue when test acc stagnants.')

    if user_param:
        p = user_param(p)

    if type(p) != argparse.ArgumentParser:
        raise ValueError(
            f'user_param must return an ArgumentParser object, found {type(p)} instead.')

    return p


def worker_init_fn_seed(args):
    def worker_init_fn(x):
        seed = x + args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        return
    return worker_init_fn


def set_randomness(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init(user_param=None, user_args_modifier=None):
    """Parse and return arguments, prepare the environment. Set up logging.
    Args:
        user_param :callable: append user parameters to the argument parser.
        user_args_modifier :callable: override parsed arguments.
    Returns:
        args :Namespace: of arguments
    """

    # parse input arguments
    parser = create_parser(user_param)
    args = parser.parse_args()

    # detect CUDA
    args.cuda = torch.cuda.is_available() if args.cuda is None else args.cuda

    # detect my hostname and gpu id
    hostname = subprocess.check_output(
        "hostname", shell=True).decode("utf-8")[:-1]
    git_commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8')[:-1]
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES") if args.cuda else 'null'
    date_time = subprocess.check_output(['date', '-Iminutes']).decode('utf-8')[:-7]

    args.git_commit = git_commit

    # randomness control
    set_randomness(args.seed)

    # model ID
    # if debug_mode flag is set, all logs will be saved to debug/model_id folder,
    # otherwise will be saved to runs/model_id folder
    if not args.debug_mode:
        args.model_id = args.uuid if args.uuid is not None else f'{str(uuid.uuid4().hex)[:8]}_{hostname}_{gpu_id}_{os.getpid()}_{git_commit}_{date_time}'
        args.summary_to = args.summary_to if args.summary_to is not None else f'./runs/{args.model_id}/'
    else:
        args.model_id = f'debug_{hostname}_{gpu_id}_{os.getpid()}_{git_commit}'
        args.summary_to = 'debug'

    # create model save path
    if not args.test_only:
        # create logger
        if args.log_dir is None:
            args.log_dir = os.path.join(args.save_model_to, args.model_id)
        if not os.path.exists(args.save_model_to):
            os.mkdir(args.save_model_to)
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        if not os.path.exists(os.path.join(args.save_model_to, args.model_id)):
            os.mkdir(os.path.join(args.save_model_to, args.model_id))
        config_logger(os.path.join(args.log_dir, args.model_id + '.log'))

        # log
        logging.getLogger(myself()).info(
            f'Model {args.model_id}, running in {sys.argv[0]}, code revision {git_commit}')
        for arg in vars(args):
            logging.getLogger(myself()).debug(
                f'{arg:<30s} = {str(getattr(args, arg)):<30s}')
    else:
        print(f'Model {args.model_id}, running with {sys.argv[0]}')
        for arg in vars(args):
            print(f'{arg:<30s} = {str(getattr(args, arg)):<30s}')

    if user_args_modifier:
        args = user_args_modifier(args)

    if not args:
        raise ValueError('user_args_modifier must return args, not None.')

    return args


# ---------- Debug use only ---------- #
if __name__ == '__main__':
    get_usable_gpu()
    pass
