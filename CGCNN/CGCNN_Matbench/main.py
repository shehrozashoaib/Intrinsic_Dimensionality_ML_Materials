import argparse
import os
import shutil
import sys
import time
import warnings
import random
from pathlib import Path
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader

from cgcnn.data import CIFData, TensorCIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet
from cgcnn.subspace import RandomSubspaceWrapper, resolve_subspace_dim

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='*',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--cached-data-dir', default=None, type=str,
                    help='load precomputed CGCNN tensors from this directory')
parser.add_argument('--matbench-task', default='matbench_phonons', type=str,
                    help='Matbench task to train when using cached tensors')
parser.add_argument('--output-dir', default='.', type=str,
                    help='directory for checkpoints and test_results.csv')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=None, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='Adam', type=str, metavar='Adam',
                    help='choose an optimizer, SGD or Adam, (default: Adam)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=6, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=2, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('--random-seed', default=123, type=int, metavar='N',
                    help='random seed for model initialization and subspace projection')
parser.add_argument('--data-seed', default=None, type=int, metavar='N',
                    help='seed of cached tensor split to load. If omitted, defaults to --random-seed')
parser.add_argument('--subspace-method', '--wrapper-type', default='none',
                    choices=['none', 'dense', 'fastfood'],
                    help='random subspace projection method (default: none)')
parser.add_argument('--id-dim', default=None,
                    help='intrinsic dimension d, or a fraction in (0, 1] of full parameter count')
parser.add_argument('--id-ortho', action='store_true',
                    help='orthonormalize dense projection columns with QR')
parser.add_argument('--subspace-full-rotation', action='store_true',
                    help='for full-dimensional dense subspace, use permutation/sign rotation')
parser.add_argument('--subspace-z-init-std', default=0.0, type=float,
                    help='stddev for random initialization of intrinsic vector z (default: 0 starts at theta0)')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_parameters(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('No trainable parameters found for optimizer.')
    return params


def maybe_wrap_subspace(model):
    if args.subspace_method == 'none':
        return model

    d_value = resolve_subspace_dim(args.id_dim)
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    wrapped = RandomSubspaceWrapper(
        model,
        d=d_value,
        method=args.subspace_method,
        orthonormal=args.id_ortho,
        full_rotation=args.subspace_full_rotation,
        seed=args.random_seed,
        z_init_std=args.subspace_z_init_std,
        device=device,
    )
    print(f'[Subspace] method={wrapped.method}, D={wrapped.D:,}, d={wrapped.d:,}')
    print(f'[Subspace] trainable parameters={sum(p.numel() for p in wrapped.parameters() if p.requires_grad):,}')
    return wrapped


def make_normalizer(dataset):
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
        return normalizer

    if len(dataset) < 500:
        warnings.warn('Dataset has less than 500 data points. Lower accuracy is expected. ')
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
    _, sample_target = collate_pool(sample_data_list)
    return Normalizer(sample_target)


def build_cgcnn_from_dataset(dataset):
    structures, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=True if args.task == 'classification' else False,
    )
    model = maybe_wrap_subspace(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Model] total parameters={total_params:,}, trainable parameters={trainable_params:,}')
    if args.cuda:
        model.cuda()
    return model


def make_criterion_and_optimizer(model):
    criterion = nn.NLLLoss() if args.task == 'classification' else nn.MSELoss()
    opt_params = trainable_parameters(model)
    if args.optim == 'SGD':
        optimizer = optim.SGD(
            opt_params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam':
        optimizer = optim.Adam(opt_params, args.lr, weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')
    return criterion, optimizer


def run_training(task_name, train_loader, val_loader, test_loader, train_dataset):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_cgcnn_from_dataset(train_dataset)
    criterion, optimizer = make_criterion_and_optimizer(model)
    normalizer = make_normalizer(train_dataset)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    best_metric = 1e10 if args.task == 'regression' else 0.0
    checkpoint_name = output_dir / f'{task_name}_checkpoint.pth.tar'
    best_name = output_dir / f'{task_name}_model_best.pth.tar'

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, normalizer)
        metric = validate(val_loader, model, criterion, normalizer)
        if metric != metric:
            print('Exit due to NaN')
            sys.exit(1)
        scheduler.step()

        if args.task == 'regression':
            is_best = metric < best_metric
            best_metric = min(metric, best_metric)
        else:
            is_best = metric > best_metric
            best_metric = max(metric, best_metric)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_metric,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args),
        }, is_best, filename=str(checkpoint_name), best_filename=str(best_name))

    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load(best_name, map_location='cpu')
    model.load_state_dict(best_checkpoint['state_dict'])
    test_metric = validate(test_loader, model, criterion, normalizer, test=True)
    print(f'{task_name} test metric is: {test_metric}')
    return test_metric


def run_cached_training():
    task_dir = Path(args.cached_data_dir)
    data_seed = args.data_seed if args.data_seed is not None else args.random_seed
    if (task_dir / args.matbench_task).is_dir():
        task_dir = task_dir / args.matbench_task / f'seed{data_seed}'
    elif (task_dir / f'seed{data_seed}').is_dir():
        task_dir = task_dir / f'seed{data_seed}'

    train_path = task_dir / 'train.pt'
    val_path = task_dir / 'val.pt'
    test_path = task_dir / 'test.pt'
    for split_path in (train_path, val_path, test_path):
        if not split_path.exists():
            raise FileNotFoundError(f'Missing cached tensor split: {split_path}')

    print(f'[cache] loading tensors from {task_dir}')
    print(f'[seed] data_seed={data_seed}, model_seed={args.random_seed}')
    train_dataset = TensorCIFData(train_path)
    val_dataset = TensorCIFData(val_path)
    test_dataset = TensorCIFData(test_path)
    train_loader = DataLoader(
        train_dataset, collate_fn=collate_pool, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.cuda, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, collate_fn=collate_pool, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.cuda,
    )
    test_loader = DataLoader(
        test_dataset, collate_fn=collate_pool, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.cuda,
    )
    return run_training(args.matbench_task, train_loader, val_loader, test_loader, train_dataset)


def main():
    global args
    set_random_seed(args.random_seed)

    if args.cached_data_dir:
        run_cached_training()
        return

    # load data (matbench only needed for the non-cached benchmark path)
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False)

    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            best_mae_error = 1e10
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            data_seed = args.data_seed if args.data_seed is not None else args.random_seed
            dataset = CIFData(train_inputs, train_outputs, random_seed=data_seed)
            collate_fn = collate_pool
            train_ratio = args.train_ratio if args.train_ratio is not None else 0.8
            val_ratio = args.val_ratio if args.val_ratio is not None else 0.2
            train_loader, val_loader = get_train_val_test_loader(
                dataset, collate_fn=collate_pool, batch_size=args.batch_size,
                return_test=False, train_ratio=train_ratio,
                val_ratio=val_ratio, test_ratio=0.0,
                num_workers=args.workers, pin_memory=args.cuda,
                train_size=args.train_size, test_size=None,
                val_size=args.val_size, random_seed=data_seed
            )

            # obtain target value normalizer
            if args.task == 'classification':
                normalizer = Normalizer(torch.zeros(2))
                normalizer.load_state_dict({'mean': 0., 'std': 1.})
            else:
                if len(dataset) < 500:
                    warnings.warn('Dataset has less than 500 data points. '
                                  'Lower accuracy is expected. ')
                    sample_data_list = [dataset[i] for i in range(len(dataset))]
                else:
                    sample_data_list = [dataset[i] for i in
                                        sample(range(len(dataset)), 500)]
                _, sample_target = collate_pool(sample_data_list)
                normalizer = Normalizer(sample_target)

            # build model
            structures, _ = dataset[0]
            orig_atom_fea_len = structures[0].shape[-1]
            nbr_fea_len = structures[1].shape[-1]
            model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                        atom_fea_len=args.atom_fea_len,
                                        n_conv=args.n_conv,
                                        h_fea_len=args.h_fea_len,
                                        n_h=args.n_h,
                                        classification=True if args.task ==
                                                               'classification' else False)
            model = maybe_wrap_subspace(model)
            if args.cuda:
                model.cuda()

            # define loss func and optimizer
            if args.task == 'classification':
                criterion = nn.NLLLoss()
            else:
                criterion = nn.MSELoss()
            opt_params = trainable_parameters(model)
            if args.optim == 'SGD':
                optimizer = optim.SGD(opt_params, args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)
            elif args.optim == 'Adam':
                optimizer = optim.Adam(opt_params, args.lr,
                                       weight_decay=args.weight_decay)
            else:
                raise NameError('Only SGD or Adam is allowed as --optim')

            # optionally resume from a checkpoint
            if args.resume:
                if os.path.isfile(args.resume):
                    print("=> loading checkpoint '{}'".format(args.resume))
                    checkpoint = torch.load(args.resume)
                    args.start_epoch = checkpoint['epoch']
                    best_mae_error = checkpoint['best_mae_error']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    normalizer.load_state_dict(checkpoint['normalizer'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))

            scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                                    gamma=0.1)

            for epoch in range(args.start_epoch, args.epochs):
                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, normalizer)

                # evaluate on validation set
                mae_error = validate(val_loader, model, criterion, normalizer)

                if mae_error != mae_error:
                    print('Exit due to NaN')
                    sys.exit(1)

                scheduler.step()

                # remember the best mae_eror and save checkpoint
                if args.task == 'regression':
                    is_best = mae_error < best_mae_error
                    best_mae_error = min(mae_error, best_mae_error)
                else:
                    is_best = mae_error > best_mae_error
                    best_mae_error = max(mae_error, best_mae_error)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mae_error': best_mae_error,
                    'optimizer': optimizer.state_dict(),
                    'normalizer': normalizer.state_dict(),
                    'args': vars(args)
                }, is_best)

            # test best model
            print('---------Evaluate Model on Test Set---------------')
            best_checkpoint = torch.load('model_best.pth.tar')
            model.load_state_dict(best_checkpoint['state_dict'])
            test_inputs,test_outputs = task.get_test_data(fold, include_target=True)
            test_dataset = CIFData(test_inputs, test_outputs, random_seed=data_seed)
            test_loader = DataLoader(test_dataset, collate_fn=collate_fn,
                                          batch_size=args.batch_size,
                                          num_workers=args.workers,
                                          pin_memory=args.cuda)

            mae_test = validate(test_loader, model, criterion, normalizer, test=True)
            print("Test MAE is:",mae_test)
            predictions = []
            with open('test_results.csv', 'r') as f:
                import csv
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    predictions.append(float(row[1]))
            task.record(fold, predictions)

            
        break
    mb.to_file("my_models_benchmark.json.gz")


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )


def validate(val_loader, model, criterion, normalizer, test=False):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []


    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)


        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        if args.cached_data_dir:
            result_path = Path(args.output_dir) / f'{args.matbench_task}_test_results.csv'
        else:
            result_path = Path('test_results.csv')
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open('w', newline='') as f:
            writer = csv.writer(f)
            for target, pred in zip(test_targets, test_preds):
                writer.writerow((target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    prediction = prediction.view(-1)
    target = target.view(-1)
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename=None):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename or 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
