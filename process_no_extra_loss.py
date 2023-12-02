import logging
import math
import operator
import time
import os
import torch as t

from util import AverageMeter

__all__ = ['train', 'validate', 'PerformanceScoreboard']
LPR8B = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 14, 14, 15, 15, 16, 16, 17, 17, 22, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 40, 41, 41, 46, 46, 47, 47, 48, 48, 49, 49, 54, 54, 55, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 72, 72, 72, 73, 73, 73, 73, 78, 78, 78, 78, 79, 79, 79, 79, 112, 112, 112, 112, 113, 113, 113, 113, 118, 118, 118, 118, 119, 119, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127, -128, -128, -127, -127, -126, -126, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -120, -120, -120, -119, -119, -119, -119, -114, -114, -114, -114, -113, -113, -113, -113, -80, -80, -80, -80, -79, -79, -79, -79, -74, -74, -74, -74, -73, -73, -73, -73, -72, -72, -71, -71, -70, -70, -69, -69, -68, -68, -67, -67, -66, -66, -65, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -56, -55, -55, -50, -50, -49, -49, -48, -48, -47, -47, -42, -42, -41, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -24, -23, -23, -18, -18, -17, -17, -16, -16, -15, -15, -10, -10, -9, -9, -8, -7, -6, -5, -4, -3, -2, -1]

logger = logging.getLogger()

# def get_branch_costs():
#     RANKS = [6, 4, 2]
#     branch_costs = list()
#     rsum = 0
#     for r in RANKS:
#         rsum += r*r
#     for r in RANKS:
#         branch_costs.append(r*r/rsum)
#     return t.tensor(branch_costs)


# def make_alpha_dict(model):
#     branch_costs = get_branch_costs()

#     alpha_dict = dict()
#     for name, value in model.named_parameters():
#         if 'alpha' in name:
#             alpha_dict[value] = branch_costs
#     return alpha_dict
def get_branch_costs(state_dict,total_num,name_alpha):
    BITS = [2,4,8]
    SPARSITY=[0.875,0.75,0.5]
    name_weight = name_alpha.replace(".alpha",".op_list.0.weight")
    num = state_dict[name_weight].data.detach().numel()
    branch_costs = list()
    total = 0
    for b in BITS:
        total += b*total_num
    for b in BITS:
        branch_costs.append(b*num/total)
    return t.tensor(branch_costs)
import copy
def make_alpha_dict(model):
    model_cp = copy.deepcopy(model)
    total_num = model_cp.module.calculate_complexity()
    state_dict = model_cp.state_dict()
    alpha_dict = dict()
    for name, value in model.named_parameters():
        if 'alpha' in name:
            branch_costs = get_branch_costs(state_dict,total_num,name)
            alpha_dict[value] = branch_costs
    return alpha_dict


def calc_comp_cost(alpha_dict, ori_loss, gamma=0.1):
    # print("ori_loss",ori_loss)
    comp_cost = t.tensor(0.0).to(ori_loss.device) 
    for alpha in alpha_dict.keys():
        softmax_alpha = t.softmax(alpha, -1)
        branch_cost = alpha_dict[alpha].to(softmax_alpha.device)
        comp_cost += (branch_cost * softmax_alpha).sum()
        # print("comp_cost",comp_cost)
    cost = comp_cost.detach()
    if(cost != 0):
        scale = ori_loss.detach() / comp_cost.detach()
    else:
        scale = 0
    #scale = 1
    #print("comp_cost",comp_cost)
    total_loss = ori_loss*(1-gamma) + comp_cost*scale*gamma
    return total_loss



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    # alpha_dict = make_alpha_dict(model)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device.type)
        targets = targets.to(args.device.type)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # loss = calc_comp_cost(alpha_dict, loss)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch, batch=batch_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LRG0': optimizer.param_groups[0]['lr']
                    #'LRG1': optimizer.param_groups[1]['lr']
                })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


def validate(data_loader, model, criterion, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)
    id_image = 0
    model.eval()
    end_time = time.time()
    
    directory = "/home/gaoconghao/mix_lcd/quant_mix_lr/feat_txt/"


    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with t.no_grad():
            inputs = inputs.to(args.device.type)
            targets = targets.to(args.device.type)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            # print(acc1)
#             inputs = inputs * 8
#             inputs = inputs.round()
            
#             x = t.clamp(inputs, -128, 127)
#             # x = round_pass(x)
            
#             # dim_data = t.tensor(x.reshape(-1), dtype=t.long).clone().detach().cuda()
            
#             dim_data = x.view(-1).long().clone().cuda()

#             label_data = t.tensor(LPR8B, dtype=t.long).cuda()
            
#             data_x = t.take(label_data,  dim_data).cuda() 
#             #print(data_x)
#             feat = data_x.cpu().numpy()
#             name_t = str(id_image)
#             filename = name_t + '.txt'

# # 构建文件路径
#             filepath = os.path.join(directory, filename)
#             fp = open(filepath, 'w')
#             for d in feat:
#                 n = str(d)
#                 fp.write(n + '\n')
#             # if acc1 != 100:
#             #     print(id_image)
#             fp.close()
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            id_image = id_image + 1
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.log.print_freq == 0:
                for m in monitors:
                    m.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                        'Loss': losses,
                        'Top1': top1,
                        'Top5': top5,
                        'BatchTime': batch_time
                    })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
