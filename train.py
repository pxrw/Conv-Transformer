import os, time, sys
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from model.build_model_ori import Attn2AttnDepth
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.tool import silog_loss, normalize_result, inv_normalize, flip_lr, post_process_depth, compute_errors, \
    eval_metrics, block_print, enable_print
from utils.opt import args

if args.dataset == 'kitti':
    from dataset.dataloader_kitti import AttnDataLoader
else:
    from dataset.dataloader_nyu import AttnDataLoader

def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

def online_eval(model, dataloader_eval, gpu, ngpus, post_process=False):
    '''
    online_eval时需要用到的数据，及处理过程
    从dataloader里面可以知道，当mode = online_eval时，生成的数据是[h, w, c]格式的
    '''
    # 生成一个列表，里面有10个0
    args_dataset = args.kitti if args.dataset == 'kitti' else args.nyu
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))  # [224, 1216]
            gt_depth = eval_sample_batched['depth']  # [224, 1216]
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                continue

            # [n, 1, h, w]
            pred_depth = model(image)

            # 一般情况下，验证过程不需要对数据进行增强处理
            if post_process:
                image_flipped = flip_lr(image)
                # [n, 1, h, w]
                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args_dataset.do_kb_crop:
            # 对于需要做裁切的情况，先获取当前depth的深度图
            height, width = gt_depth.shape
            # 根据要求拿到顶部和左侧的边界
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            # 创造一个全黑的大小和depth_gt一样的背景
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            # 将需要裁切的部分赋值为pred_depth
            pred_depth_uncropped[top_margin: top_margin + 352, left_margin: left_margin + 1216] = pred_depth
            # 再赋值回去
            pred_depth = pred_depth_uncropped

        # 做一下数据截断，圈定depth中的数据最大、最小值，防止越界
        pred_depth[pred_depth < args_dataset.min_depth_eval] = args_dataset.min_depth_eval
        pred_depth[pred_depth > args_dataset.max_depth_eval] = args_dataset.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args_dataset.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args_dataset.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args_dataset.min_depth_eval, gt_depth < args_dataset.max_depth_eval)

        if args_dataset.garg_crop or args_dataset.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args_dataset.garg_crop:
                eval_mask[int(0.40810811 * gt_height): int(0.99189189 * gt_height),
                int(0.03594771 * gt_width): int(0.96405229 * gt_width)] = 1
            elif args_dataset.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height): int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width): int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45: 471, 41: 601] = 1

            valid_mask = np.logcal_and(valid_mask, eval_mask)

        meaures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(meaures).cuda(device=gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = Attn2AttnDepth(args)

    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    # TODO 这里关于model.train()需要冻结哪些参数
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            if args.norm == 'BN':
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("== Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=args.learning_rate, eps=args.adam_eps)

    # 加载整个模型的预训练模型，而不是encoder的
    model_just_loaded = False
    if args.ckpt != '':
        if os.path.isfile(args.ckpt):
            print('== Loading checkpoint {}'.format(args.ckpt))
            if args.gpu is None:
                checkpoint = torch.load(args.ckpt)
            else:
                loc = 'cuda: {}'.format(args.gpu)
                checkpoint = torch.load(args.ckpt, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("== Loaded checkpoint '{}' (global_step {})".format(args.ckpt, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.ckpt))
        model_just_loaded = True
        del checkpoint

    cudnn.benchmark = True

    dataloader = AttnDataLoader(args, 'train')
    dataloader_eval = AttnDataLoader(args, 'online_eval')

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_dir + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_dir != '':
                eval_summary_path = os.path.join(args.eval_summary_dir, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_dir, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    # TODO loss的计算方法，看看还有没有什么改进的地方
    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.03 * args.learning_rate

    # len(dataloader)返回的是根据batch进行化分的数据集又多少个（比如total = 1000, batch = 64, len(dataloader) = 15，这是向下取整的结果）
    steps_per_epoch = len(dataloader.data)  # 当batch = 8的情况下，steps_per_epoch = 2806
    num_total_steps = args.num_epochs * steps_per_epoch  # 50 x 2806 = 140300
    epoch = global_step // steps_per_epoch
    args.eval_freq = steps_per_epoch // 2

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)
            dataloader.training_samples.epoch_now = epoch
            dataloader.g = get_ddp_generator()

        # 从dataloader中读取数据，开始训练
        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            # 从dataloader中读取数据，因为dataloader返回的是sample_dict，通过key找到对应的值 image: [n, c, h, w]
            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))

            # 从dataloader_train中读出的depth_gt: [n, 1, h, w]
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            depth_pred = model(image)

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1
            else:
                mask = depth_gt > 1.0

            # 求损失，然后反传
            loss = silog_criterion(depth_pred, depth_gt, mask.to(torch.bool))
            loss.backward()

            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (
                            1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (
                    args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print(
                    '[epoch][step/step_per_epoch/global_step]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch,
                                                                                                                  step,
                                                                                                                  steps_per_epoch,
                                                                                                                  global_step,
                                                                                                                  current_lr,
                                                                                                                  loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (
                        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item() / var_cnt,
                                          time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('silog_loss', loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var average', var_sum.item() / var_cnt, global_step)
                    depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                    for i in range(num_log_images):
                        writer.add_image('depth_gt/image/{}'.format(i), normalize_result(1 / depth_gt[i, ...].data),
                                         global_step)
                        writer.add_image('depth_pred/image/{}'.format(i), normalize_result(1 / depth_pred[i, ...].data),
                                         global_step)
                        writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, ...]).data, global_step)
                    writer.flush()

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node, post_process=True)
                if eval_measures is not None:
                    print('This session will save model, please check model has been saved successfully...')
                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i - 6]:
                            old_best = best_eval_measures_lower_better[i - 6].item()
                            best_eval_measures_higher_better[i - 6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.save_path + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'model': model.state_dict()}
                            torch.save(checkpoint, args.save_path + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()


def main():
    print('current dataset is:', args.dataset)
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    command = 'mkdir ' + os.path.join(args.log_dir, args.model_name)
    os.system(command)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # args_out_path = os.path.join(args.log_dir, args.model_name)
    # command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    # os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print(
            "This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()