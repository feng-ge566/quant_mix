import logging
from pathlib import Path

import torch as t
import yaml

import process_no_extra_loss as process
import quan
import util
from model.my_model import create_model
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml')

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        t.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = True
        t.backends.cudnn.deterministic = False

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    #Create the model
    model = create_model(args)
    print(model)

    state_dict =t.load('/home/likai/mix_vgg/vgg_quant_test/lsq_group_sparse/out/VGG16_ImageNet_bmix_s0_20230520-144724/VGG16_ImageNet_bmix_s0_best.pth.tar')['state_dict']
   

    tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))
    logger.info('Inserted quantizers into the original model')

    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)

    model.to(args.device.type)


    model.load_state_dict(state_dict,strict=False)


    model.module.init_weight()
    start_epoch = 0
    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, args.device.type, lean=args.resume.lean)

    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(args.device.type)


    #conv0_params  = list(map(id, model.module.features[ 0].parameters())) #8
    #conv1_params  = list(map(id, model.module.features[ 2].parameters())) #4
    conv2_params  = list(map(id, model.module.features[ 5].parameters()))  #4
    conv3_params  = list(map(id, model.module.features[ 7].parameters()))  #2
    conv4_params  = list(map(id, model.module.features[10].parameters()))  #4
    conv5_params  = list(map(id, model.module.features[12].parameters()))  #2
    conv6_params  = list(map(id, model.module.features[14].parameters()))  #4
    conv7_params  = list(map(id, model.module.features[17].parameters()))  #4
    conv8_params  = list(map(id, model.module.features[19].parameters()))  #2
    conv9_params  = list(map(id, model.module.features[21].parameters()))  #4
    conv10_params = list(map(id, model.module.features[24].parameters()))  #4
    conv11_params = list(map(id, model.module.features[26].parameters()))  #2
    conv12_params = list(map(id, model.module.features[28].parameters()))  #4

    # fc0_params    = list(map(id, model.module.classifier[0].parameters()))
    # fc1_params    = list(map(id, model.module.classifier[3].parameters()))
    # fc2_params    = list(map(id, model.module.classifier[6].parameters()))

    #base_params = filter(lambda p: id(p) not in conv8_params, model.parameters())


    base_params = filter(lambda p: id(p) not in conv2_params + conv3_params + conv4_params + 
                         conv5_params + conv6_params + conv7_params + conv8_params + 
                         conv9_params + conv10_params + conv11_params + conv12_params, model.parameters())
    params = [{'params': base_params},
          {'params': model.module.features[ 5].parameters(), 'lr': args.optimizer.learning_rate * 1},#4
          {'params': model.module.features[ 7].parameters(), 'lr': args.optimizer.learning_rate * 1000},#2
          {'params': model.module.features[10].parameters(), 'lr': args.optimizer.learning_rate * 1},#4
          {'params': model.module.features[12].parameters(), 'lr': args.optimizer.learning_rate * 1000},#2
          {'params': model.module.features[14].parameters(), 'lr': args.optimizer.learning_rate * 1},#4
          {'params': model.module.features[17].parameters(), 'lr': args.optimizer.learning_rate * 1},#4
          {'params': model.module.features[19].parameters(), 'lr': args.optimizer.learning_rate * 1000},#2
          {'params': model.module.features[21].parameters(), 'lr': args.optimizer.learning_rate * 1},#4
          {'params': model.module.features[24].parameters(), 'lr': args.optimizer.learning_rate * 1},#4
          {'params': model.module.features[26].parameters(), 'lr': args.optimizer.learning_rate * 1000},#2
          {'params': model.module.features[28].parameters(), 'lr': args.optimizer.learning_rate * 1}]#4
    optimizer = t.optim.SGD(params,
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    

    # base_params = filter(lambda p: id(p) not in conv5_params + conv4_params, model.parameters())
    # params = [{'params': base_params},
    #       {'params': net.conv5.parameters(), 'lr': args.optimizer.learning_rate * 100},
    #       {'params': net.conv4.parameters(), 'lr': args.optimizer.learning_rate * 100}]
    # optimizer = t.optim.SGD(params, lr=lr, momentum=0.9)


    print(model.parameters())
    print(model)
    print(model.module.features[19])

    #optimizer = t.optim.AdamW(model.parameters(), lr=args.optimizer.learning_rate)
    # optimizer = t.optim.SGD(model.parameters(),
    #                         lr=args.optimizer.learning_rate,
    #                         momentum=args.optimizer.momentum,
    #                         weight_decay=args.optimizer.weight_decay)

    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)

    if args.eval:
        process.validate(test_loader, model, criterion, -1, monitors, args)
    else:  # training
        # if args.resume.path or args.pre_trained:
        #     logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
        #     top1, top5, _ = process.validate(val_loader, model, criterion,
        #                                      start_epoch - 1, monitors, args)
        #     perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = process.train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            v_top1, v_top5, v_loss = process.validate(val_loader, model, criterion, epoch, monitors, args)

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, model, criterion, -1, monitors, args)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please contact chenchen')


if __name__ == "__main__":
    main()
