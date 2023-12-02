import logging
from .my_vgg_s0_4bit import vgg16 as vgg16_s0_4bit
from .my_vgg_s0_2bit import vgg16 as vgg16_s0_2bit
from .my_vgg_s0_1bit import vgg16 as vgg16_s0_1bit
from .my_vgg19_s0_2bit import vgg19 as vgg19_2bit
def create_model(args,select_dict=None):
    logger = logging.getLogger()

    model = None

    if args.arch == 'vgg16_s0_4bit':
        model = vgg16_s0_4bit(pretrained=args.pre_trained)
    elif args.arch == 'vgg16_s0_2bit':
        model = vgg16_s0_2bit(pretrained=args.pre_trained)
    elif args.arch == 'vgg16_s0_1bit':
        model = vgg16_s0_1bit(pretrained=args.pre_trained)
    elif args.arch == 'vgg19_2bit':
        model = vgg19_2bit(pretrained=args.pre_trained)        

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)

    msg = 'Created `%s` model for `imagenet` dataset' % (args.arch)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    logger.info(msg)

    return model
