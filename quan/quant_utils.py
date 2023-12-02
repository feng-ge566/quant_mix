import torch
import numpy as np

GATHER = True


LPR8B = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 14, 14, 15, 15, 16, 16, 17, 17, 22, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 40, 41, 41, 46, 46, 47, 47, 48, 48, 49, 49, 54, 54, 55, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 72, 72, 72, 73, 73, 73, 73, 78, 78, 78, 78, 79, 79, 79, 79, 112, 112, 112, 112, 113, 113, 113, 113, 118, 118, 118, 118, 119, 119, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127, -128, -128, -127, -127, -126, -126, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -120, -120, -120, -119, -119, -119, -119, -114, -114, -114, -114, -113, -113, -113, -113, -80, -80, -80, -80, -79, -79, -79, -79, -74, -74, -74, -74, -73, -73, -73, -73, -72, -72, -71, -71, -70, -70, -69, -69, -68, -68, -67, -67, -66, -66, -65, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -56, -55, -55, -50, -50, -49, -49, -48, -48, -47, -47, -42, -42, -41, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -24, -23, -23, -18, -18, -17, -17, -16, -16, -15, -15, -10, -10, -9, -9, -8, -7, -6, -5, -4, -3, -2, -1]
LPR8B_fast = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24, 24, 28, 28, 28, 28, 32, 32, 32, 32, 36, 36, 36, 36, 40, 40, 40, 40, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 56, 56, 56, 56, 60, 60, 60, 60, 64, 64, 64, 64, 64, 64, 64, 64, 68, 68, 68, 68, 68, 68, 68, 68, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 120, 120, 120, 120, 120, 120, 120, 120, 124, 124, 124, 124, 124, 124, 124, 124, -128, -128, -128, -128, -128, -128, -128, -128, -124, -124, -124, -124, -124, -124, -124, -124, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -72, -72, -72, -72, -72, -72, -72, -72, -68, -68, -68, -68, -68, -68, -68, -68, -64, -64, -64, -64, -60, -60, -60, -60, -56, -56, -56, -56, -56, -56, -56, -56, -48, -48, -48, -48, -48, -48, -48, -48, -40, -40, -40, -40, -36, -36, -36, -36, -32, -32, -32, -32, -28, -28, -28, -28, -24, -24, -24, -24, -24, -24, -24, -24, -16, -16, -16, -16, -16, -16, -16, -16, -8, -8, -8, -8, -4, -4, -4, -4]
#LPR8B_4bit = [0, 1, 1, 2, 2, 4, 4, 4, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,  -4, -4, -4, -2, -2, -2, -1, -1]
LPR8B_4bit = [0, 1, 1, 2, 2, 4, 4, 4, -4, -4, -4, -2, -2, -2, -1, -1]
#35ge34  35ge35
# def quantize(tensor: torch.Tensor, bit_width=8, scale=None):
#     tensor_copy = tensor.clone().detach()
#     mean = tensor_copy.mean()
#     tensor_copy = tensor_copy - mean
#     if scale is None:
#         scale = tensor_copy.view((-1,)).abs().sort()[0][-1]
#     step = scale / (2**(bit_width-1))
#     quantified_integer = torch.round(tensor_copy.data/step)
#     tensor.data = quantified_integer*step+mean
#     return tensor, (quantified_integer, scale, mean)

# TD:   this module set the input/output quantization of the module
#       first while register the pre forward hook, the register checks the params signature of input/output
#       after checking sig, then generate the quantify attr for each in/out
#       then register the pre-hook and post-hook
#       in the pre-hook and post-hook, the hook quantify the input and output,
#       also set the quantify attr for debug usage
class ModuleQuantifyAttr:
    def __init__(self, input_attrs=None, output_attrs=None):
        self.input_attrs = input_attrs
        self.output_attrs = output_attrs


class TensorQuantifyAttr:
    def __init__(self, quantified=None, bit_width=None, scale=None, ori_data=None, int_data=None, mean=None):
        super(TensorQuantifyAttr, self).__init__()
        self.quantified = quantified
        self.bit_width = bit_width
        self.scale = scale
        self.ori_data = ori_data
        self.int_data = int_data
        self.mean = mean
        self.child = tuple()



def quantize(tensor: torch.Tensor, bit_width, scale=None, approxweight=None):
    
    device = torch.device("cuda") # GPU
    ori_data = tensor.data.clone().to(device)
    qut_data = tensor.data.clone().to(device)
    # mean = ori_data.mean()
    # ori_data = ori_data - mean
    # if str(mean.cpu().numpy()) == 'nan':
    #     hook = 0
    import ipdb
    ipdb.set_trace()
    if scale is None:
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        scale = ori_data.view((-1,)).abs_()
        scale = scale.sort()[0][int(0.99*scale.shape[-1])]
        # scale = torch.log2(scale)-(bit_width-1)
        scale = torch.ceil_(torch.log2_(scale))-(bit_width-1)
        # scale = torch.floor(torch.log2(scale))-(bit_width-1)
    else:
        scale = np.array(scale, dtype=float)
        scale = torch.from_numpy(scale)
        scale = scale.cuda()
        # print("scale is", scale)
    

    data_range = torch.pow(2, scale+(bit_width-1))
    data_step = torch.pow(2, scale)
    # data_range = 2**(scale+(bit_width-1))
    # data_step = 2**scale
    qut_data.clamp_(-data_range, data_range-data_step)
    # step = scale / (2**(bit_width-1))
    # quantizied_data = torch.floor_(ori_data/data_step)
    quantizied_data = torch.round_(qut_data/data_step + 1e-6)
    if approxweight is None:
        # print("Transform dat into approximate multipiler!")
        quan_shape = quantizied_data.shape
        dim_data = torch.tensor(quantizied_data.reshape(-1).clone(), dtype=torch.uint8).cuda()
        dim_data = torch.tensor(dim_data, dtype=torch.long).cuda()
        if bit_width == 8:
            label_data = torch.tensor(LPR8B, dtype=float).cuda() # LPR8B or LPR8B_fast
        elif bit_width == 4:
            label_data = torch.tensor(LPR8B_4bit, dtype=float).cuda() # LPR8B_4bit
        dim_data = torch.gather(label_data, dim=-1, index=dim_data)
        quantizied_data = dim_data.reshape(quan_shape).float()

    if GATHER:
        quant = TensorQuantifyAttr(quantified=True,
                                   bit_width=bit_width,
                                   scale=scale.cuda(),
                                   ori_data=tensor.data.clone(),
                                   int_data=quantizied_data.detach().cuda(),
                                   mean=None)
    else:
        quant = TensorQuantifyAttr(quantified=True,
                                   bit_width=bit_width,
                                   scale=scale.cuda(),
                                   ori_data=tensor.data.clone(),
                                   int_data=None,
                                   mean=None)
    tensor.data = (quantizied_data*data_step)
    if hasattr(tensor, 'quant_attr'):
        del tensor.quant_attr.ori_data
        delattr(tensor, 'quant_attr')
    setattr(tensor, 'quant_attr', quant)
    del ori_data
    del qut_data
    del quantizied_data
    del scale
    return tensor

