import torch as t

from .quantizer import Quantizer
import math

LPR8B = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 14, 14, 15, 15, 16, 16, 17, 17, 22, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 40, 41, 41, 46, 46, 47, 47, 48, 48, 49, 49, 54, 54, 55, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 72, 72, 72, 73, 73, 73, 73, 78, 78, 78, 78, 79, 79, 79, 79, 112, 112, 112, 112, 113, 113, 113, 113, 118, 118, 118, 118, 119, 119, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127, -128, -128, -127, -127, -126, -126, -125, -125, -124, -124, -123, -123, -122, -122, -121, -121, -120, -120, -120, -120, -119, -119, -119, -119, -114, -114, -114, -114, -113, -113, -113, -113, -80, -80, -80, -80, -79, -79, -79, -79, -74, -74, -74, -74, -73, -73, -73, -73, -72, -72, -71, -71, -70, -70, -69, -69, -68, -68, -67, -67, -66, -66, -65, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -56, -55, -55, -50, -50, -49, -49, -48, -48, -47, -47, -42, -42, -41, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -24, -23, -23, -18, -18, -17, -17, -16, -16, -15, -15, -10, -10, -9, -9, -8, -7, -6, -5, -4, -3, -2, -1]
LPR8B_fast = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24, 24, 28, 28, 28, 28, 32, 32, 32, 32, 36, 36, 36, 36, 40, 40, 40, 40, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 56, 56, 56, 56, 60, 60, 60, 60, 64, 64, 64, 64, 64, 64, 64, 64, 68, 68, 68, 68, 68, 68, 68, 68, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 120, 120, 120, 120, 120, 120, 120, 120, 124, 124, 124, 124, 124, 124, 124, 124, -128, -128, -128, -128, -128, -128, -128, -128, -124, -124, -124, -124, -124, -124, -124, -124, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -72, -72, -72, -72, -72, -72, -72, -72, -68, -68, -68, -68, -68, -68, -68, -68, -64, -64, -64, -64, -60, -60, -60, -60, -56, -56, -56, -56, -56, -56, -56, -56, -48, -48, -48, -48, -48, -48, -48, -48, -40, -40, -40, -40, -36, -36, -36, -36, -32, -32, -32, -32, -28, -28, -28, -28, -24, -24, -24, -24, -24, -24, -24, -24, -16, -16, -16, -16, -16, -16, -16, -16, -8, -8, -8, -8, -4, -4, -4, -4]
#LPR8B_4bit = [0, 1, 1, 2, 2, 4, 4, 4, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,  -4, -4, -4, -2, -2, -2, -1, -1]
LPR8B_4bit = [0, 1, 1, 2, 2, 4, 4, 4, -4, -4, -4, -2, -2, -2, -1, -1]
CE_6bit = [0, 1, 2, 2, 4, 4, 6, 7, 8, 9, 10, 10, 12, 12, 14, 15, 16, 16, 17, 17, 18, 18, 18, 18, 28, 28, 28, 28, 30, 30, 31, 31, -32, -32, -31, -31, -30, -30, -30, -30, -20, -20, -20, -20, -18, -18, -17, -17, -16, -15, -14, -14, -12, -12, -10, -9, -8, -7, -6, -6, -4, -4, -2, -1]

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def one_or_minus_one(x):
    minus_one = t.zeros_like(x)-1
    one = t.ones_like(x)
    x_cp = x.clone()
    x_cp = t.where(x_cp >= 0, one, x_cp)
    x_cp = t.where(x_cp < 0, minus_one, x_cp)
    x_cp = (x_cp-x).detach()+x
    return x_cp

def ac_one_or_minus_one(x):
    minus_one = t.zeros_like(x)-1
    one = t.ones_like(x)
    x_cp = x.clone()
    x_cp = t.where(x_cp >= 0, one, x_cp)
    x_cp = t.where(x_cp < 0, minus_one, x_cp)
    x_cp = (x_cp-x).detach()+x
    return x_cp



def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def hold_grad(x_rep, x):
    y = x_rep
    y_grad = x
    return (y - y_grad).detach() + y_grad

class LsqQuanSRP_acti(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            elif (bit==1):
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.precision = bit
        self.s = t.nn.Parameter(t.ones([]))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        ss_grad = self.s.clone().detach().requires_grad_(True)
        ss_grad = t.clamp(ss_grad, 0.0001, 1)
        s_scale = grad_scale(ss_grad, s_grad_scale)
        # s_scale = grad_scale(ss_scale, s_grad_scale)
        #print('the s_grad_scale is :', s_grad_scale)
        # print('the s data is :', self.s)
        
        # s_scale = grad_scale(self.s, s_grad_scale)
        #print('the first scale is :', s_scale)
        final_scale = s_scale.clone()
        # final_scale = t.floor(final_scale.log2())
        final_scale = floor_pass(final_scale.log2())
        #print('the first final scale is :', final_scale)
        s_scale = s_scale*0 +  (2 ** final_scale)
        #print('final scale is :', s_scale)
        # s_scale = round_pass(final_scale)
        x = x / s_scale
        x = x.cuda()
        
        #print('raw data are :', x)
        quan_shape = x.shape
        if (self.precision==1):
            #x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = ac_one_or_minus_one(x)
        else :
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            # x = round_pass(x)
            
            # dim_data = t.tensor(x.reshape(-1), dtype=t.long).clone().detach().cuda()
            
            x_cp = x.clone()
            x_cp = round_pass(x_cp)

            x_cp= t.tensor(x_cp, dtype = t.float).cuda()  
            #x = round_pass(x_cp)
            x = (x_cp - x).detach() + x
        x = x * s_scale
        return x

class LsqQuanSRP(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            elif (bit==1):
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.precision = bit
        self.s = t.nn.Parameter(t.ones([]))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        ss_grad = self.s.clone().detach().requires_grad_(True)
        ss_grad = t.clamp(ss_grad, 0.0001, 1)
        s_scale = grad_scale(ss_grad, s_grad_scale)
        # s_scale = grad_scale(ss_scale, s_grad_scale)
        #print('the s_grad_scale is :', s_grad_scale)
        # print('the s data is :', self.s)
        
        # s_scale = grad_scale(self.s, s_grad_scale)
        #print('the first scale is :', s_scale)
        final_scale = s_scale.clone()
        # final_scale = t.floor(final_scale.log2())
        final_scale = floor_pass(final_scale.log2())
        #print('the first final scale is :', final_scale)
        s_scale = s_scale*0 +  (2 ** final_scale)
        #print('final scale is :', s_scale)
        # s_scale = round_pass(final_scale)
        x = x / s_scale
        x = x.cuda()
        
        #print('raw data are :', x)
        quan_shape = x.shape
        if (self.precision==1):
            #x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = ac_one_or_minus_one(x)
        elif (self.precision == 8):
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            # x = round_pass(x)
            
            # dim_data = t.tensor(x.reshape(-1), dtype=t.long).clone().detach().cuda()
            
            x_cp = x.clone()
            x_cp = round_pass(x_cp)
            dim_data = x_cp.view(-1).long().clone().detach().cuda()

            label_data = t.tensor(LPR8B, dtype=t.long).cuda()
# for v in label_data:
#     if v < 0:
#         x = 256 + v
            #print('data dim_date are :', dim_data) 
            
            data_x = t.take(label_data,  dim_data).cuda() 
            #print('data x mapping are :', x) 
              
            x_cp = data_x.reshape(quan_shape).cuda()   
            # x = (x_cp - x).detach() + x
            x_cp= t.tensor(x_cp, dtype = t.float).cuda()  
            #x = round_pass(x_cp)
            x = (x_cp - x).detach() + x

        #x = t.clamp(x, self.thd_neg, self.thd_pos)
        #x = round_pass(x)
        elif (self.precision == 6):
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            # x = round_pass(x)
            
            # dim_data = t.tensor(x.reshape(-1), dtype=t.long).clone().detach().cuda()
            x_cp = x.clone()
            x_cp = round_pass(x_cp)
            dim_data = x_cp.view(-1).long().clone().detach().cuda()

            label_data = t.tensor(CE_6bit, dtype=t.long).cuda()
# for v in label_data:
#     if v < 0:
#         x = 256 + v
            #print('data dim_date are :', dim_data) 
            
            data_x = t.take(label_data,  dim_data).cuda() 
            #print('data x mapping are :', x) 
              
            x_cp = data_x.reshape(quan_shape).cuda()   
            # x = (x_cp - x).detach() + x
            x_cp= t.tensor(x_cp, dtype = t.float).cuda() 
            x = (x_cp - x).detach() + x
            #x = round_pass(x_cp)
        elif (self.precision == 4):
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            # x = round_pass(x)
            
            # dim_data = t.tensor(x.reshape(-1), dtype=t.long).clone().detach().cuda()
            x_cp = x.clone()
            x_cp = round_pass(x_cp)
            dim_data = x_cp.view(-1).long().clone().detach().cuda()

            label_data = t.tensor(LPR8B_4bit, dtype=t.long).cuda()
# for v in label_data:
#     if v < 0:
#         x = 256 + v
            #print('data dim_date are :', dim_data) 
            
            data_x = t.take(label_data,  dim_data).cuda() 
            #print('data x mapping are :', x) 
              
            x_cp = data_x.reshape(quan_shape).cuda()
            x_cp= t.tensor(x_cp, dtype = t.float).cuda()   
            x = (x_cp - x).detach() + x
            #x = round_pass(x_cp)
        else:
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
        x = x * s_scale
        return x

#SRP:scale round pass





class LsqQuanSparseSRP_Kai(Quantizer):
    def __init__(self, bit, sparsity=0.5, TIN=8, group_size=2, weight:t.Tensor=None,all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            elif (bit==1):
                self.thd_neg = -1
                self.thd_pos = 1               
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones([]))
        self._mask_set_flag_ = False
        self.register_buffer('mask',t.zeros_like(weight))
        self.sparsity = sparsity
        self.TIN=TIN
        self.group_size=group_size
        self.precision=bit
        #print("sparsity is: ",self.sparsity)

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        ss_grad = self.s.clone().detach().requires_grad_(True)
        
        ss_grad = t.clamp(ss_grad, 0.0001, 1)
        s_scale = grad_scale(ss_grad, s_grad_scale)
        # s_scale = grad_scale(ss_scale, s_grad_scale)
        #print('the s_grad_scale is :', s_grad_scale)
        # print('the s data is :', self.s)
        
        # s_scale = grad_scale(self.s, s_grad_scale)
        #print('the first scale is :', s_scale)
        final_scale = s_scale.clone()
        # final_scale = t.floor(final_scale.log2())
        final_scale = floor_pass(final_scale.log2())
        #print('the first final scale is :', final_scale)
        s_scale = s_scale*0 +  (2 ** final_scale)
        #print('final scale is :', s_scale)
        # s_scale = round_pass(final_scale)
        x = x / s_scale
        x = x.cuda()
        #print('raw data are :', x)
        quan_shape = x.shape
        if (self.precision==1):
            #x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = one_or_minus_one(x)
        elif (self.precision == 8):
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            # x = round_pass(x)
            
            # dim_data = t.tensor(x.reshape(-1), dtype=t.long).clone().detach().cuda()
            x_cp = x.clone()
            x_cp = round_pass(x_cp)
            dim_data = x_cp.view(-1).long().clone().detach().cuda()

            label_data = t.tensor(LPR8B, dtype=t.long).cuda()
# for v in label_data:
#     if v < 0:
#         x = 256 + v
            #print('data dim_date are :', dim_data) 
            
            data_x = t.take(label_data,  dim_data).cuda() 
            #print('data x mapping are :', x) 
              
            x_cp = data_x.reshape(quan_shape).cuda()   
            #x = (x_cp - x).detach() + x
            x_cp= t.tensor(x_cp, dtype = t.float).cuda() 
            x = (x_cp - x).detach() + x
        elif (self.precision == 6):
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            # x = round_pass(x)
            
            # dim_data = t.tensor(x.reshape(-1), dtype=t.long).clone().detach().cuda()
            x_cp = x.clone()
            x_cp = round_pass(x_cp)
            dim_data = x_cp.view(-1).long().clone().detach().cuda()

            label_data = t.tensor(CE_6bit, dtype=t.long).cuda()
# for v in label_data:
#     if v < 0:
#         x = 256 + v
            #print('data dim_date are :', dim_data) 
            
            data_x = t.take(label_data,  dim_data).cuda() 
            #print('data x mapping are :', x) 
              
            x_cp = data_x.reshape(quan_shape).cuda()   
            # x = (x_cp - x).detach() + x
            x_cp= t.tensor(x_cp, dtype = t.float).cuda()
            x = (x_cp - x).detach() + x
        elif (self.precision == 4):
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            # x = round_pass(x)
            
            # dim_data = t.tensor(x.reshape(-1), dtype=t.long).clone().detach().cuda()
            x_cp = x.clone()
            x_cp = round_pass(x_cp)
            dim_data = x_cp.view(-1).long().clone().detach().cuda()

            label_data = t.tensor(LPR8B_4bit, dtype=t.long).cuda()
# for v in label_data:
#     if v < 0:
#         x = 256 + v
            #print('data dim_date are :', dim_data) 
            
            data_x = t.take(label_data,  dim_data).cuda() 
            #print('data x mapping are :', x) 
              
            x_cp = data_x.reshape(quan_shape).cuda()   
            #x = (x_cp - x).detach() + x
            x_cp= t.tensor(x_cp, dtype = t.float).cuda()
            x = (x_cp - x).detach() + x
        else:
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)

        x = x * s_scale
        return x
    # def _reset_mask(self):
    #     self.mask.zero_()
    #     self._mask_set_flag_ = False
    
    # def _calculate_zeros(self,tensor:t.Tensor):
    #     return (tensor.numel() - tensor.nonzero().shape[0])/tensor.numel()
    
    # def _unset_mask(self):
    #     self._mask_set_flag_ = True

    # def _set_mask(self,tensor:t.Tensor):
    #     ptensor = tensor.clone().detach()
    #     final_scale = self.s.clone().detach()
    #     final_scale = t.floor(final_scale.log2())
    #     s_scale = (2 ** final_scale)
    #     ptensor = ptensor / s_scale
    #     ptensor = t.clamp(ptensor, self.thd_neg, self.thd_pos)
    #     ptensor = round_pass(ptensor)
    #     self.mask = multi_channel_fine_grained_prune_mask_TIN8(tensor=ptensor, 
    #                                                            sparsity=self.sparsity, 
    #                                                            TIN=self.TIN, 
    #                                                            group_size=self.group_size)


# def multi_channel_fine_grained_prune_mask_TIN8(tensor: t.Tensor, sparsity : float,TIN:int=8,group_size=2) -> t.Tensor:
#     #TIN = 8
#     MULT_TIN = TIN*group_size
#     #print(MULT_TIN)
#     device = GPU
#     #print(tensor.shape)
#     from einops import rearrange
#     sparsity = min(max(0.0, sparsity), 1.0)
#     if sparsity == 1.0:
#         tensor.zero_()
#         return t.zeros_like(tensor)
#     elif sparsity == 0.0:
#         return t.ones_like(tensor)
#     assert tensor.dim() >= 2, "Only weights can be pruned"
#     ptensor = tensor.data.clone()
#     # ptensor = quantize_for_mask(ptensor)
#     N, C, H, W = 0, 0, 0, 0
#     if tensor.dim() == 4:
#         N, C, H, W = ptensor.shape
#         try:
#             from einops import rearrange
#             ptensor = rearrange(ptensor, 'N C H W -> (N H W) C')
#         except(ImportError):
#             raise ImportError('Please install einops first')
#             #TODO: add a function to replace einops
#     elif tensor.dim() == 2:
#         pass
#     else:
#         print("dim is not 2 or 4")
#         raise NotImplementedError
#     # num_elements = tensor[2].numel()
#     num_elements = ptensor.shape[1]
#     dim_less = math.ceil(num_elements/MULT_TIN)*MULT_TIN-num_elements
#     # r_times = math.ceil(num_elements/MULT_TIN)
#     # dim_left = TIN-dim_less
#     ptensor_appended = t.cat([ptensor, t.zeros((ptensor.shape[0],dim_less),device=device)], dim=1)
#     r_times = math.ceil(num_elements/MULT_TIN)
#     ptensor_appended = rearrange(ptensor_appended, 'N (r g c) -> (N r) g c', N=ptensor_appended.shape[0], r=r_times, c = TIN, g=group_size)
#     # Step 1: calculate the #zeros (please use round())
#     # num_zeros = round(num_elements * sparsity)
#     num_zeros = round(TIN * sparsity)
#     # sub_masks = ()
#     if num_zeros == 0:
#         return t.ones_like(tensor)
#     # print('appended tensor is',ptensor_appended)
#     ptensor_appended_norm = t.norm(ptensor_appended, p=2,dim=-2)
#     # print('normed',ptensor_appended_norm)
#     ptensor_appended_norm = ptensor_appended_norm.abs()
#     mask = t.zeros_like(ptensor_appended_norm)
#     thresholds = ptensor_appended_norm.kthvalue(num_zeros,dim=1).values
#     mask = ptensor_appended_norm.gt(thresholds.unsqueeze(1))
#     mask = mask.repeat(1,group_size)
#     # mask = t.cat([mask,mask],dim=-1)
#     # for channel in ptensor_appended:
#     #     # Step 2: calculate the importance of weight
#     #     importance = channel.abs()
#     #     # Step 3: calculate the pruning threshold
#     #     threshold = importance.view(-1).kthvalue(num_zeros).values
#     #     # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
#     #     mask = t.gt(importance, threshold)
#     #     sub_masks += (mask,)
#     # mask = t.stack(sub_masks, dim=0)
#     mask = rearrange(mask, '(N r) c -> N (r c)',N=ptensor.shape[0], r=r_times, c = MULT_TIN)[:, :num_elements]
#     if tensor.dim() == 4:
#         try:
#             from einops import rearrange
#             mask = rearrange(mask, '(N H W) C -> N C H W', N=N, H=H, W=W)
#         except(ImportError):
#             raise ImportError('Please install einops first')
#             #TODO: add a function to replace einops
#     return mask