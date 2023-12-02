import torch as t
from .quantizer.lsq import LsqQuanSRP,LsqQuanSparseSRP_Kai, LsqQuanSRP_acti

class QuanSparseLinearSRP_Kai(t.nn.Linear):
    def __init__(self, in_features, out_features, sparsity=0.0, TIN=8, group_size=2, bias=True,weight_bit_width=2,activation_bit_width=8):
        super().__init__(in_features, out_features, bias)
        self.use_bias = bias
        
        self.sparsity = sparsity
        self.TIN = TIN
        self.group_size = group_size
        self.weight_bit_width = weight_bit_width
        self.activation_bit_width = activation_bit_width
        self.quan_w_fn = LsqQuanSparseSRP_Kai(bit=weight_bit_width,weight= self.weight,
                                              sparsity=self.sparsity, 
                                              TIN = self.TIN,
                                              group_size = self.group_size,
                                              per_channel=False)
        self.quan_a_fn = LsqQuanSRP_acti(bit=self.activation_bit_width, per_channel=False)
        
        self.quan_w_fn.init_from(self.weight)
        #print("bit_width: ",self.weight_bit_width,", sparsity",self.sparsity)
        #print("weight_bit_width: ",self.weight_bit_width,",","activation_bit_width: ",self.activation_bit_width,", sparsity: ",self.sparsity,", Tin: ", self.TIN, ", Group: ", self.group_size)
        if bias:
            self.quan_b_fn = LsqQuanSRP(bit=8, per_channel=False)
            self.quan_b_fn.init_from(self.bias)
        else:
            self.bias = None
            self.quan_b_fn = None
        self.size = self._calculate_coded_complexity_MB()

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight).cuda()
        quantized_act = self.quan_a_fn(x).cuda()
        if(self.use_bias):
            quantized_bias = self.quan_b_fn(self.bias).cuda()
        else:
            quantized_bias = None
        return t.nn.functional.linear(quantized_act, quantized_weight, quantized_bias)

    def _init_weight(self):
        self.quan_w_fn.init_from(self.weight)
        if self.use_bias:
            self.quan_b_fn.init_from(self.bias)
    def _reset_mask(self):
        self.quan_w_fn._reset_mask()
    def _unset_mask(self):
        self.quan_w_fn._unset_mask()
    def _set_mask(self):
        self.quan_w_fn._set_mask(self.weight)
    def _calculate_complexity(self):
        # print(self.weight_bit_width,self.sparsity,self.weight.numel())
        return self.weight.numel() * self.weight_bit_width/8*(1-self.sparsity)
    def _calculate_coded_complexity(self):
        #print(self.weight_bit_width,self.sparsity,self.weight.numel())
        if(self.sparsity==0):
            return self._calculate_complexity()
        else:
            ch = self.weight.shape[1]
            if(ch%8!=0):
                ch = ch + 8 - ch%8
                print(ch)
            return self._calculate_complexity()+self.weight.numel()/self.weight.shape[1]*ch /8
    def _calculate_coded_complexity_MB(self):
        return self._calculate_coded_complexity()/1024/1024
    

class QuanSparseConv2dSRP_Kai(t.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 sparsity=0.0, TIN=8, group_size=2,
                 stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros',weight_bit_width=2,activation_bit_width=8):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode)
        self.use_bias = bias
        self.sparsity = sparsity
        self.TIN = TIN
        self.group_size = group_size
        self.weight_bit_width = weight_bit_width
        self.activation_bit_width = activation_bit_width
        self.quan_w_fn = LsqQuanSparseSRP_Kai(bit=self.weight_bit_width,weight= self.weight,
                                                  sparsity=self.sparsity, 
                                                  TIN = self.TIN,
                                                  group_size = self.group_size,
                                                  per_channel=False)
        self.quan_a_fn = LsqQuanSRP_acti(bit=self.activation_bit_width, per_channel=False)

        #print("weight_bit_width: ",self.weight_bit_width,", ","activation_bit_width: ",self.activation_bit_width,", sparsity: ",self.sparsity,", Tin: ", self.TIN, ", Group: ", self.group_size)

        # self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(self.weight)
        if bias:
            # change the bit_width type for bias
            self.quan_b_fn = LsqQuanSRP(self.weight_bit_width, per_channel=False)
            self.bias = t.nn.Parameter(self.bias)
        else:
            self.bias = None
            self.quan_b_fn = None
            # self.bias = t.nn.Parameter(bias)
        self.size = self._calculate_coded_complexity_MB()

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight).cuda()
        quantized_act = self.quan_a_fn(x)
        if(self.use_bias):
            quantized_bias = self.quan_b_fn(self.bias)
        else:
            quantized_bias = None
        return self._conv_forward(quantized_act, quantized_weight,quantized_bias)
    def _init_weight(self):
        self.quan_w_fn.init_from(self.weight)
        if self.use_bias:
            self.quan_b_fn.init_from(self.bias)
    def _reset_mask(self):
        self.quan_w_fn._reset_mask()
    def _unset_mask(self):
        self.quan_w_fn._unset_mask()
    def _set_mask(self):
        self.quan_w_fn._set_mask(self.weight)
    def _calculate_complexity(self):
        # print(self.weight_bit_width,self.sparsity,self.weight.numel())
        return self.weight.numel() * self.weight_bit_width/8*(1-self.sparsity)
    def _calculate_coded_complexity(self):
        #print(self.weight_bit_width,self.sparsity,self.weight.numel())
        if(self.sparsity==0):
            return self._calculate_complexity()
        else:
            ch = self.weight.shape[1]
            if(ch%8!=0):
                ch = ch + 8 - ch%8
                print(ch)
            return self._calculate_complexity()+self.weight.numel()/self.weight.shape[1]*ch /8
    def _calculate_coded_complexity_MB(self):
        return self._calculate_coded_complexity()/1024/1024

# QuanModuleMapping = {
#     t.nn.Conv2d: QuanConv2d,
#     t.nn.Linear: QuanLinear
# }
