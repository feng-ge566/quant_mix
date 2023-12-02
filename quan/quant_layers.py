import torch as t
from .quantizer.lsq import LsqQuanSRP,LsqQuanSparseSRP_Kai
# class QuanConv2d(t.nn.Conv2d):
#     def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
#         assert type(m) == t.nn.Conv2d
#         super().__init__(m.in_channels, m.out_channels, m.kernel_size,
#                          stride=m.stride,
#                          padding=m.padding,
#                          dilation=m.dilation,
#                          groups=m.groups,
#                          bias=True if m.bias is not None else False,
#                          padding_mode=m.padding_mode)
#         self.quan_w_fn = quan_w_fn
#         self.quan_a_fn = quan_a_fn

#         self.weight = t.nn.Parameter(m.weight.detach())
#         self.quan_w_fn.init_from(m.weight)
#         if m.bias is not None:
#             self.bias = t.nn.Parameter(m.bias.detach())

#     def forward(self, x):
#         quantized_weight = self.quan_w_fn(self.weight)
#         quantized_act = self.quan_a_fn(x)
#         return self._conv_forward(quantized_act, quantized_weight)

class QuanConv2d(t.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros',weight_bit_width=8,act_bit_width=8,bias_bit_width=8):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode)
        self.quan_w_fn = LsqQuanSRP(bit=weight_bit_width, per_channel=False)
        self.quan_a_fn = LsqQuanSRP(bit=act_bit_width, per_channel=False)
        self.use_bias = bias

        # self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(self.weight)
        if bias:
            self.quan_b_fn = LsqQuanSRP(bit=bias_bit_width, per_channel=False)
        else:
            self.bias = None
            self.quan_b_fn = None
            # self.bias = t.nn.Parameter(bias)

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
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


class QuanLinear(t.nn.Linear):
    def __init__(self, in_features, out_features, bias=True,weight_bit_width=8,act_bit_width=8,bias_bit_width=8):
        super().__init__(in_features, out_features, bias)
        self.quan_w_fn = LsqQuanSRP(bit=weight_bit_width, per_channel=False)
        self.quan_a_fn = LsqQuanSRP(bit=act_bit_width, per_channel=False)
        self.use_bias = bias
        self.quan_w_fn.init_from(self.weight)

        if bias:
            self.quan_b_fn = LsqQuanSRP(bit=bias_bit_width, per_channel=False)
        else:
            self.bias = None
            self.quan_b_fn = None

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        if(self.use_bias):
            quantized_bias = self.quan_b_fn(self.bias)
        else:
            quantized_bias = None
        return t.nn.functional.linear(quantized_act, quantized_weight, quantized_bias)

    def _init_weight(self):
        self.quan_w_fn.init_from(self.weight)
        if self.use_bias:
            self.quan_b_fn.init_from(self.bias)


QuanModuleMapping = {
    t.nn.Conv2d: QuanConv2d,
    t.nn.Linear: QuanLinear
}
