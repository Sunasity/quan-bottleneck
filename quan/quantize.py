import torch
import torch.nn as nn
import copy

class QuantizeScheme(object):
    def __init__(self):
        self.scheme = 'google'
        self.subscheme = 'per_channel'
        self.is_scale_pow = True
        self.weight_bits = 8
        self.act_bits = 8

quan_scheme = QuantizeScheme()

class GoogleQuanParameter(object):
    def __init__(self, len, is_cuda=True):
        if is_cuda:
            self.FloatMax = torch.zeros([len], dtype=torch.float32).cuda()
            self.FloatMin = torch.zeros([len], dtype=torch.float32).cuda()
            self.Float2QuanScale = torch.zeros([len], dtype=torch.float32).cuda()
            self.Float2QaunBias = torch.zeros([len], dtype=torch.float32).cuda()
            self.QuantizeMax = torch.zeros([len], dtype=torch.float32).cuda()
            self.QuantizeMin = torch.zeros([len], dtype=torch.float32).cuda()

    def init(self, param):
        if quan_scheme.subscheme == 'per_layer':
            self.FloatMax = torch.max(param)
            self.FloatMin = torch.min(param)
        elif quan_scheme.subscheme == 'per_channel':
            self.FloatMax = torch.max(param.view(param.size()[0], -1), dim=1)[0]
            self.FloatMin = torch.min(param.view(param.size()[0], -1), dim=1)[0]
        TensorTwo = torch.tensor(2, dtype=torch.float32).cuda()
        self.QuantizeMax.fill_(torch.pow(TensorTwo, quan_scheme.weight_bits - 1))
        #self.QuantizeMin = 0
        self.Float2QuanScale = self.QuantizeMax / (self.FloatMax - self.FloatMin)
        if quan_scheme.is_scale_pow:
            self.Float2QuanScale = torch.pow(TensorTwo, torch.log2(self.Float2QuanScale).round_())
        #self.Float2QaunBias = torch.round(self.QuantizeMin - self.Float2QuanScale * self.FloatMin)

def WeightQuantize(Weight, Bias):
    if quan_scheme.scheme == 'google':
        if quan_scheme.subscheme == 'per_layer':            
            quan_param = GoogleQuanParameter(1)
            quan_param.init(Weight)
            with torch.no_grad():
                #float weight to quantized weight
                Weight.mul_(quan_param.Float2QuanScale)
                Weight.round_()
                #the two operations below are balanced
                #m.weight.add_(quan_param.Float2QaunBias)
                #quantized weight to float weight
                #m.weight.sub_(quan_param.Float2QaunBias)
                Weight.div_(quan_param.Float2QuanScale)
                if Bias is not None:
                    Bias.mul_(quan_param.Float2QuanScale)
                    Bias.round_()                
                    Bias.div_(quan_param.Float2QuanScale)
        elif quan_scheme.subscheme == 'per_channel':
            quan_param = GoogleQuanParameter(Weight.size()[0])
            quan_param.init(Weight)
            with torch.no_grad():
                #float weight to quantized weight
                Scale = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(quan_param.Float2QuanScale, 1), 2), 3)
                Weight.mul_(Scale)
                Weight.round_()
                Weight.div_(Scale)
                if Bias is not None:
                    Bias.mul_(quan_param.Float2QuanScale)
                    Bias.round_()                
                    Bias.div_(quan_param.Float2QuanScale)
    return Weight, Bias


def WeightQuantizeForModule(module):
    if quan_scheme.scheme == 'google':
        for m in module.modules():
            if isinstance(m, nn.Linear):
                if quan_scheme.subscheme == 'per_layer':            
                    quan_param = GoogleQuanParameter(1)
                    quan_param.init(m.weight)
                    with torch.no_grad():
                        #float weight to quantized weight
                        m.weight.mul_(quan_param.Float2QuanScale)
                        m.weight.round_()
                        #the two operations below are balanced
                        #m.weight.add_(quan_param.Float2QaunBias)
                        #quantized weight to float weight
                        #m.weight.sub_(quan_param.Float2QaunBias)
                        m.weight.div_(quan_param.Float2QuanScale)
                        if m.bias is not None:
                            m.bias.mul_(quan_param.Float2QuanScale)
                            m.bias.round_()
                            m.bias.div_(quan_param.Float2QuanScale)
                elif quan_scheme.subscheme == 'per_channel':
                    pass
    return None


def ActQuantization(input, FloatMax=6.0, FloatMin=-6.0, num_bits=quan_scheme.act_bits):
    QuantizeMax = 1.0*(torch.pow(torch.tensor(2, dtype=torch.float32), num_bits) - 1)
    #QuantizeMin = 0.0
    Float2QuanScale = QuantizeMax / (FloatMax - FloatMin)
    #Float2QuanBias is not used in training
    #Float2QaunBias = torch.round(QuantizeMin - Float2QuanScale * FloatMin)
    input = torch.mul(input, Float2QuanScale)
    input = DifferentialRound.apply(input)
    #the two operations below are balanced
    #input = torch.add(input, Float2QaunBias)
    #input = torch.sub(input, Float2QaunBias)
    input = torch.div(input, Float2QuanScale)
    return input

class DifferentialRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        i.round_()
        ctx.save_for_backward(i)
        return i

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def FoldData(real_conv, real_bn):
    delta = torch.sqrt(real_bn.running_var + real_bn.eps)
    FoldScale = torch.div(real_bn.weight, delta)
    FlodedWeight = nn.Parameter(torch.mul(real_conv.weight,
        torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(FoldScale, 1), 2), 3)))
    FoldedBias = real_bn.bias-torch.div(torch.mul(real_bn.weight, real_bn.running_mean), delta)
    return FlodedWeight, FoldedBias