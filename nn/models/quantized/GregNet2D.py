# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class GregNet2D(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(GregNet2D, self).__init__()
        self.module_0 = py_nndct.nn.Input() #GregNet2D::input_0(GregNet2D::nndct_input_0)
        self.module_1 = py_nndct.nn.Module('nndct_shape') #GregNet2D::GregNet2D/752(GregNet2D::nndct_shape_1)
        self.module_2 = py_nndct.nn.Module('nndct_shape') #GregNet2D::GregNet2D/757(GregNet2D::nndct_shape_2)
        self.module_3 = py_nndct.nn.Module('nndct_shape') #GregNet2D::GregNet2D/762(GregNet2D::nndct_shape_3)
        self.module_4 = py_nndct.nn.Module('nndct_reshape') #GregNet2D::GregNet2D/ret.9(GregNet2D::nndct_reshape_4)
        self.module_5 = py_nndct.nn.Module('nndct_cast') #GregNet2D::GregNet2D/775(GregNet2D::nndct_cast_5)
        self.module_6 = py_nndct.nn.BatchNorm(num_features=3, eps=0.0, momentum=0.1) #GregNet2D::GregNet2D/BatchNorm2d[batchnorm]/ret.11(GregNet2D::nndct_batch_norm_6)
        self.module_7 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[16, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #GregNet2D::GregNet2D/Conv2d[conv1]/ret.13(GregNet2D::nndct_conv2d_7)
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #GregNet2D::GregNet2D/ReLU[relu]/ret.15(GregNet2D::nndct_relu_8)
        self.module_9 = py_nndct.nn.MaxPool2d(kernel_size=[4, 1], stride=[4, 1], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #GregNet2D::GregNet2D/MaxPool2d[maxpool]/818(GregNet2D::nndct_maxpool_9)
        self.module_10 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[8, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #GregNet2D::GregNet2D/Conv2d[conv2]/ret.17(GregNet2D::nndct_conv2d_10)
        self.module_11 = py_nndct.nn.ReLU(inplace=False) #GregNet2D::GregNet2D/ReLU[relu]/ret.19(GregNet2D::nndct_relu_11)
        self.module_12 = py_nndct.nn.MaxPool2d(kernel_size=[4, 1], stride=[4, 1], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #GregNet2D::GregNet2D/MaxPool2d[maxpool]/856(GregNet2D::nndct_maxpool_12)
        self.module_13 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[4, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #GregNet2D::GregNet2D/Conv2d[conv3]/ret.21(GregNet2D::nndct_conv2d_13)
        self.module_14 = py_nndct.nn.ReLU(inplace=False) #GregNet2D::GregNet2D/ReLU[relu]/ret.23(GregNet2D::nndct_relu_14)
        self.module_15 = py_nndct.nn.MaxPool2d(kernel_size=[4, 1], stride=[4, 1], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #GregNet2D::GregNet2D/MaxPool2d[maxpool]/894(GregNet2D::nndct_maxpool_15)
        self.module_16 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[8, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #GregNet2D::GregNet2D/Conv2d[conv4]/ret.25(GregNet2D::nndct_conv2d_16)
        self.module_17 = py_nndct.nn.ReLU(inplace=False) #GregNet2D::GregNet2D/ReLU[relu]/ret.27(GregNet2D::nndct_relu_17)
        self.module_18 = py_nndct.nn.MaxPool2d(kernel_size=[4, 1], stride=[4, 1], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #GregNet2D::GregNet2D/MaxPool2d[maxpool]/932(GregNet2D::nndct_maxpool_18)
        self.module_19 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[16, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #GregNet2D::GregNet2D/Conv2d[conv5]/ret.29(GregNet2D::nndct_conv2d_19)
        self.module_20 = py_nndct.nn.ReLU(inplace=False) #GregNet2D::GregNet2D/ReLU[relu]/ret.31(GregNet2D::nndct_relu_20)
        self.module_21 = py_nndct.nn.MaxPool2d(kernel_size=[4, 1], stride=[4, 1], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #GregNet2D::GregNet2D/MaxPool2d[maxpool]/970(GregNet2D::nndct_maxpool_21)
        self.module_22 = py_nndct.nn.Module('nndct_flatten') #GregNet2D::GregNet2D/Flatten[flatten]/ret.33(GregNet2D::nndct_flatten_22)
        self.module_23 = py_nndct.nn.Linear(in_features=37632, out_features=128, bias=True) #GregNet2D::GregNet2D/Linear[fc1]/ret.35(GregNet2D::nndct_dense_23)
        self.module_24 = py_nndct.nn.ReLU(inplace=False) #GregNet2D::GregNet2D/ReLU[relu]/ret.37(GregNet2D::nndct_relu_24)
        self.module_25 = py_nndct.nn.Linear(in_features=128, out_features=1, bias=True) #GregNet2D::GregNet2D/Linear[fc2]/ret(GregNet2D::nndct_dense_25)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_1 = self.module_1(input=output_module_0, dim=0)
        output_module_2 = self.module_2(input=output_module_0, dim=1)
        output_module_3 = self.module_3(input=output_module_0, dim=-1)
        output_module_4 = self.module_4(input=output_module_0, shape=[output_module_1,output_module_2,output_module_3,1])
        output_module_4 = self.module_5(input=output_module_4, device='cpu', dtype=torch.float, non_blocking=False, copy=False)
        output_module_4 = self.module_6(output_module_4)
        output_module_4 = self.module_7(output_module_4)
        output_module_4 = self.module_8(output_module_4)
        output_module_4 = self.module_9(output_module_4)
        output_module_4 = self.module_10(output_module_4)
        output_module_4 = self.module_11(output_module_4)
        output_module_4 = self.module_12(output_module_4)
        output_module_4 = self.module_13(output_module_4)
        output_module_4 = self.module_14(output_module_4)
        output_module_4 = self.module_15(output_module_4)
        output_module_4 = self.module_16(output_module_4)
        output_module_4 = self.module_17(output_module_4)
        output_module_4 = self.module_18(output_module_4)
        output_module_4 = self.module_19(output_module_4)
        output_module_4 = self.module_20(output_module_4)
        output_module_4 = self.module_21(output_module_4)
        output_module_4 = self.module_22(input=output_module_4, start_dim=1, end_dim=-1)
        output_module_4 = self.module_23(output_module_4)
        output_module_4 = self.module_24(output_module_4)
        output_module_4 = self.module_25(output_module_4)
        return output_module_4
