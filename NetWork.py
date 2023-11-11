import torch
from torch.functional import Tensor
import torch.nn as nn
from torchsummary import summary


""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        # define conv1
        self.start_layer = nn.Conv2d(in_channels=3,out_channels=first_num_filters,kernel_size=3, stride=1, padding=1, bias=False)        
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        start_layer = True
        for i in range(3):
            if i == 0:
                filters = self.first_num_filters    
            else:
                filters = self.first_num_filters * 2
            strides = 1 if i == 0 else 2
            if start_layer == True:
                self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters,True))
                start_layer = False
            else:
                self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
            if block_fn is bottleneck_block:
                self.first_num_filters *= 2
            else:
                self.first_num_filters = filters
            filters = filters * 2 if block_fn is bottleneck_block else filters
        self.output_layer = output_layer(filters, self.resnet_version, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.relu = nn.ReLU()
        ### YOUR CODE HERE
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        outputs = self.batch_norm(inputs)
        outputs = self.relu(outputs)
        return outputs
        ### YOUR CODE HERE

class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        self.projection_shortcut = projection_shortcut
        self.conv1 = nn.Conv2d(in_channels=first_num_filters,out_channels=filters,kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn_relu1 = batch_norm_relu_layer(num_features=filters, eps=1e-5, momentum=0.997)
        self.conv2 = nn.Conv2d(in_channels=filters,out_channels=filters,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=filters, eps=1e-5, momentum=0.997)
        if self.projection_shortcut is not None or strides != 1:
            self.projection_shortcut = nn.Sequential(nn.Conv2d(first_num_filters, filters,
                                                               kernel_size=1, stride=strides, bias=False), 
                                                    nn.BatchNorm2d(num_features=filters, eps=1e-5, 
                                                                              momentum=0.997))
        else:
            self.projection_shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        cnv_ot1 = self.conv1(inputs)
        bn_relu1 = self.bn_relu1(cnv_ot1)
        cn2_ot2 =  self.conv2(bn_relu1)
        bn2 = self.bn2(cn2_ot2)
        output = inputs
        #print(bn2)
        """print("---------")
        for m in self.projection_shortcut.children():
            output = m(output)
            print(m,output.shape)"""

        bn2 += self.projection_shortcut(inputs)
        output = self.relu(bn2)
        return output
        ### YOUR CODE HERE

class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters,resnet_first_layer=False,first_layer=False) -> None:
        super(bottleneck_block, self).__init__()
        self.projection_shortcut = projection_shortcut
        self.filters = filters
        self.first_num_filters = first_num_filters
        self.upsample_input = resnet_first_layer
        self.first_layer = first_layer
        if self.projection_shortcut is not None or strides != 1:
            if self.first_layer == True:
                self.projection_shortcut = nn.Sequential(nn.Conv2d(filters//2, filters,
                                                               kernel_size=1, stride=strides, bias=False))
            else:
                self.projection_shortcut = nn.Sequential(nn.Conv2d(first_num_filters, filters,
                                                               kernel_size=1, stride=strides, bias=False))
        else:
            self.projection_shortcut = nn.Sequential()
        
        """self.upsample_input = nn.Sequential(nn.Conv2d(first_num_filters, 64,
                                                               kernel_size=1, stride=1, bias=False), 
                                                    nn.BatchNorm2d(num_features=filters, eps=1e-5, 
                                                                              momentum=0.997))"""
        self.bn_relu1_stack1_input1 = batch_norm_relu_layer(num_features=first_num_filters, eps=1e-5, momentum=0.997)
        self.bn_relu1_inputs1 = batch_norm_relu_layer(num_features=filters//2, eps=1e-5, momentum=0.997)
        self.bn_relu1 = batch_norm_relu_layer(num_features=filters, eps=1e-5, momentum=0.997)

        self.conv1_stack1_input1 = nn.Conv2d(in_channels=first_num_filters,out_channels=first_num_filters,kernel_size=1, stride=strides, padding=0, bias=False)
        self.conv1_input1 = nn.Conv2d(in_channels=filters//2,out_channels=first_num_filters,kernel_size=1, stride=strides, padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_channels=filters,out_channels=first_num_filters,kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_relu2 = batch_norm_relu_layer(num_features=first_num_filters, eps=1e-5, momentum=0.997)
        self.conv2 = nn.Conv2d(in_channels=first_num_filters,out_channels=first_num_filters,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_relu3 = batch_norm_relu_layer(num_features=first_num_filters, eps=1e-5, momentum=0.997)
        self.conv3 = nn.Conv2d(in_channels=first_num_filters,out_channels=filters,kernel_size=1, stride=1, padding=0, bias=False)




        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to 
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        
        ### YOUR CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
        """print("input shape: ",inputs.shape)
        print("********",self.upsample_input)
        print("-------------",self.first_num_filters,self.filters)"""
        if (self.upsample_input==True):
            bn_relu_ot1 = self.bn_relu1_stack1_input1(inputs)
        elif (self.first_layer == True):
            bn_relu_ot1 = self.bn_relu1_inputs1(inputs)
        else:
            bn_relu_ot1 = self.bn_relu1(inputs)
        #print("__bnrelu1__",bn_relu_ot1.size())
        if (self.upsample_input==True):
            cnv_ot1 = self.conv1_stack1_input1(bn_relu_ot1) 
        elif self.first_layer == True:
            cnv_ot1 = self.conv1_input1(bn_relu_ot1)
        else:
            cnv_ot1 = self.conv1(bn_relu_ot1)
        #print("__cvn_ot1__",cnv_ot1.size())
        bn_relu_ot2 = self.bn_relu2(cnv_ot1)
        #print("__bnrelu2__",bn_relu_ot2.size())
        cnv_ot2 = self.conv2(bn_relu_ot2)
        #print("__cvn_ot2__",cnv_ot2.size())
        bn_relu_ot3 = self.bn_relu3(cnv_ot2)
        #print("__bnrelu2__",bn_relu_ot3.size())
        cnv_ot3 = self.conv3(bn_relu_ot3)
        #print("conv333",cnv_ot3.size())
        #print("__cvn_ot2__",cnv_ot2.size())
        #print("Projectionssss",self.projection_shortcut(inputs).size())
        if (self.upsample_input == True):
            cnv_ot3 += self.projection_shortcut(inputs)
        else:
            cnv_ot3 += self.projection_shortcut(bn_relu_ot1)
        #print("enddddddd")
        return cnv_ot3

        ### YOUR CODE HERE

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters,start_layer=False) -> None:
        super(stack_layer, self).__init__()
        filters_out = first_num_filters * 4 if block_fn is bottleneck_block else filters
        ### END CODE HERE
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        self.stack = nn.ModuleList()
        self.start_layer = start_layer
        self.start_layer_stack = True
        for i in range(resnet_size):
            if block_fn is bottleneck_block and self.start_layer==True:
                #print("Stack1_Layer1")
                self.stack.append(bottleneck_block(filters_out,nn.Sequential(),strides,first_num_filters,self.start_layer))
                self.start_layer = False
                self.start_layer_stack = False
            elif block_fn is bottleneck_block and self.start_layer_stack == True:
                #print("Layer1")
                self.stack.append(bottleneck_block(filters_out,nn.Sequential(),strides,first_num_filters,self.start_layer,self.start_layer_stack))
                self.start_layer_stack = False
            elif (strides != 1) and (i==0):
                self.stack.append(block_fn(filters_out,nn.Sequential(),strides,first_num_filters))
            else:
                self.stack.append(block_fn(filters_out,None,1,first_num_filters))
            #self.stack.append(block_fn(filters_out,None,strides,first_num_filters))
            if (block_fn is standard_block):
                first_num_filters = filters_out
            #print("block number",i)

        """if (block_fn is bottleneck_block):
            first_num_filters *= 2
            filters *= 2"""
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        for block in self.stack:
            inputs = block(inputs)
        return inputs
        ### END CODE HERE

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
        
        ### END CODE HERE
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(filters, num_classes)
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        outputs = self.avg_pool(inputs)
        #print("output of average pooling",outputs.size())
        #outputs = torch.flatten(outputs,1)
        outputs = outputs.view(outputs.size(0),-1)
        #print("flatten output:",outputs.size())
        outputs = self.fc(outputs)
        #print(outputs)
        #print("final outputs",outputs.size())
        return outputs
        ### END CODE HERE