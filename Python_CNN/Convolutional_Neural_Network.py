import sys
import numpy
from utils import*
from Conv_Pool_Layer import Conv_Pool_Layer
from Hidden_Layer import Hidden_Layer
from Logistic_Regression import Logistic_Regression

class Convolutional_Neural_Network():
    
    def __init__(self, image_size, channel, number_of_kernels, kernel_size, pool_sizes, hidden_layer_sizes, number_of_class, number_of_convpool_layers, number_of_hidden_layers, drop_out_possibility, mini_batch_size):

        self.image_size                = image_size
        self.channel                   = channel
        self.number_of_kernels         = number_of_kernels
        self.kernel_sizes              = kernel_size
        self.pool_sizes                = pool_sizes
        self.hidden_layer_sizes        = hidden_layer_sizes
        self.number_of_class           = number_of_class
        self.number_of_convpool_layers = number_of_convpool_layers
        self.number_of_hidden_layers   = number_of_hidden_layers
        self.drop_out_possibility      = drop_out_possibility
        self.mini_batch_size           = mini_batch_size

        print
        print "I construct the conv pool layers."
        self.conv_pool_layers = []
        self.convolved_sizes = numpy.zeros( (self.number_of_convpool_layers, 2) )
        self.pooled_sizes = numpy.zeros( (self.number_of_convpool_layers, 2) )
        
        size = numpy.zeros(2)
        for i in xrange(number_of_convpool_layers):

            if(i == 0):
                size[0] = self.image_size[0]
                size[1] = self.image_size[1]
                each_channel = self.channel
        
            else:
                size[0] = self.pooled_sizes[i - 1][0]
                size[1] = self.pooled_sizes[i - 1][1]
                each_channel = number_of_kernels[i - 1]
            
            self.convolved_sizes[i][0] = size[0] - self.kernel_sizes[i][0] + 1
            self.convolved_sizes[i][1] = size[1] - self.kernel_sizes[i][1] + 1
            self.pooled_sizes[i][0] = self.convolved_sizes[i][0] / pool_sizes[i][0]
            self.pooled_sizes[i][1] = self.convolved_sizes[i][1] / pool_sizes[i][1]

            print "%d Layer" %(i + 1)
            conv_pool_layer = Conv_Pool_Layer(size, each_channel, number_of_kernels[i], self.kernel_sizes[i], pool_sizes[i], self.convolved_sizes[i], self.pooled_sizes[i], self.mini_batch_size)
            self.conv_pool_layers.append(conv_pool_layer)
        
        print "I construct the hidden layers."
        self.hidden_layers = []
        self.flattened_size = number_of_kernels[-1] * self.pooled_sizes[-1][0] * self.pooled_sizes[-1][1]
        for i in xrange(number_of_hidden_layers):
        
            if(i == 0):
                In = self.flattened_size
            else:
                In = hidden_layer_sizes[i - 1]
            
            print "%d Layer" %(i + 1)
            hidden_layer = Hidden_Layer(In, hidden_layer_sizes[i], drop_out_possibility)
            self.hidden_layers.append(hidden_layer)
        
        print "I construct the logistic layer."
        self.logistic_layer = Logistic_Regression(hidden_layer_sizes[-1], number_of_class)
    
    def train(self, x, T, learning_rate):

        print "Conv_layer_loop"
        for conv_layer_loop in xrange(self.number_of_convpool_layers):
            if conv_layer_loop == 0:
                conv_x = x
            
            conv_x = self.conv_pool_layers[conv_layer_loop].forward(conv_x)

            if ( conv_layer_loop == (self.number_of_convpool_layers - 1) ):
                self.conved = conv_x
        
        print "Hidden_layer_loop"
        for hidden_layer_loop in xrange(self.number_of_hidden_layers):
            if(hidden_layer_loop == 0):
                hidden_x = self.flatten(self.conved)
            
            hidden_x = self.hidden_layers[hidden_layer_loop].forward(hidden_x)

            if( hidden_layer_loop == (self.number_of_hidden_layers - 1) ):
                self.hiddened_x = hidden_x
            
        print "Logistic layer and update last W"
        self.logistic_layer.train(self.hiddened_x, T, learning_rate)

        print "Back the hidden layer"
        for hidden_layer_loop in reversed(xrange(0, self.number_of_hidden_layers)):

            if(hidden_layer_loop == self.number_of_hidden_layers - 1):
                print "layer%d" %(hidden_layer_loop + 1)
                self.hidden_layers[hidden_layer_loop].backward(self.logistic_layer, learning_rate)
        
            else:
                print "layer%d" %(hidden_layer_loop + 1)
                self.hidden_layers[hidden_layer_loop].backward(self.hidden_layers[hidden_layer_loop + 1], learning_rate)

            if(hidden_layer_loop == 0):
                self.back_hiddened = self.hidden_layers[0].delta

        print "Back the conv_layer"
        for conv_layer_loop in reversed(xrange(0, self.number_of_convpool_layers)):
            
            if ( conv_layer_loop == (self.number_of_convpool_layers - 1) ):
                unflattened = self.unflatten(self.back_hiddened)
                print "layer%d" %(conv_layer_loop + 1)
                self.conv_pool_layers[conv_layer_loop].backward(unflattened, learning_rate)

            else:
                print "layer%d" %(conv_layer_loop + 1)
                self.conv_pool_layers[conv_layer_loop].backward(self.conv_pool_layers[conv_layer_loop + 1].dconved_delta, learning_rate)
    
    def flatten(self, z):

        mini_batch_size = len(z)
        flatten_x = numpy.zeros((mini_batch_size, self.flattened_size))
        
        for i in xrange(mini_batch_size):
            index = 0
            for j in xrange(self.number_of_kernels[-1]):
                for k in xrange(int(self.pooled_sizes[-1][0])):
                    for l in xrange(int(self.pooled_sizes[-1][1])):
                    
                        flatten_x[i][index] = z[i][j][k][l]
                        index += 1
        
        return flatten_x

    def unflatten(self, z):

        mini_batch_size = len(z)
        delta_flatten = numpy.zeros((mini_batch_size, self.flattened_size))
        for i in xrange(mini_batch_size):
            for j in xrange(int(self.flattened_size)):
                for k in xrange(self.hidden_layer_sizes[0]):
                    delta_flatten[i][j] += self.hidden_layers[0].W[k][j] * z[i][k]

        delta = numpy.zeros((mini_batch_size, self.number_of_kernels[-1], self.pooled_sizes[-1][0], self.pooled_sizes[-1][1]))
        
        for i in xrange(mini_batch_size):
            index = 0
            for j in xrange(self.number_of_kernels[-1]):
                for k in xrange(int(self.pooled_sizes[-1][0])):
                    for l in xrange(int(self.pooled_sizes[-1][1])):
                    
                        delta[i][j][k][l] = delta_flatten[i][index]
                        index += 1
        
        return delta

    def pretest(self):

        for layer_loop in xrange(self.number_of_hidden_layers):

            if(layer_loop == 0):
                pretest_In = self.flattened_size

            else:
                pretest_In = self.hidden_layer_sizes[layer_loop - 1]
        
            if( layer_loop == (self.number_of_hidden_layers - 1) ):
                pretest_Out = self.number_of_class

            else:
                pretest_Out = self.hidden_layer_sizes[layer_loop + 1]
        
            for i in xrange(pretest_Out):
                for j in xrange(pretest_In):
                    self.hidden_layer[layer_loop].W[i][j] *= (1 - self.drop_out_possibility)
            
    def predict(self, Input):

        for conv_layer_loop in xrange(self.number_of_convpool_layers):
            if (conv_layer_loop == 0):
                conv = Input
        
            conv = self.conv_pool_layers[conv_layer_loop].forward(conv)
            
            if ( conv_layer_loop == (self.number_of_convpool_layers - 1) ):
                self.conved = self.flatten(conv)

        for hidden_layer_loop in xrange(self.number_of_hidden_layers):
            if(hidden_layer_loop == 0):
                hidden = self.conved
        
            hidden = self.hidden_layers[hidden_layer_loop].forward(hidden)
            
            if ( hidden_layer_loop == (self.number_of_hidden_layers - 1) ):
                self.hiddened = hidden
    
        return self.logistic_layer.predict(self.hiddened)