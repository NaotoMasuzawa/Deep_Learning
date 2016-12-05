import sys
import numpy
from utils import*

class Conv_Pool_Layer(object):
    
    def __init__(self, image_size, channel, number_of_kernels, kernel_size, pool_size, convolved_size, pooled_size, minibatch_size):
         
        self.image_size        = image_size
        self.channel           = channel
        self.number_of_kernels = number_of_kernels
        self.kernel_size       = kernel_size
        self.pool_size         = pool_size
        self.convolved_size    = convolved_size
        self.pooled_size       = pooled_size

        print "image_size"
        print image_size
        print "channel"
        print channel
        print "number_of_kernels"
        print number_of_kernels
        print "kernel_size"
        print kernel_size
        print "pool_size"
        print pool_size
        print "convolved_size"
        print convolved_size
        print "pooled_size"
        print pooled_size
        print

        conv_in = channel * kernel_size[0] * kernel_size[1]
        conv_out = number_of_kernels * kernel_size[0] * kernel_size[1] / (pool_size[0] * pool_size[1])
        uniform_boundary = numpy.sqrt(6.0 / (conv_in + conv_out))
     
        self.W = numpy.array(numpy.random.uniform(
                                low = -uniform_boundary,
                                high = uniform_boundary,
                                size = (number_of_kernels, channel, kernel_size[0], kernel_size[1])
                            )
                        )
 
        self.b = numpy.zeros(number_of_kernels)

        self.dconved_delta = numpy.zeros((minibatch_size, self.channel, self.image_size[0], self.image_size[1]))
 
    def forward(self, input):
        
        z = self.convolve(input)
        return self.max_pooling(z)
 
    def backward(self, prev_layer_delta, learning_rate):
        
        delta = self.dmax_pooling(prev_layer_delta, self.activated_input, learning_rate)
        self.dconvolve(delta, learning_rate)
 
    def convolve(self, input):
        
        minibatch_size = len(input)
        pre_activated_input = numpy.zeros((minibatch_size, self.number_of_kernels, self.convolved_size[0], self.convolved_size[1]))
        activated_input = numpy.zeros((minibatch_size, self.number_of_kernels, self.convolved_size[0], self.convolved_size[1]))
 
        for batch_loop in xrange(minibatch_size):
            for num_ker_loop in xrange(self.number_of_kernels):
                for conved_col_loop in xrange(int(self.convolved_size[0])):
                    for conved_row_loop in xrange(int(self.convolved_size[1])):
                 
                        for channel_loop in xrange(self.channel):
                            for ker_col_loop in xrange(self.kernel_size[0]):
                                for ker_row_loop in xrange(self.kernel_size[1]):
                             
                                    pre_activated_input[batch_loop][num_ker_loop][conved_col_loop][conved_row_loop] += self.W[num_ker_loop][channel_loop][ker_col_loop][ker_row_loop] * input[batch_loop][channel_loop][conved_col_loop + ker_col_loop][conved_row_loop + ker_row_loop]
                         
                        pre_activated_input[batch_loop][num_ker_loop][conved_col_loop][conved_row_loop] = pre_activated_input[batch_loop][num_ker_loop][conved_col_loop][conved_row_loop] + self.b[num_ker_loop]
                        activated_input[batch_loop][num_ker_loop][conved_col_loop][conved_row_loop]     = ReLU(pre_activated_input[batch_loop][num_ker_loop][conved_col_loop][conved_row_loop])
                         
        self.input = input
        self.pre_activated_input = pre_activated_input
        self.activated_input = activated_input
         
        return activated_input
 
    def dconvolve(self, prev_layer_delta, learning_rate):
        
        minibatch_size = len(prev_layer_delta)
        grad_W = numpy.zeros( (self.number_of_kernels, self.channel, self.kernel_size[0], self.kernel_size[1]) )
        grad_b = numpy.zeros( self.number_of_kernels )
 
        for batch_loop in xrange(minibatch_size):
            for num_ker_loop in xrange(self.number_of_kernels):
                for conved_col_loop in xrange(int(self.convolved_size[0])):
                    for conved_row_loop in xrange(int(self.convolved_size[1])):
 
                        d = prev_layer_delta[batch_loop][num_ker_loop][conved_col_loop][conved_row_loop] * dReLU(self.pre_activated_input[batch_loop][num_ker_loop][conved_col_loop][conved_row_loop])
                        grad_b[num_ker_loop] += d
 
                    for channel_loop in xrange(self.channel):
                        for ker_col_loop in xrange(self.kernel_size[0]):
                            for ker_row_loop in xrange(self.kernel_size[1]):
 
                                grad_W[num_ker_loop][channel_loop][ker_col_loop][ker_row_loop] = d * self.input[batch_loop][channel_loop][conved_col_loop + ker_col_loop][conved_row_loop + ker_row_loop]
         
        for num_ker_loop in xrange(self.number_of_kernels):
            self.b[num_ker_loop] -= learning_rate * grad_b[num_ker_loop] / minibatch_size
            for channel_loop in xrange(self.channel):
                for ker_col_loop in xrange(self.kernel_size[0]):
                    for ker_row_loop in xrange(self.kernel_size[1]):
                        self.W[num_ker_loop][channel_loop][ker_col_loop][ker_row_loop] -= learning_rate * grad_W[num_ker_loop][channel_loop][ker_col_loop][ker_row_loop] / minibatch_size
       
 
        for batch_loop in xrange(minibatch_size):
            for channel_loop in xrange(self.channel):
                for image_col_loop in xrange(int(self.image_size[0])):
                    for image_row_loop in xrange(int(self.image_size[1])):
                        for num_ker_loop in xrange(self.number_of_kernels):
                            for ker_col_loop in xrange(self.kernel_size[0]):
                                for ker_row_loop in xrange(self.kernel_size[1]):
 
                                    d =0.0
                                 
                                    if (image_col_loop - (self.kernel_size[0] - 1) - ker_col_loop >= 0) and (image_row_loop - (self.kernel_size[1] - 1) - ker_row_loop >= 0):
                                     
                                        d = prev_layer_delta[batch_loop][num_ker_loop][image_col_loop - (self.kernel_size[0] - 1) - ker_col_loop][image_row_loop - (self.kernel_size[1] - 1) - ker_row_loop] *  dReLU(self.pre_activated_input[batch_loop][num_ker_loop][image_col_loop - (self.kernel_size[0] -1) - ker_col_loop][image_row_loop - (self.kernel_size[1] - 1) - ker_row_loop]) * self.W[num_ker_loop][channel_loop][ker_col_loop][ker_row_loop]
                                     
                                    self.dconved_delta[batch_loop][channel_loop][image_col_loop][image_row_loop] += d
       
    def max_pooling(self, input):
        
        minibatch_size = len(input)
        pooled_input = numpy.zeros( (minibatch_size, self.number_of_kernels, self.pooled_size[0], self.pooled_size[1]) )
         
        for batch_loop in xrange(minibatch_size):
            for num_ker_loop in xrange(self.number_of_kernels):
                for pooled_col_loop in xrange(int(self.pooled_size[0])):
                    for pooled_row_loop in xrange(int(self.pooled_size[1])):
 
                        max = 0.0
 
                        for pool_col_loop in xrange(self.pool_size[0]):
                            for pool_row_loop in xrange(self.pool_size[1]): 
 
                                if (pool_col_loop == 0) and (pool_row_loop == 0):
                                    max = input[batch_loop][num_ker_loop][self.pool_size[0] * pooled_col_loop][self.pool_size[1] * pooled_row_loop]
                                    next
                         
                                if (max < input[batch_loop][num_ker_loop][self.pool_size[0] * pooled_col_loop + pool_col_loop][self.pool_size[1] * pooled_row_loop + pool_row_loop]):
                                    max = input[batch_loop][num_ker_loop][self.pool_size[0] * pooled_col_loop + pool_col_loop][self.pool_size[1] * pooled_row_loop + pool_row_loop]
                    
                        pooled_input[batch_loop][num_ker_loop][pooled_col_loop][pooled_row_loop] = max
 
        self.pooled_input = pooled_input
        return pooled_input
 
 
    def dmax_pooling(self, prev_layer_delta, layer_input, learning_rate):
    
        minibatch_size = len(prev_layer_delta)
   
        delta = numpy.zeros((minibatch_size, self.number_of_kernels, self.convolved_size[0], self.convolved_size[1]))
 
        for batch_loop in xrange(minibatch_size):
            for num_ker_loop in xrange(self.number_of_kernels):
                for pooled_col_loop in xrange(int(self.pooled_size[0])):
                    for pooled_row_loop in xrange(int(self.pooled_size[1])):
                        for pool_col_loop in xrange(self.pool_size[0]): 
                            for pool_row_loop in xrange(self.pool_size[1]):
 
                                d = 0.0
 
                                if self.pooled_input[batch_loop][num_ker_loop][pooled_col_loop][pooled_row_loop] == layer_input[batch_loop][num_ker_loop][self.pool_size[0] * pooled_col_loop + pool_col_loop][self.pool_size[1] * pooled_row_loop + pool_row_loop]:
                                     
                                    d = prev_layer_delta[batch_loop][num_ker_loop][pooled_col_loop][pooled_row_loop]
                                 
                                delta[batch_loop][num_ker_loop][self.pool_size[0] * pooled_col_loop + pool_col_loop][self.pool_size[1] * pooled_row_loop + pool_row_loop] = d
             
        return delta