import sys
import numpy
from PIL import Image
from matplotlib import pylab as plt
from Convolutional_Neural_Network import Convolutional_Neural_Network

def get_data(class_name, image_number):
    
    name = class_name + str(image_number) + ".jpg"
    image = numpy.array(Image.open(name))
    return image

def main():
    
    image_size = 
    channel = 
    number_of_kernels = 
    kernel_sizes = 
    pool_sizes = 
    Hidden_Layer_sizes = 
    number_of_class = 
    learning_rate = 
    drop_out_possibility = 
    epochs =
    
    number_of_convpool_layers = len(number_of_kernels)
    number_of_hidden_layers = len(Hidden_Layer_sizes)

    train_number_each_class = 
    test_number_each_class = 
    
    train_number = train_number_each_class * number_of_class
    test_number = test_number_each_class * number_of_class
    
    minibatch_size = 
    minibatch_number = train_number / minibatch_size

    train_input = numpy.zeros((minibatch_number, minibatch_size, channel, image_size[0], image_size[1]))
    train_label = numpy.zeros((minibatch_number, minibatch_size, number_of_class))
    test_input = numpy.zeros((test_number, channel, image_size[0], image_size[1]))
    test_label = numpy.zeros((test_number, number_of_class))

    #generate data

    confusion_matrix = numpy.zeros((number_of_class, number_of_class))
    accuracy = numpy.zeros(epochs + 1)
    
    print
    print "I'm building the model."
    classfier = Convolutional_Neural_Network(image_size, channel, number_of_kernels, kernel_sizes, pool_sizes, Hidden_Layer_sizes, number_of_class, number_of_convpool_layers, number_of_hidden_layers, drop_out_possibility, minibatch_size)

    print
    print "I'm training the model"
    
    for i in xrange(epochs):
        for j in xrange(minibatch_number):
            
            if ((i + 1) % 1 == 0):
                print
                print "Test"
                print "Iter = %d/%d" %(i + 1, epochs)
                predicted_label = classfier.predict(test_input)
                
                predicted_index = -1
                actual_index = -1

                for k in xrange(test_number):
                    for l in xrange(number_of_class):
                        if predicted_label[k][l] == 1:
                            predicted_index = l
                        if test_label[k][l] == 1:
                            actual_index = l
            
                    confusion_matrix[actual_index][predicted_index] += 1
                    predicted_index = -1
                    actual_index = -1

                for k in xrange(number_of_class):
                    for l in xrange(number_of_class):
                        if(k == l):
                            accuracy[i] += confusion_matrix[k][l]
                
                accuracy[i] /= test_number
                print
                print "accuracy is %f." %(accuracy[i])
                
                for k in xrange(number_of_class):
                    for l in xrange(number_of_class):
                        confusion_matrix[k][l] = 0
            
            print 
            classfier.train(train_input[j], train_label[j], learning_rate)
    
    print
    print "I'm evaluateing the model."
    predicted_label = classfier.predict(test_input)

    predicted_index = -1
    actual_index = -1

    for i in xrange(test_number):
        for j in xrange(number_of_class):
            if predicted_label[i][j] == 1:
                predicted_index = j
            if test_label[i][j] == 1:
                actual_index = j

        confusion_matrix[actual_index][predicted_index] += 1
        predicted_index = -1
        actual_index = -1

    for k in xrange(number_of_class):
        for l in xrange(number_of_class):
            if(k == l):
                accuracy[-1] += confusion_matrix[k][l]
                
    accuracy[-1] /= test_number

    print
    print "Accuracy is follow."
    print accuracy
        
if __name__ == "__main__":
    main()