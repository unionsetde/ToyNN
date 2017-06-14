import os.path
import time
import re
# Core utilities
import SimpleCV
import random
import pickle
import numpy
import layer

def change_image_format(input_image):
    r = numpy.array([input_image[:,:,0]])/255.
    g = numpy.array([input_image[:,:,1]])/255.
    b = numpy.array([input_image[:,:,2]])/255.
    output_image = numpy.append(r,g,axis=0)
    output_image = numpy.append(output_image,b,axis=0)
    return output_image

class Adam_optimizer:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.lr = learning_rate
        self.b1 = beta_1
        self.b2 = beta_2
        self.ep = epsilon

    def save_optimizer(self, file_path):
        pickle.dump(self, open(file_path, 'w'))

    def load_optimizer(self, file_path):
        print "Loading Adam_optimizer..."
        old_one = pickle.load(open(file_path, 'r'))
        self.m_w = old_one.m_w
        self.v_w = old_one.v_w
        self.m_b = old_one.m_b
        self.v_b = old_one.v_b
        self.lr = old_one.lr
        self.b1 = old_one.b1
        self.b2 = old_one.b2
        self.ep = old_one.ep
        print "Loading completed..."

    def calculate_delta(self, gradient_weights, gradient_bias):
        self.m_w = self.b1*self.m_w+(1.0-self.b1)*gradient_weights
        self.v_w = self.b2*self.v_w+(1.0-self.b2)*numpy.square(gradient_weights)
        m_non_biased = self.m_w/(1.0-self.b1)
        v_non_biased = self.v_w/(1.0-self.b2)
        delta_weights = -(self.lr*m_non_biased)/(numpy.sqrt(v_non_biased)+self.ep)
        self.m_b = self.b1*self.m_b+(1.0-self.b1)*gradient_bias
        self.v_b = self.b2*self.v_b+(1.0-self.b2)*numpy.square(gradient_bias)
        m_non_biased = self.m_b/(1.0-self.b1)
        v_non_biased = self.v_b/(1.0-self.b2)
        delta_bias = -(self.lr*m_non_biased)/(numpy.sqrt(v_non_biased)+self.ep)
        return (delta_weights, delta_bias)

    def reset(self):
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0

sample_dir_path = "/cifar/"
#train_list_path = sample_dir_path+"train.list"
labels_path = sample_dir_path+"labels.txt"

network_dir_path = "/network/"
progress_path = network_dir_path+"progress_of_training"
train_list_path = network_dir_path+"train.list"
#train_list_path = network_dir_path+"valid.list"
epoch_path = network_dir_path+"current_epoch"

train_error_path = network_dir_path+"training_error"

layer_1_save_path = network_dir_path+"layer_1"
layer_2_save_path = network_dir_path+"layer_2"
layer_3_save_path = network_dir_path+"layer_3"

optimizer_dir_path = network_dir_path+"optimizer/"
optimizer_1_save_path = optimizer_dir_path+"layer_1_optimizer"
optimizer_2_save_path = optimizer_dir_path+"layer_2_optimizer"
optimizer_3_save_path = optimizer_dir_path+"layer_3_optimizer"

exit_flag = False
is_training = False
epoch_cnt = 1
last_sample_cnt = 0
start_time = time.time()
empty = numpy.array([])

train_list = open(train_list_path).readlines()
labels_list = open(labels_path).readlines()

if os.path.isfile(epoch_path):
    epoch_cnt = int(open(epoch_path).readline())
else:
    temp_file = open(epoch_path, 'w')
    temp_file.write(str(epoch_cnt))
    temp_file.close

if os.path.isfile(progress_path):
    last_sample_cnt = int(open(progress_path).readline())
    is_training = True

numpy_settings = numpy.seterr(all='raise')
###################################################################################################
print "layer 1 initialization..."
input_depth = 3
num_of_filter = 8
field_size = 4
stride = field_size
zero_pad = 0
layer_1 = layer.convolutional_layer(input_depth,num_of_filter,field_size,stride,zero_pad,empty,empty)
optimizer_1 = Adam_optimizer()
if (is_training):
    if (os.path.exists(layer_1_save_path)):
        print "Loading from ",layer_1_save_path
        layer_1.load_layer(layer_1_save_path)
    else:
        print "Unable to open layer 1 from",layer_1_save_path
    if (os.path.exists(optimizer_1_save_path)):
        print "Loading from ",optimizer_1_save_path
        optimizer_1.load_optimizer(optimizer_1_save_path)
    else:
        print "Unable to open optimizer 1 from",optimizer_1_save_path
print "################################################################################"
###################################################################################################
print "layer 2 initialization..."
input_depth = num_of_filter
num_of_filter = 16
field_size = 4
stride = field_size
zero_pad = 0
layer_2 = layer.convolutional_layer(input_depth,num_of_filter,field_size,stride,zero_pad,empty,empty)
optimizer_2 = Adam_optimizer()
if (is_training):
    if (os.path.exists(layer_2_save_path)):
        print "Loading from ",layer_2_save_path
        layer_2.load_layer(layer_2_save_path)
    else:
        print "Unable to open layer 2 from",layer_2_save_path
    if (os.path.exists(optimizer_2_save_path)):
        print "Loading from ",optimizer_2_save_path
        optimizer_2.load_optimizer(optimizer_2_save_path)
    else:
        print "Unable to open optimizer 2 from",optimizer_2_save_path
print "################################################################################"
###################################################################################################
print "layer 3 initialization..."
input_depth = num_of_filter
num_of_filter = 10
field_size = 2
stride = field_size
zero_pad = 0
layer_3 = layer.convolutional_layer(input_depth,num_of_filter,field_size,stride,zero_pad,empty,empty)
optimizer_3 = Adam_optimizer()
if (is_training):
    if (os.path.exists(layer_3_save_path)):
        print "Loading from ",layer_3_save_path
        layer_3.load_layer(layer_3_save_path)
    else:
        print "Unable to open layer 3 from",layer_3_save_path
    if (os.path.exists(optimizer_3_save_path)):
        print "Loading from ",optimizer_3_save_path
        optimizer_3.load_optimizer(optimizer_3_save_path)
    else:
        print "Unable to open optimizer 3 from",optimizer_3_save_path
print "################################################################################"
###################################################################################################
layer_4 = layer.softmax_layer()
###################################################################################################
# Trial
logo = SimpleCV.Image(train_list[0][:-1])
input_image = logo.getNumpy()
input_image = change_image_format(input_image)
numpy.seterr(all='warn')
print "validating layer_1..."
layer_1.check_input_validity(input_image)
output_1 = layer_1.propagate_forward(input_image)
print "layer_1 shape:",layer_1.weights.shape
print "output_1 size:",output_1.shape
print "validating layer_2..."
layer_2.check_input_validity(output_1)
output_2 = layer_2.propagate_forward(output_1)
print "layer_2 shape:",layer_2.weights.shape
print "output_2 size:",output_2.shape
print "validating layer_3..."
layer_3.check_input_validity(output_2)
output_3 = layer_3.propagate_forward(output_2)
print "layer_3 shape:",layer_3.weights.shape
print "output_3 size:",output_3.shape
output_4 = layer_4.propagate_forward(output_3)
###################################################################################################
#batch_size = len(train_list)/100
batch_size = 500
#regularization_magnitude = 1e-5
#dropout = 0.5 # probability of keeping a unit active. higher = less dropout
#decay = 1.0
total_error = 0
min_average_error = 1.0
while (True):
    Gw3 = numpy.zeros(layer_3.weights.shape)
    Gb3 = numpy.zeros(layer_3.bias.shape)
    Gw2 = numpy.zeros(layer_2.weights.shape)
    Gb2 = numpy.zeros(layer_2.bias.shape)
    Gw1 = numpy.zeros(layer_1.weights.shape)
    Gb1 = numpy.zeros(layer_1.bias.shape)
    #if (epoch_cnt > 1):
        #regularization_magnitude *= numpy.power(decay, epoch_cnt)
        #dropout = (dropout/(1.0-decay))*(1.0-numpy.power(decay, epoch_cnt+1))
    print
    print "Current epoch:",epoch_cnt
    #print "Regularization magnitude:",regularization_magnitude
    #print "Dropout:",dropout
    print "Start training from sample:",last_sample_cnt
    for sample_cnt in range(last_sample_cnt, len(train_list)):
        train_sample_label = re.split('[,_,.]', train_list[sample_cnt])[1]
        cost = numpy.zeros(len(labels_list))
        for cnt in range(0, len(labels_list)):
            if (labels_list[cnt][:-1] == train_sample_label):
                cost[cnt] = 1
                break
            elif (cnt == len(labels_list)-1):
                print "Unknown image:",train_list[sample_cnt][:-1]
                exit_flag = True
        if (exit_flag):
            continue
        logo = SimpleCV.Image(train_list[sample_cnt][:-1])
        input_image = logo.getNumpy()
        input_image = change_image_format(input_image)
        output_1 = layer_1.propagate_forward(input_image)
        #Md1 = (numpy.random.rand(*output_1.shape)<dropout)/dropout
        #output_1 *= Md1
        output_2 = layer_2.propagate_forward(output_1)
        #Md2 = (numpy.random.rand(*output_2.shape)<dropout)/dropout
        #output_2 *= Md2
        output_3 = layer_3.propagate_forward(output_2)
        output_4 = layer_4.propagate_forward(output_3)
        error_0 = output_4-cost
        error_1 = layer_4.propagate_backward(error_0)
        (error_2, gw3, gb3) = layer_3.propagate_backward(error_1)
        #error_2 *= Md2
        (error_3, gw2, gb2) = layer_2.propagate_backward(error_2)
        #error_3 *= Md1
        (error_4, gw1, gb1) = layer_1.propagate_backward(error_3)
        Gw3 += gw3
        Gb3 += gb3
        Gw2 += gw2
        Gb2 += gb2
        Gw1 += gw1
        Gb1 += gb1
        total_error += numpy.sum(error_0*error_0)
        #if ((sample_cnt+1)%(batch_size*5) == 0):
            #time_passed = time.time()-start_time
            #hours = int(time_passed)/3600
            #minutes = int(time_passed-3600*hours)/60
            #print
            #print "Computation duration:",hours,"hrs,",minutes,"min,",time_passed%60,"sec"
            #print "Counter:",sample_cnt
            #print "Label:",train_sample_label
            #print "Error sum:",0.5*numpy.sum(numpy.square(error_0))
            #print "Error:"
            #print error_0
        if ((sample_cnt+1)%batch_size == 0):
            # regularize weights
            #layer_3.regularize_weights(0, regularization_magnitude)
            #layer_2.regularize_weights(0, regularization_magnitude)
            #layer_1.regularize_weights(0, regularization_magnitude)
            # update delta weights
            (V_Gw3, V_Gb3) = optimizer_3.calculate_delta(Gw3, Gb3)
            (V_Gw2, V_Gb2) = optimizer_2.calculate_delta(Gw2, Gb2)
            (V_Gw1, V_Gb1) = optimizer_1.calculate_delta(Gw1, Gb1)
            # update the weights
            #V_Gw2[:,:-1,:,:] = 0
            #V_Gw1[:-1,:,:,:] = 0
            #V_Gb1[:-1] = 0
            layer_3.update_weights(V_Gw3, V_Gb3)
            layer_2.update_weights(V_Gw2, V_Gb2)
            layer_1.update_weights(V_Gw1, V_Gb1)
            Gw3 = numpy.zeros(layer_3.weights.shape)
            Gb3 = numpy.zeros(layer_3.bias.shape)
            Gw2 = numpy.zeros(layer_2.weights.shape)
            Gb2 = numpy.zeros(layer_2.bias.shape)
            Gw1 = numpy.zeros(layer_1.weights.shape)
            Gb1 = numpy.zeros(layer_1.bias.shape)
            layer_1.save_layer(layer_1_save_path)
            layer_2.save_layer(layer_2_save_path)
            layer_3.save_layer(layer_3_save_path)
            optimizer_1.save_optimizer(optimizer_1_save_path)
            optimizer_2.save_optimizer(optimizer_2_save_path)
            optimizer_3.save_optimizer(optimizer_3_save_path)
            temp_file = open(progress_path, 'w')
            temp_file.write(str(sample_cnt+1))
            temp_file.close
            #print
            #print "##############################"
            #print "Saving completed..."
            #print "##############################"
        if ((sample_cnt+1)%(batch_size*5) == 0):
            time_passed = time.time()-start_time
            hours = int(time_passed)/3600
            minutes = int(time_passed-3600*hours)/60
            print
            print "Computation duration:",hours,"hrs,",minutes,"min,",time_passed%60,"sec"
            print "Epoch",epoch_cnt,"progress:",100.0*(sample_cnt+1.0)/len(train_list),"%"
            print "layer_1 velocity:",numpy.linalg.norm(V_Gw1.ravel())
            print "layer_2 velocity:",numpy.linalg.norm(V_Gw2.ravel())
            print "layer_3 velocity:",numpy.linalg.norm(V_Gw3.ravel())
    # finish current epoch
    #optimizer_1.reset()
    #optimizer_2.reset()
    #optimizer_3.reset()
    optimizer_1.save_optimizer(optimizer_1_save_path)
    optimizer_2.save_optimizer(optimizer_2_save_path)
    optimizer_3.save_optimizer(optimizer_3_save_path)
    layer_1.save_layer(network_dir_path+"epoch_"+str(epoch_cnt)+"_layer_1_completed")
    layer_2.save_layer(network_dir_path+"epoch_"+str(epoch_cnt)+"_layer_2_completed")
    layer_3.save_layer(network_dir_path+"epoch_"+str(epoch_cnt)+"_layer_3_completed")
    if (total_error/len(train_list) <  min_average_error):
        layer_1.save_layer(network_dir_path+"min_layer_1_completed")
        layer_2.save_layer(network_dir_path+"min_layer_2_completed")
        layer_3.save_layer(network_dir_path+"min_layer_3_completed")
        min_average_error = total_error/len(train_list)
    # record training error
    temp_file = open(train_error_path, 'a+')
    temp_file.write("%s\n" % str(total_error/len(train_list)))
    temp_file.close()
    last_sample_cnt = 0
    temp_file = open(progress_path, 'w')
    temp_file.write(str(last_sample_cnt))
    temp_file.close
    print
    print "Average training error:",total_error/len(train_list)
    print "##############################"
    print "Epoch",epoch_cnt,"completed..."
    print "##############################"
    # shuffle the train dataset
    random.shuffle(train_list)
    temp_file = open(train_list_path, 'w')
    for item in train_list:
        temp_file.writelines("%s" % item)
    temp_file.close()
    epoch_cnt += 1
    temp_file = open(epoch_path, 'w')
    temp_file.write(str(epoch_cnt))
    temp_file.close
    total_error = 0
###################################################################################################
