import os.path
import time
import re
# Core utilities
import SimpleCV
import random
import numpy
import layer

def change_image_format(input_image):
    r = numpy.array([input_image[:,:,0]])/255.
    g = numpy.array([input_image[:,:,1]])/255.
    b = numpy.array([input_image[:,:,2]])/255.
    output_image = numpy.append(r,g,axis=0)
    output_image = numpy.append(output_image,b,axis=0)
    return output_image

sample_dir_path = "/cifar/"
#train_list_path = sample_dir_path+"train.list"
labels_path = sample_dir_path+"labels.txt"

network_dir_path = "/network/"
train_list_path = network_dir_path+"train.list"
progress_path = network_dir_path+"progess_of_training"
epoch_path = network_dir_path+"current_epoch"

layer_1_save_path = network_dir_path+"layer_1"
layer_2_save_path = network_dir_path+"layer_2"
layer_3_save_path = network_dir_path+"layer_3"
layer_4_save_path = network_dir_path+"layer_4"

exit_flag = False
is_training = False
epoch_cnt = 0
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
###################################################################################################
print "layer 1 initialization..."
input_depth = 3
num_of_filter = 64
field_size = 3
stride = 1
zero_pad = 1
layer_1 = layer.convolutional_layer(input_depth,num_of_filter,field_size,stride,zero_pad,empty,empty)
if (is_training):
    if (os.path.exists(layer_1_save_path)):
        print "Loading from ",layer_1_save_path
        layer_1.load_layer(layer_1_save_path)
    else:
        print "Unable to open layer 1 from",layer_1_save_path
print "################################################################################"
###################################################################################################
input_depth = num_of_filter
field_size = 2
stride = 2
layer_2 = layer.max_pooling_layer(input_depth, field_size, stride)
###################################################################################################
print "layer 3 initialization..."
num_of_input = 16384
num_of_output = 10
layer_3 = layer.fully_connected_layer(num_of_input,num_of_output,empty,empty)
if (is_training):
    if (os.path.exists(layer_3_save_path)):
        print "Loading from ",layer_3_save_path
        layer_3.load_layer(layer_3_save_path)
    else:
        print "Unable to open layer 3 from",layer_3_save_path
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
print "validating layer_2..."
layer_2.check_input_validity(output_1)
output_2 = layer_2.propagate_forward(output_1)
print "validating layer_3..."
layer_3.check_input_validity(output_2)
output_3 = layer_3.propagate_forward(output_2)
print "validating layer_4..."
output_4 = layer_4.propagate_forward(output_3)
###################################################################################################
batch_size = 500
friction_coefficient = 0.5
regularization_magnitude = 0.01
relative_update_magnitude = 0.001
while (True):
    Gw3 = numpy.zeros(layer_3.weights.shape)
    Gb3 = numpy.zeros(layer_3.bias.shape)
    Gw1 = numpy.zeros(layer_1.weights.shape)
    Gb1 = numpy.zeros(layer_1.bias.shape)
    V_Gw3 = numpy.zeros(layer_3.weights.shape)
    V_Gb3 = numpy.zeros(layer_3.bias.shape)
    V_Gw1 = numpy.zeros(layer_1.weights.shape)
    V_Gb1 = numpy.zeros(layer_1.bias.shape)
    if (epoch_cnt > 0):
        friction_coefficient = friction_coefficient*(1.0-(numpy.power(0.5, epoch_cnt+1)))/0.5
        regularization_magnitude *= numpy.power(0.5, epoch_cnt)
        relative_update_magnitude *= numpy.power(0.5, epoch_cnt)
    print
    print "Current epoch:",epoch_cnt
    print "Learning momentum:",friction_coefficient
    print "Learning magnitude:",relative_update_magnitude
    print "Regularization magnitude:",regularization_magnitude
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
        output_2 = layer_2.propagate_forward(output_1)
        output_3 = layer_3.propagate_forward(output_2)
        output_4 = layer_4.propagate_forward(output_3)
        error_0 = output_4 - cost
        error_1 = layer_4.propagate_backward(error_0)
        (error_2, gw3, gb3) = layer_3.propagate_backward(error_1, output_2, True)
        error_3 = layer_2.propagate_backward(error_2)
        (error_4, gw1, gb1) = layer_1.propagate_backward(error_3, True)
        Gw3 += gw3
        Gb3 += gb3
        Gw1 += gw1
        Gb1 += gb1

        if (sample_cnt%5 == 0):
            time_passed = time.time()-start_time
            hours = int(time_passed)/3600
            minutes = int(time_passed-3600*hours)/60
            print
            print "Computation duration:",hours,"hrs,",minutes,"min,",time_passed%60,"sec"
            print "Counter:",sample_cnt
            print "Label:",train_sample_label
            print "Error sum:",0.5*numpy.sum(numpy.square(error_0))
            print "Error:"
            print error_0

        if ((sample_cnt+1)%batch_size == 0):
            # regularize weights
            layer_3.regularize_weights(regularization_magnitude)
            layer_1.regularize_weights(regularization_magnitude)
            # evaluate learning rate
            w3_scale = numpy.linalg.norm(layer_3.weights.ravel())
            w1_scale = numpy.linalg.norm(layer_1.weights.ravel())
            Gw3_scale = numpy.linalg.norm(Gw3.ravel())
            Gw1_scale = numpy.linalg.norm(Gw1.ravel())
            learning_rate_3 = (relative_update_magnitude*w3_scale)/Gw3_scale
            learning_rate_1 = (relative_update_magnitude*w1_scale)/Gw1_scale
            # update training velocity
            V_Gw3 = friction_coefficient*V_Gw3 - learning_rate_3*Gw3
            V_Gb3 = friction_coefficient*V_Gb3 - learning_rate_3*Gb3
            V_Gw1 = friction_coefficient*V_Gw1 - learning_rate_1*Gw1
            V_Gb1 = friction_coefficient*V_Gb1 - learning_rate_1*Gb1
            # update the weights
            layer_3.update_weights(V_Gw3, V_Gb3, -1.0)
            layer_1.update_weights(V_Gw1, V_Gb1, -1.0)
            Gw3 = numpy.zeros(layer_3.weights.shape)
            Gb3 = numpy.zeros(layer_3.bias.shape)
            Gw1 = numpy.zeros(layer_1.weights.shape)
            Gb1 = numpy.zeros(layer_1.bias.shape)
            layer_1.save_layer(layer_1_save_path)
            layer_3.save_layer(layer_3_save_path)
            temp_file = open(progress_path, 'w')
            temp_file.write(str(sample_cnt+1))
            temp_file.close
            print
            print "layer_1 velocity:",numpy.linalg.norm(V_Gw1.ravel())
            print "layer_3 velocity:",numpy.linalg.norm(V_Gw3.ravel())
            print "layer_1 learning rate:",learning_rate_1
            print "layer_3 learning rate:",learning_rate_3
            print "##############################"
            print "Saving completed..."
            print "##############################"

        if ((sample_cnt+1)%5000 == 0):
            checkpoint = (sample_cnt+1)/5000
            layer_1.save_layer(layer_1_save_path+str(sample_cnt+1))
            layer_3.save_layer(layer_3_save_path+str(sample_cnt+1))
            layer_1.save_layer(network_dir_path+"epoch_"+str(epoch_cnt)+"layer_1_"+str(checkpoint))
            layer_3.save_layer(network_dir_path+"epoch_"+str(epoch_cnt)+"layer_3_"+str(checkpoint))
            print
            print "##############################"
            print "Checkpoint saving completed..."
            print "##############################"
        
    # finish current epoch
    layer_1.save_layer(layer_1_save_path)
    layer_3.save_layer(layer_3_save_path)
    layer_1.save_layer(network_dir_path+"epoch_"+str(epoch_cnt)+"layer_1")
    layer_3.save_layer(network_dir_path+"epoch_"+str(epoch_cnt)+"layer_3")
    # shuffle the train dataset
    random.shuffle(train_list)
    temp_file = open(train_list_path, 'w')
    for item in train_list:
        temp_file.writelines("%s" % item)
    temp_file.close
    last_sample_cnt = 0
    temp_file = open(progress_path, 'w')
    temp_file.write(str(last_sample_cnt))
    temp_file.close
    epoch_cnt += 1
    temp_file = open(epoch_path, 'w')
    temp_file.write(str(epoch_cnt))
    temp_file.close
    print
    print "##############################"
    print "Saving completed..."
    print "##############################"
###################################################################################################
