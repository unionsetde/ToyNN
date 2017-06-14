import os.path
import time
import re
# Core utilities
import SimpleCV
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
test_list_path  = sample_dir_path+"test.list"
labels_path = sample_dir_path+"labels.txt"

network_dir_path = "/network/"
test_error_path = network_dir_path+"testing_error"

exit_flag = False
start_time = time.time()
empty = numpy.array([])

test_list = open(test_list_path).readlines()
labels_list = open(labels_path).readlines()
###################################################################################################
print "layer 1 initialization..."
input_depth = 3
num_of_filter = 8
field_size = 4
stride = field_size
zero_pad = 0
layer_1 = layer.convolutional_layer(input_depth,num_of_filter,field_size,stride,zero_pad,empty,empty)
print "################################################################################"
###################################################################################################
print "layer 2 initialization..."
input_depth = num_of_filter
num_of_filter = 16
field_size = 4
stride = field_size
zero_pad = 0
layer_2 = layer.convolutional_layer(input_depth,num_of_filter,field_size,stride,zero_pad,empty,empty)
print "################################################################################"
###################################################################################################
print "layer 3 initialization..."
input_depth = num_of_filter
num_of_filter = 10
field_size = 2
stride = field_size
zero_pad = 0
layer_3 = layer.convolutional_layer(input_depth,num_of_filter,field_size,stride,zero_pad,empty,empty)
print "################################################################################"
###################################################################################################
layer_4 = layer.softmax_layer()
###################################################################################################
labels_stat = numpy.zeros(len(labels_list))
total_error = 0
total_match = 0
accuracy = 0
if (os.path.exists(test_error_path)):
    test_error = open(test_error_path).readlines()
    epoch_num = len(test_error)+1
else:
    epoch_num = 1

print
print "Waiting epoch",epoch_num,"to complete..."
while (True):
    layer_1_save_path = network_dir_path+"epoch_"+str(epoch_num)+"_layer_1_completed"
    layer_2_save_path = network_dir_path+"epoch_"+str(epoch_num)+"_layer_2_completed"
    layer_3_save_path = network_dir_path+"epoch_"+str(epoch_num)+"_layer_3_completed"
    if (os.path.exists(layer_1_save_path)):
        print "Loading from ",layer_1_save_path
        layer_1.load_layer(layer_1_save_path)
    else:
        time.sleep(10)
        continue
    if (os.path.exists(layer_2_save_path)):
        print "Loading from ",layer_2_save_path
        layer_2.load_layer(layer_2_save_path)
    else:
        time.sleep(10)
        continue
    if (os.path.exists(layer_3_save_path)):
        print "Loading from ",layer_3_save_path
        layer_3.load_layer(layer_3_save_path)
    else:
        time.sleep(10)
        continue
    print "********************************************************************************"
    print "Testing after epoch",epoch_num
    for sample_cnt in range(0, len(test_list)):
        test_sample_label = re.split('[,_,.]', test_list[sample_cnt])[1]
        cost = numpy.zeros(len(labels_list))
        for cnt in range(0, len(labels_list)):
            if (labels_list[cnt][:-1] == test_sample_label):
                cost[cnt] = 1
                break
            elif (cnt == len(labels_list)-1):
                print "Unknown image:",test_list[sample_cnt][:-1]
                exit_flag = True
        if (exit_flag):
            break
        logo = SimpleCV.Image(test_list[sample_cnt][:-1])
        input_image = logo.getNumpy()
        input_image = change_image_format(input_image)
        output_1 = layer_1.propagate_forward(input_image)
        output_2 = layer_2.propagate_forward(output_1)
        output_3 = layer_3.propagate_forward(output_2)
        output_4 = layer_4.propagate_forward(output_3)
        error_0 = output_4-cost
        error_pos = numpy.argmax(output_4)
        transformed_error_0 = numpy.zeros(output_4.shape)
        transformed_error_0[error_pos] = 1
        total_error += numpy.sum(error_0*error_0)
        if (numpy.sum(transformed_error_0*cost) == 1):
            total_match += 1.
            labels_stat += cost
    temp_file = open(test_error_path, 'a+')
    temp_file.write("%s\n" % str(total_error/len(test_list)))
    temp_file.close()
    print "********************************************************************************"
    print "================================================================================"
    time_passed = time.time()-start_time
    hours = int(time_passed)/3600
    minutes = int(time_passed-3600*hours)/60
    print "Computation duration:",hours,"hrs,",minutes,"min,",time_passed%60,"sec"
    print "Total sample number:",len(test_list)
    print "Total error:",total_error
    print "Total accuracy:",total_match/len(test_list)
    print "Label statistics:",labels_stat
    print "Label list:",labels_list[:][:-1]
    print "Average testing error:",total_error/len(test_list)
    print "================================================================================"
    print "********************************************************************************"
    total_match = 0
    total_error = 0
    labels_stat = 0
    epoch_num = epoch_num+1
    print
    print "Waiting epoch",epoch_num,"to complete..."
###################################################################################################
