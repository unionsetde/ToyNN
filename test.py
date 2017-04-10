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

layer_1_save_path = network_dir_path+"layer_1"
layer_2_save_path = network_dir_path+"layer_2"
layer_3_save_path = network_dir_path+"layer_3"
layer_4_save_path = network_dir_path+"layer_4"

exit_flag = False
start_time = time.time()
empty = numpy.array([])

test_list = open(test_list_path).readlines()
labels_list = open(labels_path).readlines()
###################################################################################################
print "layer 1 initialization..."
input_depth = 3
num_of_filter = 64
field_size = 3
stride = 1
zero_pad = 1
layer_1 = layer.convolutional_layer(input_depth,num_of_filter,field_size,stride,zero_pad,empty,empty)
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
logo = SimpleCV.Image(test_list[0][:-1])
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
labels_stat = numpy.zeros(len(labels_list))
total_error = 0
total_match = 0
accuracy = 0
start = 0
end = 500
while (True):
    print
    print "Start testing from sample:",start
    for sample_cnt in range(start, end):
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
        error_0 = output_4 - cost

        error_pos = numpy.argmax(output_4)
        transformed_error_0 = numpy.zeros(output_4.shape)
        transformed_error_0[error_pos] = 1
        if (numpy.sum(transformed_error_0*cost) == 1):
            total_match += 1.
            labels_stat += cost
        total_error += numpy.sum(error_0*error_0)
        if ((sample_cnt+1)%5 == 0):
            time_passed = time.time()-start_time
            hours = int(time_passed)/3600
            minutes = int(time_passed-3600*hours)/60
            print
            print "Computation duration:",hours,"hrs,",minutes,"min,",time_passed%60,"sec"
            print "Sample counter:",sample_cnt
            print "Label:",test_sample_label
            print "Current accuracy:",total_match/(sample_cnt-start+1)
            print "Current total error:",0.5*total_error/(sample_cnt-start+1)
            print "Error:"
            print error_0
    break
print
print "********************************************************************************"
time_passed = time.time()-start_time
hours = int(time_passed)/3600
minutes = int(time_passed-3600*hours)/60
print "Computation duration:",hours,"hrs,",minutes,"min,",time_passed%60,"sec"
print "Total sample number:",end-start
print "Total accuracy:",total_match/(end-start)
print "Total error:",total_error
print "Label statistics:",labels_stat
print "Label list:",labels_list[:][:-1]
