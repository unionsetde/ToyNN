import os.path
import matplotlib.pyplot as plt

network_dir_path = "/network/"
test_error_path = network_dir_path+"testing_error"
train_error_path = network_dir_path+"training_error"

if (os.path.exists(test_error_path)):
    test_error = open(test_error_path).readlines()
    epoch_num = len(test_error)+1

if (os.path.exists(train_error_path)):
    train_error = open(train_error_path).readlines()

tmp = [float(i) for i in test_error]
for i in range(len(tmp)):
    if (tmp[i] == (min(tmp))):
        print "Smallest test error"
        print "epoch:",i+1
        print "error:",tmp[i]

plt.plot([float(i) for i in test_error],label="test_error")
plt.plot([float(i) for i in train_error],label="train_error")
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.ylabel('Average error')
plt.show()
