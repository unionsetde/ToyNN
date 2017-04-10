import numpy
import pickle

class fully_connected_layer:
    def __init__(self, num_of_input, num_of_output, w, b):
        self.input_size = num_of_input
        self.output_size = num_of_output
        self.weights = numpy.random.randn(self.output_size, self.input_size)
        self.bias = numpy.random.randn(self.output_size)
        if (w.shape == self.weights.shape and b.shape == self.bias.shape):
            self.weights = w
            self.bias = b
        else:
            print "Initializing weights for fully_connected_layer..."
            self.weights = self.weights*numpy.sqrt(2./(self.input_size+1))
            self.bias = self.bias*numpy.sqrt(2./(self.input_size+1))
        self.is_valid_input = False
        self.input_shape = None

    def save_layer(self, file_path):
        pickle.dump(self, open(file_path, 'w'))

    def load_layer(self, file_path):
        print "Loading fully_connected_layer weights and bias..."
        old_one = pickle.load(open(file_path, 'r'))
        self.input_size = old_one.input_size
        self.output_size = old_one.output_size
        self.weights = old_one.weights
        self.bias = old_one.bias
        print "Loading completed..."
        self.is_valid_input = False
        self.input_shape = None

    def check_input_validity(self, input_array):
        if (input_array.flatten().shape[0] == self.weights.shape[1]):
            self.is_valid_input = True
        else:
            self.is_valid_input = False
            print "** Warning: Input dimensions do not match fully_connected_layer **"
            print "Input shape:",input_array.shape
            print "Layer's input_size",self.input_size
        return self.is_valid_input

    def propagate_forward(self, input_array):
        self.input_shape = input_array.shape
        reshaped_input = input_array.reshape(1, self.input_size)
        output = numpy.inner(reshaped_input, self.weights)
        output += self.bias
        return output

    def update_weights(self, gradient_weights, gradient_bias, learning_rate=0.01):
        self.weights = self.weights-learning_rate*gradient_weights
        self.bias = self.bias-learning_rate*gradient_bias

    def propagate_backward(self, error_array, input_array, is_batching=False, learning_rate=0.01):
        reshaped_error = error_array.reshape(1, self.output_size)
        gradient_bias = reshaped_error.flatten()
        gradient_weights = numpy.outer(input_array, reshaped_error).T
        new_error = numpy.inner(self.weights.T, reshaped_error)
        new_error = new_error.reshape(self.input_shape)
        if (is_batching):
            return (new_error, gradient_weights, gradient_bias)
        else:
            self.update_weights(gradient_weights, gradient_bias, learning_rate)
            return new_error

    def regularize_weights(self, lambda_l2=0.01):
        self.weights = self.weights-lambda_l2*numpy.square(self.weights)

class convolutional_layer:
    def __init__(self, input_depth, num_filter, field_size, stride, zero_pad, w, b):
        self.d = input_depth
        self.n = num_filter
        self.f = field_size
        self.s = stride
        self.p = zero_pad
        self.weights = numpy.random.randn(self.n, self.d, self.f, self.f)
        self.bias = numpy.random.randn(self.n)
        if (w.shape == self.weights.shape and b.shape == self.bias.shape):
            self.weights = w
            self.bias = b
        else:
            print "Initializing weights for convolutional_layer..."
            self.weights = self.weights*numpy.sqrt(2./(self.d*self.f*self.f+1))
            self.bias = self.bias*numpy.sqrt(2./(self.d*self.f*self.f+1))
        self.is_valid_input = False
        self.padded_input = None
        self.activation_derivative_mask = None

    def save_layer(self, file_path):
        pickle.dump(self, open(file_path, 'w'))

    def load_layer(self, file_path):
        print "Loading convolutional_layer weights and bias..."
        old_one = pickle.load(open(file_path, 'r'))
        self.d = old_one.d
        self.n = old_one.n
        self.f = old_one.f
        self.s = old_one.s
        self.p = old_one.p
        self.weights = old_one.weights
        self.bias = old_one.bias
        print "Loading completed..."
        self.is_valid_input = False
        self.padded_input = None
        self.activation_derivative_mask = None

    def check_input_validity(self, input_array):
        input_shape = input_array.shape
        input_h = input_shape[1]
        input_w = input_shape[2]
        if (0 == (input_w+input_h-2*self.f+4*self.p)%self.s):
            self.is_valid_input = True
        else:
            self.is_valid_input = False
            print "** Warning: Input dimensions do not match convolutional_layer **"
            print "Input shape:",input_shape
            print "Layer's parameters:"
            print "input_depth:", self.d
            print "num_filter:",self.n
            print "field_size:",self.f
            print "stride:",self.s
            print "zero_pad:",self.p
        return self.is_valid_input

    def propagate_forward(self, input_array, ReLU_alpha=0.01):
        input_shape = input_array.shape
        padded_input_h = input_shape[1]+2*self.p
        padded_input_w = input_shape[2]+2*self.p
        padded_input_h_end = padded_input_h-self.p
        padded_input_w_end = padded_input_w-self.p
        self.padded_input = numpy.zeros((input_shape[0], padded_input_h, padded_input_w))
        self.padded_input[:,self.p:padded_input_h_end,self.p:padded_input_w_end] = input_array
        output_h = ((input_shape[1]-self.f+2*self.p)/self.s)+1
        output_w = ((input_shape[2]-self.f+2*self.p)/self.s)+1
        output_d = self.n
        output = numpy.zeros((output_d, output_h, output_w))
        self.activation_derivative_mask = numpy.ones((output_d, output_h, output_w))
        for filter_cnt in range(0, self.n):
            output[filter_cnt] += self.bias[filter_cnt]*numpy.ones((output_h, output_w))
            for depth_cnt in range(0, self.d):
                # silding-window 2D convolution
                for h_cnt in range(0, output_h):
                    h_begin = h_cnt*self.s
                    h_end = h_begin+self.f
                    for w_cnt in range(0, output_w):
                        w_begin = w_cnt*self.s
                        w_end = w_begin+self.f
                        buff_w = self.weights[filter_cnt][depth_cnt]
                        buff_x = self.padded_input[depth_cnt][h_begin:h_end,w_begin:w_end]
                        output[filter_cnt][h_cnt][w_cnt] += numpy.sum((buff_w*buff_x))
                        # leaky ReLU activation
                        if output[filter_cnt][h_cnt][w_cnt] < 0:
                            output[filter_cnt][h_cnt][w_cnt] *= ReLU_alpha
                            self.activation_derivative_mask[filter_cnt][h_cnt][w_cnt] = ReLU_alpha
        return output

    def update_weights(self, gradient_weights, gradient_bias, learning_rate=0.01):
        self.weights = self.weights-learning_rate*gradient_weights
        self.bias = self.bias-learning_rate*gradient_bias

    def propagate_backward(self, error_array, is_batching=False, learning_rate=0.01):
        masked_error = self.activation_derivative_mask*error_array
        gradient_weights = numpy.zeros(self.weights.shape)
        gradient_bias = numpy.zeros(self.bias.shape)
        new_error = numpy.zeros(self.padded_input.shape)
        error_h = error_array.shape[1]
        error_w = error_array.shape[2]
        for filter_cnt in range(0, self.n):
            gradient_bias[filter_cnt] = (numpy.sum(masked_error[filter_cnt]))/(error_h*error_w)
            for depth_cnt in range(0, self.d):
                for h_cnt in range(0, error_h):
                    h_begin = h_cnt*self.s
                    h_end = h_begin+self.f
                    for w_cnt in range(0, error_w):
                        w_begin = w_cnt*self.s
                        w_end = w_begin+self.f
                        buff_e = masked_error[filter_cnt][h_cnt][w_cnt] # scalar value
                        buff_y = self.padded_input[depth_cnt][h_begin:h_end,w_begin:w_end]
                        gradient_weights[filter_cnt][depth_cnt] += (buff_e*buff_y)
                        buff_w = self.weights[filter_cnt][depth_cnt]
                        new_error[depth_cnt][h_begin:h_end,w_begin:w_end] += (buff_e*buff_w)
        new_error_h_end = new_error.shape[1]-self.p
        new_error_w_end = new_error.shape[2]-self.p
        new_error = new_error[:,self.p:new_error_h_end,self.p:new_error_w_end]
        if (is_batching):
            return (new_error, gradient_weights, gradient_bias)
        else:
            self.update_weights(gradient_weights, gradient_bias, learning_rate)
            return new_error

    def regularize_weights(self, lambda_l2=0.01):
        self.weights = self.weights-lambda_l2*numpy.square(self.weights)

class max_pooling_layer:
    def __init__(self, input_depth, field_size, stride):
        self.d = input_depth
        self.f = field_size
        self.s = stride
        self.is_valid_input = False
        self.input_shape = None
        self.activation_positions = None

    def check_input_validity(self, input_array):
        input_shape = input_array.shape
        input_h = input_shape[1]
        input_w = input_shape[2]
        if (0 == (input_w+input_h-2*self.f)%self.s):
            self.is_valid_input = True
        else:
            self.is_valid_input = False
            print "** Warning: Input dimensions do not match max_pooling_layer **"
            print "Input shape:",input_shape
            print "Layer's parameters:"
            print "input_depth:", self.d
            print "field_size:",self.f
            print "stride:",self.s
        return self.is_valid_input

    def propagate_forward(self, input_array):
        self.input_shape = input_array.shape
        output_h = ((self.input_shape[1]-self.f)/self.s)+1
        output_w = ((self.input_shape[2]-self.f)/self.s)+1
        output_d = self.d
        output = numpy.zeros((output_d, output_h, output_w))
        self.activation_positions = numpy.zeros((output_d, output_h, output_w))
        for depth_cnt in range(0, self.d):
            for h_cnt in range(0, output_h):
                h_begin = h_cnt*self.s
                h_end = h_begin+self.f
                for w_cnt in range(0, output_w):
                    w_begin = w_cnt*self.s
                    w_end = w_begin+self.f
                    buff_x = input_array[depth_cnt][h_begin:h_end,w_begin:w_end]
                    output[depth_cnt][h_cnt][w_cnt] = numpy.max(buff_x)
                    # record the activation position of the sub-input
                    self.activation_positions[depth_cnt][h_cnt][w_cnt] = numpy.argmax(buff_x)
        return output

    def propagate_backward(self, error_array):
        new_error = numpy.zeros(self.input_shape)
        error_h = error_array.shape[1]
        error_w = error_array.shape[2]
        for depth_cnt in range(0, self.d):
            for h_cnt in range(0, error_h):
                h_begin = h_cnt*self.s
                h_end = h_begin+self.f
                for w_cnt in range(0, error_w):
                    w_begin = w_cnt*self.s
                    w_end = w_begin+self.f
                    buff_e = numpy.zeros((self.f*self.f))
                    buff_pos = self.activation_positions[depth_cnt][h_cnt][w_cnt]
                    buff_e[numpy.int32(buff_pos)] = error_array[depth_cnt][h_cnt][w_cnt]
                    buff_e = buff_e.reshape(self.f, self.f)
                    new_error[depth_cnt][h_begin:h_end,w_begin:w_end] += buff_e
        return new_error

class average_pooling_layer:
    def __init(self):
        self.input_shape = None

    def propagate_forward(self, input_array):
        self.input_shape = input_array.shape
        output = numpy.average(input_array, axis=2)
        output = numpy.average(output, axis=1)
        return output # output is 1-D array

    def propagate_backward(self, error_array):
        input_h = self.input_shape[1]
        input_w = self.input_shape[2]
        new_error = numpy.ones(self.input_shape)*(1./(input_h*input_w))
        reshaped_error = error_array.reshape(self.input_shape[0], 1, 1)
        new_error = reshaped_error*new_error
        return new_error

class softmax_layer:
    def __init__(self):
        self.input_shape = None
        self.activation_derivative_mask = None

    def propagate_forward(self, input_array):
        self.input_shape = input_array.shape
        reshaped_input = numpy.exp(input_array.flatten())
        output = reshaped_input/numpy.sum(reshaped_input)
        self.activation_derivative_mask = output*(1-output)
        return output

    def propagate_backward(self, error_array):
        new_error = self.activation_derivative_mask*error_array.flatten()
        new_error = new_error.reshape(self.input_shape)
        return new_error
