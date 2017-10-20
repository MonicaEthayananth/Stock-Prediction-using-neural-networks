# Ethayananth, Monica Rani
# 1001-417-942
#2016-10-16

import numpy as np
import Tkinter as Tk
import matplotlib
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pdb
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorsys
import scipy.misc
import random
from sklearn.preprocessing import normalize
import numpy as np

import pdb
file_name = "stock_data.csv" #please enter the mnist_images path

def Normalize(data):
    mean_x = np.mean(data[:,0])
    std_x = np.std(data[:,0])
    mean_y = np.mean(data[:,1])
    std_y = np.std(data[:,1])
    data[:,0]=(data[:,0]-mean_x)/std_x
    data[:,1] =(data[:,1]-mean_y)/std_y
    return data

class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self, file_name):
        self.data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
        self.data = normalize(self.data,axis=0)

        self.data = np.transpose(self.data)
nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights": -1.0,  # minimum initial weight
    "max_initial_weights": 1.0,  # maximum initial weight
    "number_of_inputs": 785,  # number of inputs to the network
    "train_network": 0.00001,  # learning rate
    "delayed_elements":1, #number of delayed elements
    "hidden_nodes":100,#select the nodes in the hidden layer
    "number_iterations":1,# number of times system goes over the samples.
    "momentum": 0.1,  # momentum
    "weight_regularization": 1,  # lamda value
    "layers_specification": [{"number_of_neurons": 10, "activation_function": "linear"}],  # list of dictionaries
    "data_set": ClDataSet("stock_data.csv"),
    'number_of_classes': 10,
    'number_of_samples_in_each_class': 3
}


class ClNNExperiment:
    """
    This class presents an experimental setup for a single layer Perceptron
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings={}):
        self.__dict__.update(nn_experiment_default_settings)
        self.__dict__.update(settings)
        # Set up the neural network
        settings = {"min_initial_weights": self.min_initial_weights,  # minimum initial weight
                    "max_initial_weights": self.max_initial_weights,  # maximum initial weight
                    "number_of_inputs": self.number_of_inputs,  # number of inputs to the network
                    "learning_rate": self.learning_rate,  # learning rate
                    "layers_specification": self.layers_specification
                    }
        self.neural_network = ClNeuralNetwork(self, settings)
        # Make sure that the number of neurons in the last layer is equal to number of classes
        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes

    def run_forward_pass(self, display_input=True, display_output=True,
                         display_targets=True, display_target_vectors=True,
                         display_error=True):
        self.neural_network.calculate_output(self.data_set.samples)

        if display_input:
            print "Input : ", self.data_set.samples
        if display_output:
            print 'Output : ', self.neural_network.output
        if display_targets:
            print "Target (class ID) : ", self.target
        if display_target_vectors:
            print "Target Vectors : ", self.desired_target_vectors
        if self.desired_target_vectors.shape == self.neural_network.output.shape:
            self.error = self.desired_target_vectors - self.neural_network.output
            if display_error:
                print 'Error : ', self.error
        else:
            print "Size of the output is not the same as the size of the target.", \
                "Error cannot be calculated."


    def create_samples(self,fraction):
        size = (fraction * self.data_set.data.shape[1])/100
        #print self.data_set.data.shape
        index = random.sample(np.arange(self.data_set.data.shape[1]),size)
        #print index
        self.samples = self.data_set.data[:,index]
        #print self.samples.shape


    def randomize_weights(self,min,max):
        self.neural_network.randomize_weights(min,max)



    def adjust_weights(self,delayed_elements,batch_size):
        mse_arr = []
        mae_arr = []

        for i in range(delayed_elements+1,self.samples.shape[1]-1,batch_size):
            j = min(self.samples.shape[1]-1,i+batch_size)
            for s in  range(i,j):

                x = self.samples[:,s-delayed_elements-1:s]
                y = self.samples[:,s]

                x = np.append(np.append(x[0,:],x[1,:]),1.0).reshape(-1,1)
                y = y.reshape(-1,1)
                predict = self.neural_network.calculate_output(x)
                error = np.subtract(y,predict)
                self.neural_network.adjust_weights(x,y,error)
            mse = self.neural_network.calculate_error(delayed_elements,self.samples[:,i:j],"mse")
            mae = self.neural_network.calculate_error(delayed_elements,self.samples[:,i:j],"mae")
            #print "mean square error",mse
            #print "max abs error",mae
            mse_arr.append(mse)
            mae_arr.append(mae)
        return mse_arr,mae_arr





class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    Farhad Kamangar 2016_09_02
    """

    def __init__(self, master, nn_experiment):
        self.master = master
        #
        self.nn_experiment = nn_experiment
        self.number_of_classes = self.nn_experiment.number_of_classes
        self.xmin = 0
        self.xmax = 10
        self.ymin = 0
        self.ymax = 0.7
        self.master.update()
        self.number_of_samples_in_each_class = self.nn_experiment.number_of_samples_in_each_class
        self.delayed_elements=self.nn_experiment.delayed_elements
        self.learning_rate = self.nn_experiment.learning_rate
        self.sample_size_percentage=self.nn_experiment.sample_size_percentage
        self.batch_size=self.nn_experiment.batch_size
        self.number_iterations=self.nn_experiment.number_iterations
        self.sample_size_percentage = self.nn_experiment.sample_size_percentage
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.step_size = 1
        self.current_sample_loss = 0
        self.sample_points = []
        self.target = []
        self.sample_colors = []
        self.weights = np.array([])
        self.class_ids = np.array([])
        self.output = np.array([])
        self.desired_target_vectors = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.loss_type = ""
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("widrow-huff Learning")
        self.axes = self.figure.add_subplot(111)
        self.figure = plt.figure("widrow-huff Learning")
        self.axes = self.figure.add_subplot(111)
        plt.title("widrow-huff Learning")

        plt.xlim(self.xmin,self.xmax)
        plt.ylim(self.ymin,self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        # Set up the sliders
        ivar = Tk.IntVar()
        # slider for delayed elements
        self.delayed_elements_slider_label = Tk.Label(self.sliders_frame, text="Delayed Elements")
        self.delayed_elements_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.delayed_elements_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=10, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.delayed_elements_slider_callback())
        self.delayed_elements_slider.set(self.delayed_elements)
        self.delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.delayed_elements_slider_callback())
        self.delayed_elements_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

# slider for adjusting learning rate
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                                from_=0.00001, to_=1, resolution=0.00001, bg="#DDDDDD",
                                                activebackground="#FF0000",
                                                highlightcolor="#00FFFF", width=10,
                                                command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
# slider for sample percentage
        self.sample_size_percentage_slider_label = Tk.Label(self.sliders_frame, text="Sample size percentage")
        self.sample_size_percentage_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sample_size_percentage_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0, to_=100, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.sample_size_percentage_slider_callback())
        self.sample_size_percentage_slider.set(self.sample_size_percentage)
        self.sample_size_percentage_slider.bind("<ButtonRelease-1>", lambda event: self.sample_size_percentage_slider_callback())
        self.sample_size_percentage_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        #slider for batch size
        self.batch_size_slider_label = Tk.Label(self.sliders_frame, text="batch size")
        self.batch_size_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batch_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0, to_=500, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        #slider for iteration range
        self.number_iterations_slider_label = Tk.Label(self.sliders_frame, text="Iteration range")
        self.number_iterations_slider_label.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_iterations_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=10,  bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.number_iterations_slider_callback())
        self.number_iterations_slider.set(self.number_iterations)
        self.number_iterations_slider.bind("<ButtonRelease-1>", lambda event: self.number_iterations_slider_callback())
        self.number_iterations_slider.grid(row=4, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_classes_slider_label = Tk.Label(self.sliders_frame, text="Number of Classes")
        # self.number_of_classes_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_classes_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
        #                                          from_=2, to_=5, bg="#DDDDDD",
        #                                          activebackground="#FF0000",
        #                                          highlightcolor="#00FFFF", width=10)
        # self.number_of_classes_slider.set(self.number_of_classes)
        # self.number_of_classes_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_classes_slider_callback())
        # self.number_of_classes_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_samples_slider = Tk.Scale(self.sliders_frame, variable=ivar, orient=Tk.HORIZONTAL,
        #                                          from_=2, to_=20, bg="#DDDDDD",
        #                                          activebackground="#FF0000",
        #                                          highlightcolor="#00FFFF", width=10)
        # self.number_of_samples_slider_label = Tk.Label(self.sliders_frame, text="Number of Samples")
        # self.number_of_samples_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_samples_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_samples_slider_callback())
        # self.number_of_samples_slider.set(self.number_of_samples_in_each_class)
        # self.number_of_samples_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.create_new_samples_bottun = Tk.Button(self.buttons_frame,
        #                                            text="Create New Samples",
        #                                            bg="yellow", fg="red",
        #                                            command=lambda: self.create_new_samples_bottun_callback())
        # self.create_new_samples_bottun.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.randomize_weights_button = Tk.Button(self.buttons_frame,
        #                                           text="Randomize Weights",
        #                                           bg="yellow", fg="red",
        #                                           command=lambda: self.randomize_weights_button_callback())
        #self.randomize_weights_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.learning_method_variable = Tk.StringVar()
        # self.learning_method_dropdown = Tk.OptionMenu(self.buttons_frame, self.learning_method_variable,
        #                                               "Filtered Learning",
        #                                               "Delta Rule", "Unsupervised Learning",
        #                                               command=lambda event: self.learning_method_dropdown_callback())
        # self.learning_method_variable.set("Filtered Learning")
        # self.learning_rule = "Filtered Learning"
        # self.learning_method_dropdown.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.print_nn_parameters_button = Tk.Button(self.buttons_frame,
                                                    text="Print NN Parameters",
                                                    bg="yellow", fg="red",
                                                    command=lambda: self.print_nn_parameters_button_callback())
        self.print_nn_parameters_button.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.x = []
        self.y = []

        self.reset_button = Tk.Button(self.buttons_frame, text="set to zero",
                                      bg="red", fg="yellow",
                                      command=lambda: self.reset_button_callback())
        self.reset_button.grid(row=4,column=0,sticky=Tk.N+Tk.E+Tk.S+Tk.W)
        self.current_epoch=0
        self.initialize()

        #self.refresh_display()

    def initialize(self):
        #self.nn_experiment.create_samples()
        self.x_axis_main = []
        self.price_mse_main = []
        self.price_mae_main  = []
        self.volume_mse_main = []
        self.volume_mae_main = []
        red_patch = mpatches.Patch(color="red", label="price_mse")
        blue_patch = mpatches.Patch(color="blue", label="volume_mse")
        yellow_patch = mpatches.Patch(color="yellow", label="price_mae")
        green_patch = mpatches.Patch(color="green", label="volume_mae")
        plt.legend(handles=[red_patch, blue_patch,yellow_patch,green_patch])
        self.nn_experiment.neural_network.randomize_weights()
        self.neighborhood_colors = plt.cm.get_cmap('Accent')
        self.sample_points_colors = plt.cm.get_cmap('Dark2')
        # self.xx, self.yy = np.meshgrid(np.arange(self.xmin, self.xmax + 0.5 * self.step_size, self.step_size),
        #                                np.arange(self.ymin, self.ymax + 0.5 * self.step_size, self.step_size))
        self.convert_binary_to_integer = []
        for k in range(0, self.nn_experiment.neural_network.layers[-1].number_of_neurons):
            self.convert_binary_to_integer.append(2 ** k)

    def delayed_elements_slider_callback(self):
        self.delayed_elements = self.delayed_elements_slider.get()
        self.nn_experiment.delayed_elements = self.delayed_elements
        self.nn_experiment.neural_network.delayed_elements = self.delayed_elements
        self.nn_experiment.neural_network.number_of_inputs = self.delayed_elements*2 +3
        self.nn_experiment.neural_network.randomize_weights()
        #print self.delayed_elements
    def sample_size_percentage_slider_callback(self):
        self.sample_size_percentage = self.sample_size_percentage_slider.get()
        self.nn_experiment.sample_size_percentage = self.sample_size_percentage
        self.nn_experiment.neural_network.sample_size_percentage = self.sample_size_percentage
        #print self.sample_size_percentage
    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()
        self.nn_experiment.batch_size = self.batch_size
        self.nn_experiment.neural_network.batch_size = self.batch_size
        #print self.batch_size
    def number_iterations_slider_callback(self):
        self.number_iterations = self.number_iterations_slider.get()
        self.nn_experiment.number_iterations = self.number_iterations
        self.nn_experiment.neural_network.number_iterations = self.number_iterations
        #print self.number_iterations



    # def learning_method_dropdown_callback(self):
    #
    #     self.learning_rule  = self.learning_method_variable.get()

    def refresh_display(self,mse_arr,mae_arr,start):
        x_axis = np.arange(start-1,start,1.0/len(mse_arr))
        self.x_axis_main.extend(x_axis)
        price_mse =[i[0] for i in mse_arr]
        volume_mse = [i[1] for i in mae_arr]
        price_mae = [i[0] for i in mae_arr]
        volume_mae = [i[1] for i in mae_arr]
        self.price_mse_main.extend(price_mse)
        self.price_mae_main.extend(price_mae)
        self.volume_mse_main.extend(volume_mse)
        self.volume_mae_main.extend(volume_mae)
        plt.plot(self.x_axis_main,self.price_mse_main,"r-",label="price_mse")
        plt.plot(self.x_axis_main,self.volume_mse_main,"b-",label="volume_mse")
        plt.plot(self.x_axis_main,self.price_mae_main,"y-",label="price_mae")
        plt.plot(self.x_axis_main,self.volume_mae_main,"g-",label="volume_mae")
        self.canvas.draw()



    def display_neighborhoods(self):
        input_samples= self.nn_experiment.data_set.samples
        targets = self.nn_experiment.data_set.desired_target_vectors
        outputs = self.nn_experiment.neural_network.calculate_output(input_samples)
        #print outputs.shape,targets.shape
        def to_integer(vector):
            return np.where(vector==max(vector))
        count = 0
        size = input_samples.shape[1]
        for i in range(size):
            #pdb.set_trace()
            if to_integer(targets[:,i]) == to_integer(outputs[:,i]):
                count+=1
                #print to_integer(outputs[:,i])
        error = float(size-count)*100.0/size
        self.x.append(self.current_epoch)
        self.y.append(error)
        plt.plot(self.x,self.y,"b")
        self.canvas.draw()

    def initialize_plot_variables(self):
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)



    def reset_button_callback(self):

        self.nn_experiment.randomize_weights(min=0.0,max=0.0)

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        print self.learning_rate



    def adjust_weights_button_callback(self):
        self.nn_experiment.create_samples(self.sample_size_percentage)
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')

        for k in range(self.number_iterations):

            mse_arr,mae_arr = self.nn_experiment.adjust_weights(self.delayed_elements,self.batch_size)
            self.refresh_display(mse_arr,mae_arr,k+1)
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()

    def randomize_weights_button_callback(self):
        temp_text = self.randomize_weights_button.config('text')[-1]
        self.randomize_weights_button.config(text='Please Wait')
        self.randomize_weights_button.update_idletasks()
        self.nn_experiment.neural_network.randomize_weights()
        # self.nn_experiment.neural_network.display_network_parameters()
        # self.nn_experiment.run_forward_pass()
        #self.refresh_display()
        self.randomize_weights_button.config(text=temp_text)
        self.randomize_weights_button.update_idletasks()

    def print_nn_parameters_button_callback(self):
        temp_text = self.print_nn_parameters_button.config('text')[-1]
        self.print_nn_parameters_button.config(text='Please Wait')
        self.print_nn_parameters_button.update_idletasks()
        self.nn_experiment.neural_network.display_network_parameters()
        #self.refresh_display()
        self.print_nn_parameters_button.config(text=temp_text)
        self.print_nn_parameters_button.update_idletasks()


neural_network_default_settings = {
    # Optional settings
    "min_initial_weights": -1.0,  # minimum initial weight
    "max_initial_weights": 1.0,  # maximum initial weight
    "number_of_inputs": 785,  # number of inputs to the network
    "learning_rate": 0.1,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 10,
                              "activation_function": "linear"}]  # list of dictionaries
}


class ClNeuralNetwork:
    """
    This class presents a multi layer neural network
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, experiment, settings={}):
        self.__dict__.update(neural_network_default_settings)
        self.__dict__.update(settings)
        # create nn
        self.experiment = experiment
        self.layers = []
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def randomize_weights(self, min=-0.1, max=0.1):
        # randomize weights for all the connections in the network
        self.layers = []
        layer = self.layers_specification[0]
        layer["number_of_inputs_to_layer"] = self.number_of_inputs
        self.layers.append(ClSingleLayer(layer))
        print "lllllll",self.layers_specification[0]["number_of_inputs_to_layer"]
        for layer in self.layers:
            layer.randomize_weights(min, max)

    def display_network_parameters(self, display_layers=True, display_weights=True):
        for layer_index, layer in enumerate(self.layers):
            print "\n--------------------------------------------", \
                "\nLayer #: ", layer_index, \
                "\nNumber of Nodes : ", layer.number_of_neurons, \
                "\nNumber of inputs : ", self.layers[layer_index].number_of_inputs_to_layer, \
                "\nActivation Function : ", layer.activation_function, \
                "\nWeights : ", layer.weights

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(output)
        self.output = output
        return self.output

    def calculate_error(self,delayed_elements,x,type_of_class="mse"):

        if type_of_class =="mse":
            mse = []
            for i in range(delayed_elements+1,x.shape[1]-1):
                xx = x[:,i-delayed_elements-1:i]
                xx = np.append(np.append(xx[0,:],xx[1,:]),1.0)

                predict = self.calculate_output(xx.reshape(-1,1))
                error = np.subtract(x[:,i].reshape(-1,1),predict)**2
                mse.append(error)
            mse = np.array(mse)
            return np.mean(mse,axis=0)
        else:
            mae = []

            for i in range(delayed_elements+1,x.shape[1]-1):
                xx = x[:, i - delayed_elements - 1:i]
                xx= np.append(np.append(xx[0, :], xx[1, :]), 1.0)
                #print xx.reshape(-1,1).shape
                predict = self.calculate_output(xx.reshape(-1,1))
                error = np.abs(np.subtract(x[:,i].reshape(-1,1),predict))
                mae.append(error)

            mae = np.array(mae)
            return np.max(mae,axis=0)






    def adjust_weights(self, input_samples, targets,error):

        for index,layer in enumerate(self.layers):
            layer.weights = layer.weights + 2* self.learning_rate * error * input_samples.T
            #print layer.weights


single_layer_default_settings = {
    # Optional settings
    "min_initial_weights": -1.0,  # minimum initial weight
    "max_initial_weights": 1.0,  # maximum initial weight
    "number_of_inputs_to_layer": 785,  # number of input signals
    "number_of_neurons": 10,  # number of neurons in the layer
    "activation_function": "linear"  # default activation function
}


class ClSingleLayer:
    """
    This class presents a single layer of neurons
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings):
        self.__dict__.update(single_layer_default_settings)
        self.__dict__.update(settings)
        self.randomize_weights()

    def randomize_weights(self, min_initial_weights=None, max_initial_weights=None):
        print "set to zero",self.number_of_inputs_to_layer
        if min_initial_weights == None:
            min_initial_weights = self.min_initial_weights
        if max_initial_weights == None:
            max_initial_weights = self.max_initial_weights
        self.weights = np.random.uniform(min_initial_weights, max_initial_weights,
                                         (self.number_of_neurons, self.number_of_inputs_to_layer))
        print self.weights
    def calculate_output(self, input_values):
        net = np.dot(self.weights,input_values)
        if self.activation_function == 'linear':
            self.output = net
        if self.activation_function == 'sigmoid':
            self.output = sigmoid(net)
        if self.activation_function == 'hardlimit':
            np.putmask(net, net > 0, 1)
            np.putmask(net, net <= 0, 0)
            self.output = net
        return self.output


if __name__ == "__main__":
    nn_experiment_settings = {
        "min_initial_weights": -1.0,  # minimum initial weight
        "max_initial_weights": 1.0,  # maximum initial weight
        "number_of_inputs": 5,  # number of inputs to the network
        "learning_rate": 0.001,  # learning rate
        "layers_specification": [{"number_of_neurons": 2, "activation_function": "linear"}],  # list of dictionaries
        "data_set": ClDataSet("stock_data.csv"),
        'number_of_classes': 2,
         "batch_size":100,
        "delayed_element":5,
        "number_iterations":1,
        "sample_size_percentage":100,

        'number_of_samples_in_each_class': 3
    }
    np.random.seed(1)
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    main_frame = Tk.Tk()
    main_frame.title("widrow-huff learning")
    main_frame.geometry('650x760')
    ob_nn_gui_2d = ClNNGui2d(main_frame, ob_nn_experiment)
    main_frame.mainloop()
