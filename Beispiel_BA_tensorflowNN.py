
########################################################################
#                            Imports                                   #
########################################################################

import tensorflow as tf
import numpy as np

########################################################################
#                            Data                                      #
########################################################################

#Upload inputdata from a .csv file, first two columns are features and the last one is the corresponding label
inputdata=np.genfromtxt('C:/Users/David/Documents/Semester VIII/BA/Excel files/Inputdata Beispiel XOR/InputdataExampleXOR.csv', dtype=np.float32,skip_header=1,delimiter=";")
#Data for validation, with this unknown data the performance of the NN will be tested 
valdata=np.genfromtxt('C:/Users/David/Documents/Semester VIII/BA/Excel files/Inputdata Beispiel XOR/validation_dataExampleXOR.csv', dtype=np.float32,skip_header=1,delimiter=";")


#Shuffle Data for validation:
#Create random sequence for rows of valdata
permuation_valdata=np.random.permutation(np.shape(valdata)[0])
valdata=valdata[permuation_valdata,:]


#Number of features of one inputdata entry
num_features=np.shape(inputdata)[1]-1

#Placeholder for running NN, x is for inputdata
x = tf.placeholder("float")
#Placeholder for running NN, y is for labels corresponding to inputdata
y= tf.placeholder("float")

########################################################################
#                       Parameter setting                              #
########################################################################

#Set seed for replicable results
set_seed=10
#Number epochs
num_epochs=300
#Learningrate
lr=0.001
#Number minibatches
number_minibatches=5
#Number hidden nodes for hiddenlayer 1
nodes_h1 = 20 
#Number hidden nodes for hiddenlayer 2
nodes_h2 = 10

########################################################################
#                          Minibatches method                          #
########################################################################

#Method shuffles dataset and creates "n" minibatches with same size
def create_minibatches(data,number_minibatches):

    #Number of rows at data input matrix
    num_rows_input=np.shape(data)[0]
    size_minibatch=int(num_rows_input/number_minibatches)

    np.random.seed(set_seed)

    #Shuffle rows of the datainput matrix in a random order
    permutation = np.random.permutation(num_rows_input)
    shuffled_data = data[permutation, :]

    #adjust dataset to various minibatch size
    #j is the number datapoints which can be used
    j=size_minibatch*number_minibatches
    #Shorten the inputdata matrix to get a integer number for splitting
    shuffled_data=shuffled_data[0:j, :]

    #Empty array to save the minibatches
    minibatches = []

    #split shuffled_data into minibatches
    minibatches=np.vsplit(shuffled_data,number_minibatches)
    
    return minibatches

########################################################################
#                       Neuronal Network                               #
########################################################################

#Initializing Weights matrices with random values
W1=tf.Variable(tf.random_normal([num_features,nodes_h1],seed=set_seed))
W2=tf.Variable(tf.random_normal([nodes_h1,nodes_h2],seed=set_seed))
#Output has only one column
W3=tf.Variable(tf.random_normal([nodes_h2,1],seed=set_seed))

#Initializing bias matrices with random values
b1=tf.Variable(tf.random_normal([nodes_h1],seed=set_seed))
b2=tf.Variable(tf.random_normal([nodes_h2],seed=set_seed))
#Output has only one column
b3=tf.Variable(tf.random_normal([1],seed=set_seed))

#Actual NN
def run_nn(x):
    #Matrix multiplication between inputmatrix and
    #weightmatrix W1,activationfunction relu
    hiddenlayer_1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
    #Matrix multiplication between result from above matrix multiplication and
    #weightmatrix W2,activationfunction relu
    hiddenlayer_2 = tf.nn.relu(tf.add(tf.matmul(hiddenlayer_1, W2), b2))
    ##Matrix multiplication between result from above matrix multiplication and
    #weightmatrix W3, linear activation
    output_layer = tf.add(tf.matmul(hiddenlayer_2, W3), b3)
    return output_layer

#Defining loss function as MSE between predicted values by NN and given labels
loss=tf.losses.mean_squared_error(labels=y,predictions=run_nn(x))
#Adam is used as a optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

########################################################################
#                            Main                                      #
########################################################################

# Initializing the variables
init = tf.global_variables_initializer()

#Initialize a graph session
sess = tf.Session()

#Run initializer
sess.run(init)

for epoch in range(num_epochs):
    #Split input dataset into minibatches
    mini_batches=create_minibatches(inputdata,number_minibatches)
    
    #One loop iteration is for one minibatch
    #We are iterating thorugh the tupel ( 5, 200,3) with 5=number_minibatches
    for minibatch_variable in mini_batches:
        
        #The first two columns contain inputdata
        input_minibatch_X=minibatch_variable[:,0:2]
        
        #the last column contains the labels
        label_minibatch_X=minibatch_variable[:,2]
        #Reshape the labels to a (size_minibatch,1) array
        label_minibatch_X=np.reshape(label_minibatch_X,[-1,1])

        #The placeholders in "loss" are initialized/
        #"feeded" with input features(placeholder x) and the corresponding labels(placeholder y)
        #The optimizer minimizes the loss
        feed_dict_batch = {x: input_minibatch_X, y: label_minibatch_X}
        #Run optimization to update weights
        sess.run(optimizer, feed_dict=feed_dict_batch)

    #Calculate and print current error per minibatch every "n" epochs
    if epoch %20==0:
        current_loss=sess.run(loss, feed_dict=feed_dict_batch)
        print("Epoch: "+str(epoch)+" current loss: "+str(current_loss))

########################################################################
#                            Validation                                #
########################################################################

#Split valdata 
#The first two columns contain inputdata
val_input=valdata[:,0:2]
#the last column contains the labels
val_label=valdata[:,2]
#Reshape val_label into array with dim ( number rows of val_input ,1)
val_label=np.reshape(val_label,[-1,1])

#Run neuronal network with unknown validation data, as the weights were
#optimized before with training iterations
predictions_val=sess.run(run_nn(x),feed_dict={x:val_input})

#Print output predicted by the NN
print("printed predictions_val: ")
print(predictions_val)


#Round predicted output
predictions_val=np.round(predictions_val,decimals=0)

#Print results for checking
print("predicted val rounded: ")
print(predictions_val)

#Print the real labels of the input .csv testing file
print("REAL Labels of testing dataset: ")
print(val_label)

#Initial value for counting the correct predicted labels
i=0

#Compare rounded predicted output to real label
for k in range(0,len(predictions_val)):
     
    #Increase counter i by 1 if the correct label was predicted
    if predictions_val [k]==val_label[k]:
            i=i+1

#Calculate and print accuracy of correct predicted labels
print("Accuracy in percent: ")
print(str((i/len(predictions_val))*100))




