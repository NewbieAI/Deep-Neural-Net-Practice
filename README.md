# Deep-Neural-Net-Practice
Practice building different deep neural net models starting with simple problems

April 29th update:
-Created a simple NeuralNet framework using the Numpy library:
  1) Enables creation of deep neural networks
  2) User can specify the depth of the network and the width of each layer
  3) XOR_test uses NeuralNet to attempt to learn the XOR function
  4) Convergence is observed for 2 bit and 3 bit XOR functions
  
-To see the framework in action:
  1) put generate_data.py, NeuralNet.py, and XOR_Test.py in the same directory
  2) run the XOR_Test module, the TrainingError vs Iterations graph will appear
  
-Known problems:
  1) The network does not always converge the where you want!!
  2) The result of convergence is highly sensitive to the initial weights
  3) Oscillation in the training error is observed for some learning rates
