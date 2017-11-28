
import unittest
import numpy as np
import pandas as pd
from my_answers import NeuralNetwork

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

data_path = 'Bike-Sharing-Dataset/hour.csv'
pd.read_csv(data_path)

class TestMethods(unittest.TestCase):

    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_forward_pass(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        final_out, hidden_out = network.forward_pass_train(inputs)
        self.assertIsNotNone(final_out)
        self.assertIsNotNone(hidden_out)
        self.assertEqual((1,2),np.shape(hidden_out))
        self.assertEqual((1,1),np.shape(final_out))
        
    def test_backprop(self):
        network = NeuralNetwork(3, 4, 1, 0.5)
        final_out, hidden_out = network.forward_pass_train(inputs)
        deltawOut = np.zeros(np.shape(network.weights_hidden_to_output))
        #self.assertEqual(np.shape(deltawOut),np.shape(test_w_h_o))
        deltawHidden = np.zeros(np.shape(network.weights_input_to_hidden))
        #self.assertEqual(np.shape(deltawHidden), np.shape(test_w_i_h))
        deltaWIH, deltaWHO = network.backpropagation(final_out, hidden_out, inputs, targets, deltawHidden,deltawOut)
        self.assertIsNotNone(deltaWIH)
        self.assertIsNotNone(deltaWHO)

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328],[-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014, -0.20185996], 
                                              [0.39775194, 0.50074398], 
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
    unittest.TextTestRunner().run(suite)

