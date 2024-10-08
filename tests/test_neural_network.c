#include <stdio.h>
#include "neural_network.h"

void test_initialize_neural_network() {
    // Initialize neural network
    NeuralNetwork nn;
    initialize_neural_network(&nn);
    if (validate_neural_network_initialization(&nn)) {
        printf("Neural Network Initialized and Validated.\n");
    } else {
        printf("Neural Network Initialization Validation Failed.\n");
    }
}

void test_forward_pass() {
    // Initialize neural network
    NeuralNetwork nn;
    initialize_neural_network(&nn);
    double input[2] = {1.0, 2.0}; // Example input for forward pass
    double output[1] = {0}; // Output buffer
    forward_pass(&nn, input, output, RELU); // Perform forward pass
    printf("Forward Pass Output: %f\n", output[0]);
}

int main() {
    test_initialize_neural_network(); // Test initialization
    test_forward_pass(); // Test forward pass
    return 0;
}
