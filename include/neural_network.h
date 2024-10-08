#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

// Define sizes
#define INPUT_SIZE 2
#define HIDDEN_SIZE 5
#define OUTPUT_SIZE 3  // Moved to header for global availability


typedef struct {
    double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE];
    double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
    double biases_hidden[HIDDEN_SIZE];
    double biases_output[OUTPUT_SIZE];
    double hidden_outputs[HIDDEN_SIZE]; // Add this line to store hidden outputs
} NeuralNetwork;

typedef struct {
    double potential;
    double charge_density;
    double current_density;
    double thermal_conductivity;
    double wave_speed;
    double viscosity;
} LossParameters;

typedef enum {
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU
} ActivationFunction;

void initialize_neural_network(NeuralNetwork *nn);
int validate_neural_network_initialization(const NeuralNetwork *nn);
void forward_pass(NeuralNetwork *nn, double input[INPUT_SIZE], double output[OUTPUT_SIZE], ActivationFunction activation_function);
void train_neural_network(NeuralNetwork *nn, const char *loss_type, const LossParameters *params, int epochs, double learning_rate, const char *activation_function);
void save_model(const NeuralNetwork *nn, const char *filename);

#endif // NEURAL_NETWORK_H