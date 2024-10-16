#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "neural_network.h"
#include "loss_functions.h"
#include "utils.h"

// Function prototypes
static void log_training_data(const char *loss_type, int epoch, double loss, double validation_loss, int run_number);
static double calculate_validation_loss(NeuralNetwork *nn, double validation_inputs[][INPUT_SIZE], double validation_targets[], int num_validation_samples, const char *loss_type, const LossParameters *params, ActivationFunction activation_func_type);
void save_model(const NeuralNetwork *nn, const char *filename);

// Function to choose activation function
static double activate(double x, ActivationFunction function) {
    double alpha = 0.01;  // Define the leakiness factor
    switch (function) {
        case RELU:
            return (x < 0) ? 0 : x;
        case SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case TANH:
            // Stable calculation for Tanh using exp
            if (x < -20) return -1.0;
            else if (x > 20) return 1.0;
            double exp_pos = exp(x);
            double exp_neg = exp(-x);
            return (exp_pos - exp_neg) / (exp_pos + exp_neg);
        case LEAKY_RELU:
            return (x > 0) ? x : alpha * x;
        default:
            return x; // Default to identity if unknown
    }
}

// Function to calculate the derivative of Tanh
static double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t; // Derivative of Tanh is 1 - tanh^2(x)
}

void initialize_neural_network(NeuralNetwork *nn) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            nn->weights_hidden_output[j][k] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        nn->biases_hidden[j] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        nn->biases_output[k] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

int validate_neural_network_initialization(const NeuralNetwork *nn) {
    if (nn == NULL) return 0;

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            if (nn->weights_input_hidden[i][j] == 0) return 0;
        }
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            if (nn->weights_hidden_output[j][k] == 0) return 0;
        }
        if (nn->biases_hidden[j] == 0) return 0;
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        if (nn->biases_output[k] == 0) return 0;
    }

    return 1;
}

void forward_pass(NeuralNetwork *nn, double input[INPUT_SIZE], double output[OUTPUT_SIZE], ActivationFunction activation_function) {
    double hidden[HIDDEN_SIZE] = {0};

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            hidden[j] += input[i] * nn->weights_input_hidden[i][j];
        }
        hidden[j] += nn->biases_hidden[j];
        hidden[j] = activate(hidden[j], activation_function);
        nn->hidden_outputs[j] = hidden[j]; // Store the hidden output in the structure
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        output[k] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[k] += hidden[j] * nn->weights_hidden_output[j][k];
        }
        output[k] += nn->biases_output[k];
    }
}

static void update_weights(NeuralNetwork *nn, double learning_rate, double input[INPUT_SIZE], double output[OUTPUT_SIZE], double target) {
    double output_error[OUTPUT_SIZE];
    double hidden_error[HIDDEN_SIZE] = {0};

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        output_error[k] = target - output[k];
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            hidden_error[j] += output_error[k] * nn->weights_hidden_output[j][k];
        }
        // Use Tanh derivative here with hidden outputs from the structure
        hidden_error[j] *= tanh_derivative(nn->hidden_outputs[j]); 
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            nn->weights_hidden_output[j][k] += learning_rate * output_error[k] * nn->hidden_outputs[j];
        }
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        nn->biases_output[k] += learning_rate * output_error[k];
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->weights_input_hidden[i][j] += learning_rate * hidden_error[j] * input[i];
        }
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        nn->biases_hidden[j] += learning_rate * hidden_error[j];
    }
}

void train_neural_network(NeuralNetwork *nn, const char *loss_type, const LossParameters *params, int epochs, double learning_rate, const char *activation_function) {
    double validation_inputs[5][INPUT_SIZE] = {
        {1.0, 2.0},
        {1.5, 2.5},
        {2.0, 3.0},
        {2.5, 3.5},
        {3.0, 4.0}
    };

    double validation_targets[5] = {1.0, 1.5, 2.0, 2.5, 3.0};
    int num_validation_samples = 5;

    ActivationFunction activation_func_type;
    if (strcmp(activation_function, "sigmoid") == 0) {
        activation_func_type = SIGMOID;
    } else if (strcmp(activation_function, "tanh") == 0) {
        activation_func_type = TANH;
    } else if (strcmp(activation_function, "relu") == 0) {
        activation_func_type = RELU;
    } else if (strcmp(activation_function, "leaky_relu") == 0) {
        activation_func_type = LEAKY_RELU;
    } else {
        fprintf(stderr, "Error: Unsupported activation function: %s\n", activation_function);
        return;
    }

    char log_filename[256];
    snprintf(log_filename, sizeof(log_filename), "log_%s.txt", loss_type);

    int run_number = 0;
    while (access(log_filename, F_OK) == 0) {
        run_number++;
        snprintf(log_filename, sizeof(log_filename), "log_%s_%d.txt", loss_type, run_number);
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        double adjusted_learning_rate = adaptive_learning_rate(learning_rate, epoch, 0.01);
        double input[INPUT_SIZE] = {1.0, 2.0};
        double target = 1.0;
        double output[OUTPUT_SIZE] = {0};

        forward_pass(nn, input, output, activation_func_type);

        double loss = 0.0;
        if (strcmp(loss_type, "schrodinger") == 0) {
            loss = schrodinger_equation_loss(output[0], target, params->potential, 0.01);
        } else if (strcmp(loss_type, "maxwell") == 0) {
            loss = maxwell_equations_loss(output[0], 0.0, params->charge_density, params->current_density);
        } else if (strcmp(loss_type, "heat") == 0) {
            loss = heat_equation_loss(output[0], target, 0.1, 0.01);
        } else if (strcmp(loss_type, "wave") == 0) {
            loss = wave_equation_loss(output[0], target, 0.0, 0.1, 0.01);
        } else if (strcmp(loss_type, "navier_stokes") == 0) {
            loss = navier_stokes_loss(output[0], output[1], output[2], params->viscosity, 0.01);
        } else {
            fprintf(stderr, "Unknown loss type: %s\n", loss_type);
            return;
        }

        update_weights(nn, adjusted_learning_rate, input, output, target);

        double validation_loss = calculate_validation_loss(nn, validation_inputs, validation_targets, num_validation_samples, loss_type, params, activation_func_type);
        log_training_data(loss_type, epoch, loss, validation_loss, run_number);
    }

    //save_model(nn, "model.txt");
}

static double calculate_validation_loss(NeuralNetwork *nn, double validation_inputs[][INPUT_SIZE], double validation_targets[], int num_validation_samples, const char *loss_type, const LossParameters *params, ActivationFunction activation_func_type) {
    double total_validation_loss = 0.0;

    for (int i = 0; i < num_validation_samples; i++) {
        double output[OUTPUT_SIZE] = {0};
        forward_pass(nn, validation_inputs[i], output, activation_func_type);

        if (strcmp(loss_type, "schrodinger") == 0) {
            total_validation_loss += schrodinger_equation_loss(output[0], validation_targets[i], params->potential, 0.01);
        } else if (strcmp(loss_type, "maxwell") == 0) {
            total_validation_loss += maxwell_equations_loss(output[0], 0.0, params->charge_density, params->current_density);
        } else if (strcmp(loss_type, "heat") == 0) {
            total_validation_loss += heat_equation_loss(output[0], validation_targets[i], 0.1, 0.01);
        } else if (strcmp(loss_type, "wave") == 0) {
            total_validation_loss += wave_equation_loss(output[0], validation_targets[i], 0.0, 0.1, 0.01);
        } else if (strcmp(loss_type, "navier_stokes") == 0) {
            total_validation_loss += navier_stokes_loss(output[0], output[1], output[2], params->viscosity, 0.01);
        }
    }

    return total_validation_loss / num_validation_samples;
}

static void log_training_data(const char *loss_type, int epoch, double loss, double validation_loss, int run_number) {
    char log_filename[256];
    if (run_number == 0) {
        snprintf(log_filename, sizeof(log_filename), "log_%s.txt", loss_type);
    } else {
        snprintf(log_filename, sizeof(log_filename), "log_%s_%d.txt", loss_type, run_number);
    }

    FILE *log_file = fopen(log_filename, "a");
    if (log_file) {
        fprintf(log_file, "Epoch %d: Loss:  %.5f, Validation Loss: %.5f\n", epoch, loss, validation_loss);
        fclose(log_file);
    } else {
        fprintf(stderr, "Failed to open log file: %s\n", log_filename);
    }
}

void save_model(const NeuralNetwork *nn, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for saving model: %s\n", filename);
        return;
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            fprintf(file, "%lf\n", nn->weights_input_hidden[i][j]);
        }
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            fprintf(file, "%lf\n", nn->weights_hidden_output[j][k]);
        }
        fprintf(file, "%lf\n", nn->biases_hidden[j]);
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        fprintf(file, "%lf\n", nn->biases_output[k]);
    }

    fclose(file);
    printf("Model saved to %s\n", filename);
}
