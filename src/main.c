#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neural_network.h"
#include "loss_functions.h"
#include "utils.h"

void print_usage() {
    printf("Usage: pinn_neural_network --loss [loss_type] [parameters] --activation [activation_function] --epochs [value] --learning_rate [value]\n");
    printf("Loss types and their parameters:\n");
    printf("  schrodinger: --potential [value]\n");
    printf("  maxwell: --charge_density [value] --current_density [value]\n");
    printf("  heat: --thermal_conductivity [value]\n");
    printf("  wave: --wave_speed [value]\n");
    printf("  navier_stokes: --viscosity [value]\n");
    printf("Activation functions:\n");
    printf("  sigmoid\n");
    printf("  tanh\n");
    printf("  relu\n");
    printf("  leaky_relu\n");
}

int main(int argc, char *argv[]) {
    if (argc < 8) {
        print_usage();
        return EXIT_FAILURE;
    }

    char *loss_type = NULL;
    char *activation_function = NULL;
    double potential = 0.0;
    double charge_density = 0.0;
    double current_density = 0.0;
    double thermal_conductivity = 0.0;
    double wave_speed = 0.0;
    double viscosity = 0.0;
    int epochs = 1000;
    double learning_rate = 0.01;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--loss") == 0 && i + 1 < argc) {
            loss_type = argv[++i];
        } else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) {
            activation_function = argv[++i];
        } else if (strcmp(argv[i], "--potential") == 0 && i + 1 < argc) {
            potential = atof(argv[++i]);
        } else if (strcmp(argv[i], "--charge_density") == 0 && i + 1 < argc) {
            charge_density = atof(argv[++i]);
        } else if (strcmp(argv[i], "--current_density") == 0 && i + 1 < argc) {
            current_density = atof(argv[++i]);
        } else if (strcmp(argv[i], "--thermal_conductivity") == 0 && i + 1 < argc) {
            thermal_conductivity = atof(argv[++i]);
        } else if (strcmp(argv[i], "--wave_speed") == 0 && i + 1 < argc) {
            wave_speed = atof(argv[++i]);
        } else if (strcmp(argv[i], "--viscosity") == 0 && i + 1 < argc) {
            viscosity = atof(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--learning_rate") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        }
    }

    if (loss_type == NULL) {
        fprintf(stderr, "Error: Loss type not specified\n");
        print_usage();
        return EXIT_FAILURE;
    }
    
    if (activation_function == NULL) {
        fprintf(stderr, "Error: Activation function not specified\n");
        print_usage();
        return EXIT_FAILURE;
    }

    // Initialize neural network
    NeuralNetwork nn;
    initialize_neural_network(&nn);
    if (!validate_neural_network_initialization(&nn)) {
        fprintf(stderr, "Neural network initialization failed!\n");
        return EXIT_FAILURE;
    }

    // Create a LossParameters struct to pass to train_neural_network
    LossParameters params = {
        .potential = potential,
        .charge_density = charge_density,
        .current_density = current_density,
        .thermal_conductivity = thermal_conductivity,
        .wave_speed = wave_speed,
        .viscosity = viscosity
    };

    // Train neural network
    train_neural_network(&nn, loss_type, &params, epochs, learning_rate, activation_function);

    // Save trained model
    save_model(&nn, "model_parameters.txt");

    return EXIT_SUCCESS;
}