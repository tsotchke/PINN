#include "utils.h"
#include <math.h>

// Define the function here if not already defined
double adaptive_learning_rate(double initial_rate, int epoch, double decay_rate) {
    return initial_rate / (1.0 + decay_rate * epoch);
}