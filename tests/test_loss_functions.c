#include <stdio.h>
#include "loss_functions.h"

void test_schrodinger_equation_loss() {
    double loss = schrodinger_equation_loss(1.0, 0.5, 0.1, 0.01);
    printf("Schrödinger Loss: %f\n", loss);
}

void test_maxwell_equations_loss() {
    double loss = maxwell_equations_loss(1.0, 1.0, 1.0, 0.01);
    printf("Maxwell Loss: %f\n", loss);
}

void test_heat_equation_loss() {
    double loss = heat_equation_loss(1.0, 0.5, 0.1, 0.01);
    printf("Heat Equation Loss: %f\n", loss);
}

void test_wave_equation_loss() {
    double loss = wave_equation_loss(1.0, 0.5, 0.1, 0.1, 0.01);
    printf("Wave Equation Loss: %f\n", loss);
}

void test_navier_stokes_loss() {
    double loss = navier_stokes_loss(1.0, 1.0, 1.0, 1.0, 0.01);
    printf("Navier-Stokes Loss: %f\n", loss);
}

int main() {
    test_schrodinger_equation_loss();
    test_maxwell_equations_loss();
    test_heat_equation_loss();
    test_wave_equation_loss();
    test_navier_stokes_loss();
    return 0;
}
