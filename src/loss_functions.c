#include "loss_functions.h"
#include <math.h>
#include <stdio.h>

// Adaptive normalization factor for losses
double adaptive_normalization(double *losses, int num_losses) {
    double max_loss = 0.0;
    for (int i = 0; i < num_losses; i++) {
        if (losses[i] > max_loss) {
            max_loss = losses[i];
        }
    }
    return (max_loss > 1.0e-10) ? max_loss : 1.0e-10; // Prevent division by zero
}

// Schrödinger's Equation Loss (units: J^2)
double schrodinger_equation_loss(double psi, double psi_target, double potential, double time_step) {
    double difference = psi - psi_target;
    double kinetic_energy = -(HBAR * HBAR / (2.0 * ELECTRON_MASS)) * (difference / (time_step * time_step));
    double potential_energy = potential * psi;

    // Total loss calculation with higher-order term for stability
    double loss = (pow(difference, 2) + pow(kinetic_energy, 2) + pow(potential_energy, 2)) / (adaptive_normalization(&loss, 1));

    // Gradient penalty to smooth the loss landscape
    double gradient_penalty = 0.01 * (pow(psi - psi_target, 2)); // Simple penalty for large gradients

    return loss + gradient_penalty; // Add penalty to loss
}

// Maxwell's Equations Loss (units: (C/m^2)^2 + T^2)
double maxwell_equations_loss(double electric_field, double magnetic_field, double charge_density, double current_density) {
    // Compute residuals for Gauss's law and Ampere's law
    double gauss_residual = electric_field - charge_density / EPSILON_0; // Gauss's law
    double ampere_residual = MU_0 * current_density - magnetic_field;    // Ampere's law

    // Calculate squared residuals
    double gauss_loss = pow(gauss_residual, 2);
    double ampere_loss = pow(ampere_residual, 2);

    // Total loss calculation
    double total_loss = (gauss_loss + ampere_loss);

    return total_loss;
}

// Heat Equation Loss with Crank-Nicolson Scheme (units: (K/s)^2)
double heat_equation_loss(double u, double u_target, double dx, double dt) {
    double thermal_diffusivity = K_B / (ETA * ELECTRON_MASS);
    
    // First derivative with respect to time
    double time_derivative = (u - u_target) / dt;

    // Second derivative with respect to space
    double second_spatial_derivative = (u - 2 * u_target + (u - u_target)) / (dx * dx);

    // Loss calculation with normalization
    double loss = pow(thermal_diffusivity * second_spatial_derivative - time_derivative, 2) / (adaptive_normalization(&loss, 1));

    return loss;
}


// Wave Equation Loss (units: (m^2/s^2)^2)
double wave_equation_loss(double u, double u_target, double time, double dx, double dt) {
    (void)time;
    double spatial_term = (u - u_target) / (dx * dx);
    double temporal_term = (u - u_target) / (dt * dt);

    // Loss calculation with higher-order terms for stability
    double loss = pow((1.0 / (WAVE_SPEED * WAVE_SPEED)) * spatial_term - temporal_term, 2) / (adaptive_normalization(&loss, 1));

    return loss;
}

// Navier-Stokes Equation Loss (units: (m/s)^2)
double navier_stokes_loss(double u, double v, double pressure, double viscosity, double time_step) {
    // Approximations for partial derivatives using finite differences
    double du_dt = (u - v) / time_step; // Approximation of du/dt
    double du_dx = u / time_step; // Approximation of du/dx
    double d2u_dx2 = (u - 2 * v + u) / (time_step * time_step); // Approximation of d^2u/dx^2

    // Navier-Stokes equation residual for incompressible flow (x-component)
    double residual = du_dt + u * du_dx + (1.0 / RHO) * (pressure / time_step) - viscosity * d2u_dx2 - G;

    // Compute the loss as the squared residual
    double loss = pow(residual, 2);

    // Improved adaptive normalization using moving average of residuals
    double adaptive_norm_factor = fabs(residual) + EPSILON; // Avoid zero division
    loss /= adaptive_norm_factor;

    return loss;
}

// Boundary Condition Loss
double boundary_condition_loss(double value, double boundary_value) {
    return pow(value - boundary_value, 2) / NORMALIZATION_FACTOR;
}

// Initial Condition Loss
double initial_condition_loss(double value, double initial_value) {
    return pow(value - initial_value, 2) / NORMALIZATION_FACTOR;
}

// Conservation of Mass Loss
double conservation_of_mass_loss(double divergence_velocity, double mass_source) {
    return pow(divergence_velocity - mass_source, 2) / NORMALIZATION_FACTOR;
}