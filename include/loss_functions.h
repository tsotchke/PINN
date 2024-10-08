#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <math.h>

// Constants
#define ELECTRON_MASS 9.10938356e-31 // Electron mass (kg)
#define HBAR 1.0545718e-34 // Reduced Planck's constant (J·s)
#define EPSILON_0 8.854187817e-12 // Vacuum permittivity (F/m)
#define MU_0 1.2566370614e-6 // Vacuum permeability (N/A^2)
#define K_B 1.380649e-23 // Boltzmann constant (J/K)
#define E 1.602176634e-19 // Elementary charge (C)
#define ETA 1.81e-5 // Dynamic viscosity of air at room temperature (Pa·s)
#define WAVE_SPEED 343.0 // Speed of sound in air (m/s)
#define G 9.81 // Gravitational acceleration (m/s^2)
#define RHO 1.225 // Density of air at sea level (kg/m^3)
#define EPSILON 1e-10 // Small value to prevent division by zero

// Normalization factor for losses (can be adjusted per equation for scale)
#define NORMALIZATION_FACTOR 1.0e-10

// Function declarations
double schrodinger_equation_loss(double psi, double psi_target, double potential, double time_step);
double maxwell_equations_loss(double electric_field, double magnetic_field, double charge_density, double current_density);
double heat_equation_loss(double u, double u_target, double dx, double dt);
double wave_equation_loss(double u, double u_target, double time, double dx, double dt);
double navier_stokes_loss(double u, double v, double pressure, double viscosity, double time_step);
double boundary_condition_loss(double value, double boundary_value);
double initial_condition_loss(double value, double initial_value);
double conservation_of_mass_loss(double divergence_velocity, double mass_source);
double adaptive_normalization(double *losses, int num_losses);

#endif // LOSS_FUNCTIONS_H