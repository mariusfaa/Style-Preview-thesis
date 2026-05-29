%% Clear Workspace and Add Paths
clearvars -except ocp_solver sim_solver; clc; close all;

% Windows root (matches workingSimulatorReducedOrder.m)
acados_root  = 'C:\Users\mariujf\acados';
project_root = 'C:\Users\mariujf\model_mismatch';

setenv('ACADOS_SOURCE_DIR',      acados_root);
setenv('ENV_ACADOS_INSTALL_DIR', acados_root);
setenv('ACADOS_INSTALL_DIR',     acados_root);

addpath(fullfile(acados_root, 'interfaces', 'acados_matlab_octave'));
addpath(fullfile(acados_root, 'external',   'jsonlab'));
addpath(fullfile(acados_root, 'external',   'casadi-matlab'));

addpath(genpath(fullfile(project_root, 'model_implementations')));
addpath(genpath(fullfile(project_root, 'system_parameters')));
addpath(genpath(fullfile(project_root, 'utilities')));

import casadi.*

%% Simulation parameters
simulationTime = 2;        % seconds
umax           = 1;        % Max current in Amps

% NMPC discretisation (matches workingSimulatorReducedOrder.m)
N      = 10;
Tf     = 0.05;
dt_mpc = Tf / N;           % 0.005 s — also the plant step
Ts     = dt_mpc;

% FAILURE bounds (10-state: x, y, z, alpha, beta, vx, vy, vz, wx, wy)
upper_bounds = [
    0.05; 0.05; 0.15;
    pi/3; pi/3;
    5.0; 5.0; 5.0;
    inf; inf;
];
lower_bounds = -upper_bounds;
lower_bounds(3) = 0.005;     % Must not drop to the floor

%% Load system parameters
parameters_maggy_V4;
modelName = 'fast';
correctionFactor = computeSolenoidRadiusCorrectionFactor(params, modelName);
params.solenoids.r = correctionFactor * params.solenoids.r;
nominal_mass = params.magnet.m;

%% Symbolic setup
nx_full = 12;
nx      = 10;
nu      = 4;

x_full = SX.sym('x_full', nx_full);
u_sym  = SX.sym('u', nu);

% --- Nominal full-order dynamics (used for equilibrium + controller) ---
f_expl_full_nom = maglevSystemDynamicsCasADi(x_full, u_sym, params);
f_func_full_nom = casadi.Function('f_full_nom', {x_full, u_sym}, {f_expl_full_nom});

%% Equilibrium (nominal mass)
fprintf('--- Searching for equilibrium z with u = 0 ---\n');
z_var       = SX.sym('z_eq');
u_zero      = zeros(nu, 1);
x_eq_sym    = [0; 0; z_var; zeros(9,1)];
accel       = f_func_full_nom(x_eq_sym, u_zero);
fz_residual = accel(9);

nlp       = struct('x', z_var, 'f', fz_residual^2);
solver_eq = nlpsol('solver_eq', 'ipopt', nlp, ...
    struct('ipopt', struct('print_level', 0)));
sol     = solver_eq('x0', 0.030, 'lbx', 0.015, 'ubx', 0.060);
zEq_cas = full(sol.x);
uEq     = zeros(nu, 1);
xEq     = [0; 0; zEq_cas; zeros(7,1)];     % 10-state equilibrium
fprintf('Equilibrium: z = %.6f m\n', zEq_cas);

%% Reduced-order state symbols
x_r    = SX.sym('x_r', nx);
xdot_r = SX.sym('xdot_r', nx);
x_full_from_r = [x_r(1:5); 0; x_r(6:8); x_r(9:10); 0];

%% Controller dynamics (nominal mass)
dx_full_nom = f_func_full_nom(x_full_from_r, u_sym);
f_expl_r_nom = [dx_full_nom(1:5); dx_full_nom(7:11)];

%% Plant dynamics (mass as runtime PARAMETER)
m_sym = SX.sym('m');
params_plant_sym = params;
params_plant_sym.magnet.m = m_sym;
f_expl_full_plant = maglevSystemDynamicsCasADi(x_full, u_sym, params_plant_sym);
f_func_full_plant = casadi.Function('f_full_plant', {x_full, u_sym, m_sym}, {f_expl_full_plant});

dx_full_plant = f_func_full_plant(x_full_from_r, u_sym, m_sym);
f_expl_r_plant = [dx_full_plant(1:5); dx_full_plant(7:11)];

%% --- OCP SETUP (controller — nominal, mirrors workingSimulatorReducedOrder.m) ---
Q = diag([1e2, 1e2, 1e3, ...
          1e3, 1e3, ...
          1e1*0.3, 1e1*0.3, 1e1*0.3, ...
          1e1*0.3, 1e1*0.3]);
R = eye(nu) * 0.1;

ocp = AcadosOcp();
ocp.model.name        = 'maglev_nmpc_mismatch';
ocp.model.x           = x_r;
ocp.model.u           = u_sym;
ocp.model.xdot        = xdot_r;
ocp.model.f_impl_expr = xdot_r - f_expl_r_nom;

ocp.solver_options.N_horizon             = N;
ocp.solver_options.tf                    = Tf;
ocp.solver_options.integrator_type       = 'IRK';
ocp.solver_options.sim_method_num_stages = 1;
ocp.solver_options.sim_method_num_steps  = 1;
ocp.solver_options.nlp_solver_type       = 'SQP_RTI';
ocp.solver_options.nlp_solver_tol_stat   = 1e-4;
ocp.solver_options.nlp_solver_tol_eq     = 1e-4;
ocp.solver_options.nlp_solver_tol_ineq   = 1e-4;
ocp.solver_options.nlp_solver_tol_comp   = 1e-4;
ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM';
ocp.solver_options.qp_solver_iter_max    = 200;
ocp.solver_options.qp_solver_warm_start  = 1;
ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON';
ocp.solver_options.regularize_method     = 'CONVEXIFY';

ocp.cost.cost_type   = 'NONLINEAR_LS';
ocp.cost.cost_type_0 = 'NONLINEAR_LS';
ocp.cost.cost_type_e = 'NONLINEAR_LS';
ocp.cost.W           = blkdiag(Q, R);
ocp.cost.W_0         = blkdiag(Q, R);
ocp.cost.W_e         = Q * 3;

ocp.model.cost_y_expr   = [x_r; u_sym];
ocp.model.cost_y_expr_0 = [x_r; u_sym];
ocp.model.cost_y_expr_e = x_r;

ocp.cost.yref   = [xEq; uEq];
ocp.cost.yref_0 = [xEq; uEq];
ocp.cost.yref_e = xEq;

ocp.constraints.idxbu = 0:nu-1;
ocp.constraints.lbu   = -umax * ones(nu,1);
ocp.constraints.ubu   =  umax * ones(nu,1);

ocp.constraints.idxbx  = [0, 1, 2, 3, 4];
ocp.constraints.lbx    = [-0.025; -0.025; 0.015; -0.35; -0.35];
ocp.constraints.ubx    = [ 0.025;  0.025; 0.055;  0.35;  0.35];
ocp.constraints.idxsbx = 0:4;

n_sbx = 5;
ocp.cost.Zl = 1e3 * ones(n_sbx, 1);
ocp.cost.Zu = 1e3 * ones(n_sbx, 1);
ocp.cost.zl = 1e3 * ones(n_sbx, 1);
ocp.cost.zu = 1e3 * ones(n_sbx, 1);

ocp.constraints.x0 = xEq;

if ~exist('ocp_solver', 'var') || ~isvalid(ocp_solver)
    fprintf('\n--- Building acados OCP solver (reduced, nx=%d) ---\n', nx);
    ocp_solver = AcadosOcpSolver(ocp);
else
    fprintf('\n--- Reusing OCP solver ---\n');
end

%% --- PLANT SIM SOLVER (mass as runtime parameter) ---
if ~exist('sim_solver', 'var') || ~isvalid(sim_solver)
    fprintf('\n--- Building acados plant sim solver (parametric mass) ---\n');
    sim = AcadosSim();
    sim.model.name        = 'maglev_plant_mismatch';
    sim.model.x           = x_r;
    sim.model.u           = u_sym;
    sim.model.xdot        = xdot_r;
    sim.model.p           = m_sym;
    sim.model.f_impl_expr = xdot_r - f_expl_r_plant;
    sim.parameter_values  = nominal_mass;
    sim.solver_options.Tsim            = Ts;
    sim.solver_options.integrator_type = 'IRK';
    sim.solver_options.num_stages      = 1;
    sim.solver_options.num_steps       = 1;
    sim_solver = AcadosSimSolver(sim);
else
    fprintf('\n--- Reusing plant sim solver ---\n');
end

%% Iterative Mismatch Test Setup
x0 = xEq;
x0(1) = 0.0001;
x0(2) = 0.0001;
x0(3) = zEq_cas + 0.0005;

time_steps = 0:Ts:simulationTime;
mass_step  = 0.001;

fprintf('\nStarting Mass Mismatch search for NMPC controller (acados plant)...\n');
fprintf('Nominal Magnet Mass: %.3f kg\n\n', nominal_mass);

%% Run Baseline (1.0x Mass)
fprintf('Testing Nominal Mass (1.0x)...\n');
[success_nom, x_hist_nom, u_hist_nom] = runMismatchSimNMPC(1.0, nominal_mass, ocp_solver, sim_solver, xEq, uEq, N, time_steps, umax, x0, upper_bounds, lower_bounds);

if ~success_nom
    error('NMPC failed to stabilize even the nominal baseline mass. Check bounds or equilibrium.');
end

%% Search: Increasing Mass (Heavier)
fprintf('\n--- Testing INCREASED Mass ---\n');
multiplier_up = 1.0 + mass_step;
max_successful_multiplier = 1.0;
x_hist_max = x_hist_nom;
u_hist_max = u_hist_nom;

while true
    fprintf('Testing multiplier: %4.3fx... ', multiplier_up);
    [success, x_hist, u_hist] = runMismatchSimNMPC(multiplier_up, nominal_mass, ocp_solver, sim_solver, xEq, uEq, N, time_steps, umax, x0, upper_bounds, lower_bounds);

    if success
        fprintf('Passed.\n');
        max_successful_multiplier = multiplier_up;
        x_hist_max = x_hist;
        u_hist_max = u_hist;
        multiplier_up = multiplier_up + mass_step;

        if multiplier_up > 3.0
            fprintf('Reached arbitrary test limit (3.0x). Breaking.\n');
            break;
        end
    else
        fprintf('FAILED!\n');
        break;
    end
end

%% Search: Decreasing Mass (Lighter)
fprintf('\n--- Testing DECREASED Mass ---\n');
multiplier_down = 1.0 - mass_step;
min_successful_multiplier = 1.0;
x_hist_min = x_hist_nom;
u_hist_min = u_hist_nom;

while true
    if multiplier_down <= 0
        fprintf('Cannot test zero or negative mass. Breaking.\n');
        break;
    end

    fprintf('Testing multiplier: %4.3fx... ', multiplier_down);
    [success, x_hist, u_hist] = runMismatchSimNMPC(multiplier_down, nominal_mass, ocp_solver, sim_solver, xEq, uEq, N, time_steps, umax, x0, upper_bounds, lower_bounds);

    if success
        fprintf('Passed.\n');
        min_successful_multiplier = multiplier_down;
        x_hist_min = x_hist;
        u_hist_min = u_hist;
        multiplier_down = multiplier_down - mass_step;
    else
        fprintf('FAILED!\n');
        break;
    end
end

%% Report Results
fprintf('\n=======================================================\n');
fprintf('            MASS MISMATCH RESULTS (NMPC)               \n');
fprintf('=======================================================\n');
fprintf('Nominal Mass:      %.3f kg\n', nominal_mass);
fprintf('Max Tolerated:     %.3fx (%.3f kg)\n', max_successful_multiplier, nominal_mass * max_successful_multiplier);
fprintf('Min Tolerated:     %.3fx (%.3f kg)\n', min_successful_multiplier, nominal_mass * min_successful_multiplier);
fprintf('Total Range:       [%.3fx, %.3fx]\n', min_successful_multiplier, max_successful_multiplier);
fprintf('=======================================================\n');

%% --- CSV EXPORT FOR EXPERIMENT 4 (NMPC vs RL comparison) ---
% Supervisor feedback: plot center-of-mass position + wobbling-oriented states.
% State layout (10): [x, y, z, alpha, beta, vx, vy, vz, wx, wy]

csv_dir = fullfile(project_root, 'experiment4_data');
if ~exist(csv_dir, 'dir'); mkdir(csv_dir); end

cases = { ...
    'min',     min_successful_multiplier, x_hist_min, u_hist_min; ...
    'nominal', 1.0,                       x_hist_nom, u_hist_nom; ...
    'max',     max_successful_multiplier, x_hist_max, u_hist_max  ...
};

% Long-format trajectory CSV (one row per timestep per case)
all_rows = [];
for c = 1:size(cases, 1)
    case_name  = cases{c, 1};
    mult       = cases{c, 2};
    x_hist     = cases{c, 3};
    u_hist     = cases{c, 4};
    n_steps    = length(time_steps);

    % Pad u_hist with NaN at final step so column lengths match
    u_pad = [u_hist, nan(nu, 1)];

    % Center of mass position
    pos_x = x_hist(1, :).';
    pos_y = x_hist(2, :).';
    pos_z = x_hist(3, :).';

    % Tilt angles (wobbling orientation)
    alpha = x_hist(4, :).';
    beta  = x_hist(5, :).';

    % Linear velocities
    vx = x_hist(6, :).';
    vy = x_hist(7, :).';
    vz = x_hist(8, :).';

    % Angular rates (wobbling rate)
    wx = x_hist(9,  :).';
    wy = x_hist(10, :).';

    % Derived wobble metrics
    tilt_mag      = sqrt(alpha.^2 + beta.^2);            % rad
    ang_rate_mag  = sqrt(wx.^2 + wy.^2);                 % rad/s
    horiz_offset  = sqrt(pos_x.^2 + pos_y.^2);           % m  (xy drift of CoM)
    z_error       = pos_z - xEq(3);                      % m  (height deviation)

    block = table( ...
        time_steps(:), ...
        repmat({case_name},  n_steps, 1), ...
        repmat(mult,         n_steps, 1), ...
        repmat(nominal_mass*mult, n_steps, 1), ...
        pos_x, pos_y, pos_z, z_error, horiz_offset, ...
        alpha, beta, tilt_mag, ...
        vx, vy, vz, ...
        wx, wy, ang_rate_mag, ...
        u_pad(1,:).', u_pad(2,:).', u_pad(3,:).', u_pad(4,:).', ...
        'VariableNames', { ...
            'time_s','case','mass_multiplier','mass_kg', ...
            'x_m','y_m','z_m','z_error_m','horizontal_offset_m', ...
            'alpha_rad','beta_rad','tilt_magnitude_rad', ...
            'vx_mps','vy_mps','vz_mps', ...
            'wx_radps','wy_radps','angular_rate_magnitude_radps', ...
            'u1_A','u2_A','u3_A','u4_A'});

    all_rows = [all_rows; block]; %#ok<AGROW>
end

traj_csv = fullfile(csv_dir, 'nmpc_mass_mismatch_trajectories.csv');
writetable(all_rows, traj_csv);
fprintf('\nWrote trajectory CSV: %s\n', traj_csv);

% Per-run summary CSV (one row per case): tolerance limits + wobble stats
summary_rows = [];
for c = 1:size(cases, 1)
    case_name = cases{c, 1};
    mult      = cases{c, 2};
    x_hist    = cases{c, 3};

    alpha = x_hist(4, :);  beta = x_hist(5, :);
    wx    = x_hist(9, :);  wy   = x_hist(10, :);
    tilt_mag     = sqrt(alpha.^2 + beta.^2);
    ang_rate_mag = sqrt(wx.^2 + wy.^2);
    horiz_offset = sqrt(x_hist(1,:).^2 + x_hist(2,:).^2);
    z_err        = x_hist(3, :) - xEq(3);

    summary_rows = [summary_rows; table( ...
        {case_name}, mult, nominal_mass*mult, ...
        max(abs(z_err)),        rms(z_err), ...
        max(horiz_offset),      rms(horiz_offset), ...
        max(tilt_mag),          rms(tilt_mag), ...
        max(ang_rate_mag),      rms(ang_rate_mag), ...
        'VariableNames', { ...
            'case','mass_multiplier','mass_kg', ...
            'z_error_max_m','z_error_rms_m', ...
            'horiz_offset_max_m','horiz_offset_rms_m', ...
            'tilt_max_rad','tilt_rms_rad', ...
            'ang_rate_max_radps','ang_rate_rms_radps'})]; %#ok<AGROW>
end

summary_csv = fullfile(csv_dir, 'nmpc_mass_mismatch_summary.csv');
writetable(summary_rows, summary_csv);
fprintf('Wrote summary CSV:    %s\n', summary_csv);

% Tolerance-range CSV (single line — matches the RL counterpart for direct comparison)
limits_csv = fullfile(csv_dir, 'nmpc_mass_mismatch_limits.csv');
writetable(table( ...
    {'NMPC'}, nominal_mass, ...
    min_successful_multiplier, max_successful_multiplier, ...
    nominal_mass*min_successful_multiplier, nominal_mass*max_successful_multiplier, ...
    'VariableNames', {'controller','nominal_mass_kg', ...
                      'min_multiplier','max_multiplier', ...
                      'min_mass_kg','max_mass_kg'}), limits_csv);
fprintf('Wrote limits CSV:     %s\n', limits_csv);

%% Plotting the Limits (Nominal vs Max vs Min)
figure('Name', 'Model Mismatch: Mass Variation (NMPC, acados plant)', 'Position', [100, 100, 850, 600]);
tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

% 1. Compare Z-Positions
nexttile;
plot(time_steps, x_hist_min(3, :) * 1000, 'Color', [0.2 0.8 0.2], 'LineWidth', 1.5); hold on;
plot(time_steps, x_hist_nom(3, :) * 1000, 'k', 'LineWidth', 1.5);
plot(time_steps, x_hist_max(3, :) * 1000, 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5);
plot(time_steps, ones(size(time_steps)) * xEq(3) * 1000, 'b--', 'LineWidth', 1.0);
hold off;
title('Z-Position Tracking Under Mass Mismatch (NMPC)');
ylabel('Z Position (mm)');
legend(sprintf('Lighter (%.3fx)', min_successful_multiplier), ...
       'Nominal (1.0x)', ...
       sprintf('Heavier (%.3fx)', max_successful_multiplier), ...
       'Controller Target', 'Location', 'eastoutside');
grid on;

% 2. Compare Control Effort (u1)
nexttile;
plot(time_steps(1:end-1), u_hist_min(1, :), 'Color', [0.2 0.8 0.2], 'LineWidth', 1.5); hold on;
plot(time_steps(1:end-1), u_hist_nom(1, :), 'k', 'LineWidth', 1.5);
plot(time_steps(1:end-1), u_hist_max(1, :), 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5);
hold off;
title('Control Effort (u_1) Reacting to Mismatch');
ylabel('Current (A)');
xlabel('Time (s)');
legend(sprintf('Lighter (%.3fx)', min_successful_multiplier), ...
       'Nominal (1.0x)', ...
       sprintf('Heavier (%.3fx)', max_successful_multiplier), ...
       'Location', 'eastoutside');
grid on;

%% --- LOCAL SIMULATION FUNCTION ---
function [success, x_history, u_history] = runMismatchSimNMPC(mass_multiplier, nominal_mass, ocp_solver, sim_solver, xEq, uEq, N, time_steps, umax, x0, upper_bounds, lower_bounds)

    nx = numel(xEq);
    nu = numel(uEq);

    % Mismatched plant mass (controller is blind to this).
    m_plant = nominal_mass * mass_multiplier;
    sim_solver.set('p', m_plant);

    x_history = zeros(nx, length(time_steps));
    x_history(:, 1) = x0;
    u_history = zeros(nu, length(time_steps)-1);

    x = x0;
    success = true;

    % Warm-start trajectory
    for k = 0:N
        a = k / N;
        ocp_solver.set('x', (1 - a) * x + a * xEq, k);
    end
    for k = 0:N-1
        ocp_solver.set('u', uEq, k);
    end

    for i = 1:length(time_steps)-1

        % 1. Solve OCP (controller uses nominal model)
        try
            ocp_solver.set('constr_x0', x);
            ocp_solver.solve();
            status = ocp_solver.get('status');
            if status ~= 0 && status ~= 2
                success = false;
                break;
            end
            u = ocp_solver.get('u', 0);
        catch
            success = false;
            break;
        end

        u = max(-umax, min(umax, u));
        u_history(:, i) = u;

        % 2. Propagate plant via acados sim (mismatched mass)
        try
            sim_solver.set('x', x);
            sim_solver.set('u', u);
            sim_solver.solve();
            x = sim_solver.get('xn');
        catch
            success = false;
            break;
        end

        x_history(:, i+1) = x;

        % 3. Check bounds
        if any(x > upper_bounds) || any(x < lower_bounds) || any(isnan(x)) || any(isinf(x))
            success = false;
            break;
        end

        % 4. Warm-start shift
        try
            x_traj = ocp_solver.get('x');
            u_traj = ocp_solver.get('u');
            x_traj_shift = [x_traj(:, 2:end), xEq];
            u_traj_shift = [u_traj(:, 2:end), uEq];
            for k = 0:N
                ocp_solver.set('x', x_traj_shift(:, k+1), k);
            end
            for k = 0:N-1
                ocp_solver.set('u', u_traj_shift(:, k+1), k);
            end
        catch
            % non-fatal
        end
    end
end
