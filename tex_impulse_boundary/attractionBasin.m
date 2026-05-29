%% Clear Workspace and Add Paths
clearvars -except ocp_solver_basin sim_solver_basin; clc; close all;

scriptPath   = fileparts(mfilename('fullpath'));
projectRoot  = scriptPath;
acados_root  = 'C:\Users\mariu\acados';

cd(projectRoot);

setenv('ACADOS_SOURCE_DIR',        acados_root);
setenv('ENV_ACADOS_INSTALL_DIR',   acados_root);
setenv('ACADOS_INSTALL_DIR',       acados_root);

addpath(fullfile(acados_root, 'interfaces', 'acados_matlab_octave'));
addpath(fullfile(acados_root, 'external',   'jsonlab'));
addpath(fullfile(acados_root, 'external',   'casadi-matlab'));

addpath(genpath(fullfile(projectRoot, 'model_implementations')));
addpath(genpath(fullfile(projectRoot, 'system_parameters')));
addpath(genpath(fullfile(projectRoot, 'utilities')));

import casadi.*

%% Simulation parameters
controllerType = 'NMPC';
simulationTime = 2.0;   % seconds (success defined as levitating for this duration)
umax = 1;

%% --- MODEL SETUP (reduced order: 10 states) ---
nx_full = 12;
nx      = 10;   % Reduced: removed gamma (yaw) and wz (yaw rate)
nu      = 4;

% Full-order symbolic variables (used to build dynamics and find equilibrium)
x_full = SX.sym('x_full', nx_full);
u_sym  = SX.sym('u', nu);

parameters_maggy_V4;
correctionFactorFast   = computeSolenoidRadiusCorrectionFactor(params, 'fast');
paramsFast             = params;
paramsFast.solenoids.r = correctionFactorFast * paramsFast.solenoids.r;

%% --- BUILD FULL-ORDER CASADI FUNCTION (for equilibrium search) ---
f_expl_full = maglevSystemDynamicsCasADi(x_full, u_sym, paramsFast);
f_func_full = casadi.Function('f_full', {x_full, u_sym}, {f_expl_full});

%% --- FIND EQUILIBRIUM (using full model with gamma=0, wz=0) ---
fprintf('--- Searching for equilibrium z with u = 0 ---\n');

z_var    = SX.sym('z_eq');
u_zero   = zeros(nu, 1);
x_eq_sym = [0; 0; z_var; zeros(9,1)];
accel    = f_func_full(x_eq_sym, u_zero);
fz_residual = accel(9);   % z-acceleration in full model

nlp       = struct('x', z_var, 'f', fz_residual^2);
solver_eq = nlpsol('solver_eq', 'ipopt', nlp, ...
    struct('ipopt', struct('print_level', 3)));

sol     = solver_eq('x0', 0.030, 'lbx', 0.015, 'ubx', 0.060);
zEq_cas = full(sol.x);
uEq     = zeros(nu, 1);

% Reduced-order equilibrium (10 states)
xEq = [0; 0; zEq_cas; zeros(7,1)];

fprintf('Equilibrium: z = %.6f m, u = [0, 0, 0, 0]\n', zEq_cas);

%% --- BUILD REDUCED-ORDER DYNAMICS ---
% Reduced state: x_r = [x, y, z, alpha, beta, vx, vy, vz, wx, wy]
x_r    = SX.sym('x_r', nx);
xdot_r = SX.sym('xdot_r', nx);

% Map reduced state to full state (gamma = 0, wz = 0)
x_full_from_r = [x_r(1:5); 0; x_r(6:8); x_r(9:10); 0];

% Evaluate full dynamics
dx_full = f_func_full(x_full_from_r, u_sym);

% Extract reduced derivatives (drop indices 6 and 12)
f_expl_r = [dx_full(1:5); dx_full(7:11)];

%% --- OCP SETUP (same configuration as workingSimulatorReducedOrder.m) ---
N      = 10;
Tf     = 0.05;
dt_mpc = Tf / N;   % 0.005 s

% Cost matrices (10 states: x, y, z, roll, pitch, vx, vy, vz, wx, wy)
Q = diag([1e2, 1e2, 1e3, ...    % x, y, z position
          1e3, 1e3, ...          % roll, pitch
          1e1*0.3, 1e1*0.3, 1e1*0.3, ...    % vx, vy, vz
          1e1*0.3, 1e1*0.3]);              % wx, wy
R = eye(nu) * 0.1;

ocp = AcadosOcp();
ocp.model.name        = 'maglev_nmpc_reduced_basin';
ocp.model.x           = x_r;
ocp.model.u           = u_sym;
ocp.model.xdot        = xdot_r;
ocp.model.f_impl_expr = xdot_r - f_expl_r;

% Solver options
ocp.solver_options.N_horizon             = N;
ocp.solver_options.tf                    = Tf;
ocp.solver_options.integrator_type       = 'IRK';
ocp.solver_options.sim_method_num_stages = 1;
ocp.solver_options.sim_method_num_steps  = 1;
ocp.solver_options.nlp_solver_type       = 'SQP_RTI';
% ocp.solver_options.nlp_solver_max_iter   = 100;
ocp.solver_options.nlp_solver_tol_stat   = 1e-4;
ocp.solver_options.nlp_solver_tol_eq     = 1e-4;
ocp.solver_options.nlp_solver_tol_ineq   = 1e-4;
ocp.solver_options.nlp_solver_tol_comp   = 1e-4;
ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM';
ocp.solver_options.qp_solver_iter_max    = 200;
ocp.solver_options.qp_solver_warm_start  = 1;
ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON';
ocp.solver_options.regularize_method     = 'CONVEXIFY';

% Cost
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

% Input constraints
ocp.constraints.idxbu = 0:nu-1;
ocp.constraints.lbu   = -1 * ones(nu,1);
ocp.constraints.ubu   =  1 * ones(nu,1);

% State constraints (soft) - indices in 10-state vector:
%   0=x, 1=y, 2=z, 3=alpha, 4=beta
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

%% --- BUILD OCP SOLVER ---
if ~exist('ocp_solver_basin', 'var') || ~isvalid(ocp_solver_basin)
    fprintf('\n--- Building acados OCP solver (reduced, nx=%d) ---\n', nx);
    ocp_solver_basin = AcadosOcpSolver(ocp);
else
    fprintf('\n--- Reusing OCP solver ---\n');
end

%% --- BUILD SIM SOLVER ---
if ~exist('sim_solver_basin', 'var') || ~isvalid(sim_solver_basin)
    fprintf('\n--- Building acados sim solver (reduced, nx=%d) ---\n', nx);
    sim = AcadosSim();
    sim.model.name        = 'maglev_sim_reduced_basin';
    sim.model.x           = x_r;
    sim.model.u           = u_sym;
    sim.model.xdot        = xdot_r;
    sim.model.f_impl_expr = xdot_r - f_expl_r;
    sim.solver_options.Tsim            = dt_mpc;
    sim.solver_options.integrator_type = 'IRK';
    sim.solver_options.num_stages      = 1;
    sim.solver_options.num_steps       = 1;
    sim_solver_basin = AcadosSimSolver(sim);
else
    fprintf('\n--- Reusing sim solver ---\n');
end

%% Define State Constraints for Basin Test
upper_bounds = [
    0.1; 0.1; 0.2;           % Max x, y, z position (m)
    pi/2; pi/2;              % Max alpha, beta (rad)
    5.0; 5.0; 5.0;           % Max linear velocities
    inf; inf;                % Max angular velocities
];
lower_bounds = -upper_bounds;
lower_bounds(3) = 0.005; % Must not drop to the floor

%% Impulse Boundary Search Setup
% The hardware-relevant experiment is now:
%   1. Let the disk levitate at closed-loop equilibrium.
%   2. Hit it with a repeatable off-center mechanical impulse.
%   3. Search for the largest impulse the controller recovers from.
%
% Suggested physical mechanism:
%   * lateral hits: a small pendulum striker released from marked heights,
%     hitting a bumper slightly above/below the disk centre plane;
%   * vertical hits: a guided non-magnetic bead/weight dropped onto a small
%     off-centre cap.
% The impact impulse can be estimated as J = m_striker*(1+e)*sqrt(2*g*h).
% In simulation we parameterise the same impulse by the instantaneous
% velocity kick it gives to the 60 g levitating disk.

magnetMass = paramsFast.magnet.m;
magnetInertia = paramsFast.magnet.I(:);
contactRadius = paramsFast.magnet.r;
contactHeightOffset = paramsFast.magnet.l / 2;  % m, intentionally off centre

numAzimuths = 16;
verticalRatios = [-0.5, 0, 0.5];  % dz/dxy in impulse direction
includePureVertical = true;

initialDeltaV = 0.02;     % m/s
maxDeltaVLimit = 2.00;    % m/s; increase if every direction still recovers
boundaryTolerance = 0.005; % m/s
maxBisectionIterations = 10;

impulseDirections = buildImpulseDirections(numAzimuths, verticalRatios, includePureVertical);
numDirections = size(impulseDirections, 2);

num_steps = round(simulationTime / dt_mpc);
fprintf('Computing impulse robustness boundary from equilibrium.\n');
fprintf('Controller: %s, reduced-order states: %d, dt = %.4f s, horizon N = %d, Tf = %.3f s\n', ...
        controllerType, nx, dt_mpc, N, Tf);
fprintf('Testing %d impulse directions, max velocity kick %.2f m/s, tolerance %.3f m/s.\n', ...
        numDirections, maxDeltaVLimit, boundaryTolerance);
fprintf('Contact model: radius %.1f mm, vertical offset %.1f mm.\n', ...
        contactRadius * 1000, contactHeightOffset * 1000);

tic;
evaluated_points = 0;
failed_solves = 0;
nonzero_status_count = 0;
status_values = zeros(num_steps * numDirections * (ceil(log2(maxDeltaVLimit/initialDeltaV)) + maxBisectionIterations + 2), 1);
status_log_idx = 0;

critical_delta_v = NaN(numDirections, 1);
critical_impulse = NaN(numDirections, 1);
first_failure_delta_v = NaN(numDirections, 1);
boundary_failure_code = zeros(numDirections, 1);
boundary_failure_step = zeros(numDirections, 1);
boundary_failure_state = NaN(nx, numDirections);
boundary_status = strings(numDirections, 1);
direction_trial_count = zeros(numDirections, 1);

for i_dir = 1:numDirections
    impulse_dir = impulseDirections(:, i_dir);
    fprintf('\nDirection %d / %d: [%+.2f, %+.2f, %+.2f]\n', ...
        i_dir, numDirections, impulse_dir(1), impulse_dir(2), impulse_dir(3));

    low = 0;
    high = initialDeltaV;
    low_result = [];
    high_result = [];

    while high <= maxDeltaVLimit
        evaluated_points = evaluated_points + 1;
        direction_trial_count(i_dir) = direction_trial_count(i_dir) + 1;
        trial = runImpulseTrial(high, impulse_dir, xEq, uEq, ocp_solver_basin, ...
            sim_solver_basin, N, num_steps, upper_bounds, lower_bounds, umax, ...
            magnetMass, magnetInertia, contactRadius, contactHeightOffset);
        status_values(status_log_idx + (1:numel(trial.status_values))) = trial.status_values;
        status_log_idx = status_log_idx + numel(trial.status_values);
        failed_solves = failed_solves + trial.exception_count;
        nonzero_status_count = nonzero_status_count + trial.nonzero_status_count;

        fprintf('  bracket dv = %.3f m/s -> %s\n', high, passFailText(trial.success));
        if trial.success
            low = high;
            low_result = trial;
            high = high * 2;
        else
            high_result = trial;
            break;
        end
    end

    if isempty(high_result)
        critical_delta_v(i_dir) = low;
        critical_impulse(i_dir) = magnetMass * low;
        boundary_status(i_dir) = "no failure found";
        if ~isempty(low_result)
            boundary_failure_code(i_dir) = low_result.failure_code;
            boundary_failure_step(i_dir) = low_result.failure_step;
            boundary_failure_state(:, i_dir) = low_result.failure_state;
        end
        fprintf('  No failure up to %.3f m/s. Increase maxDeltaVLimit for this direction.\n', maxDeltaVLimit);
        continue;
    end

    for iter = 1:maxBisectionIterations
        if high - low <= boundaryTolerance
            break;
        end
        mid = 0.5 * (low + high);
        evaluated_points = evaluated_points + 1;
        direction_trial_count(i_dir) = direction_trial_count(i_dir) + 1;
        trial = runImpulseTrial(mid, impulse_dir, xEq, uEq, ocp_solver_basin, ...
            sim_solver_basin, N, num_steps, upper_bounds, lower_bounds, umax, ...
            magnetMass, magnetInertia, contactRadius, contactHeightOffset);
        status_values(status_log_idx + (1:numel(trial.status_values))) = trial.status_values;
        status_log_idx = status_log_idx + numel(trial.status_values);
        failed_solves = failed_solves + trial.exception_count;
        nonzero_status_count = nonzero_status_count + trial.nonzero_status_count;

        fprintf('  refine %2d: dv = %.3f m/s -> %s\n', iter, mid, passFailText(trial.success));
        if trial.success
            low = mid;
            low_result = trial;
        else
            high = mid;
            high_result = trial;
        end
    end

    critical_delta_v(i_dir) = low;
    critical_impulse(i_dir) = magnetMass * low;
    first_failure_delta_v(i_dir) = high;
    boundary_failure_code(i_dir) = high_result.failure_code;
    boundary_failure_step(i_dir) = high_result.failure_step;
    boundary_failure_state(:, i_dir) = high_result.failure_state;
    boundary_status(i_dir) = "bounded";
    fprintf('  Boundary: recover %.3f m/s, fail %.3f m/s.\n', low, high);
end

elapsed = toc;
status_values = status_values(1:status_log_idx);
fprintf('Simulation complete in %.1f seconds.\n', elapsed);
fprintf('Evaluated %d impulse trials.\n', evaluated_points);
fprintf('Failed NMPC solves / sim exceptions: %d\n', failed_solves);
fprintf('Nonzero acados statuses classified as QP failures: %d\n', nonzero_status_count);

%% Save Basin Data
save_filename = sprintf('impulse_boundary_data_%s.mat', controllerType);
save(save_filename, 'impulseDirections', 'critical_delta_v', ...
    'critical_impulse', 'first_failure_delta_v', 'boundary_failure_code', ...
    'boundary_failure_step', 'boundary_failure_state', 'boundary_status', ...
    'direction_trial_count', 'controllerType', 'simulationTime', 'xEq', 'uEq', ...
    'N', 'Tf', 'dt_mpc', 'Q', 'R', 'failed_solves', 'nonzero_status_count', ...
    'status_values', 'magnetMass', 'magnetInertia', 'contactRadius', ...
    'contactHeightOffset', 'initialDeltaV', 'maxDeltaVLimit', ...
    'boundaryTolerance', 'maxBisectionIterations');
fprintf('Impulse boundary data successfully saved to %s\n', save_filename);

%% Plotting results
figure('Name', 'Impulse Robustness Boundary', 'Color', 'white', 'Position', [100, 100, 1200, 720]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
bar(critical_delta_v, 'FaceColor', [0.20 0.45 0.75]);
hold on;
plot(first_failure_delta_v, 'rx', 'LineWidth', 1.2, 'MarkerSize', 7);
yline(maxDeltaVLimit, 'k--', 'LineWidth', 1);
hold off;
grid on;
xlabel('Impulse direction index');
ylabel('Velocity kick magnitude [m/s]');
title('Largest Recoverable Impulse Velocity');
legend('recoverable lower bound', 'first failing upper bound', 'search limit', 'Location', 'best');

nexttile;
scatter3(impulseDirections(1,:), impulseDirections(2,:), impulseDirections(3,:), ...
    60, critical_delta_v, 'filled');
grid on;
axis equal;
xlabel('direction x');
ylabel('direction y');
zlabel('direction z');
title('Recoverable Boundary by Direction');
cb = colorbar;
cb.Label.String = 'critical velocity kick [m/s]';

sgtitle(sprintf('Impulse Robustness Boundary (%s Controller)', controllerType), ...
    'FontSize', 14, 'FontWeight', 'bold');

function directions = buildImpulseDirections(num_azimuths, vertical_ratios, include_pure_vertical)
    azimuths = linspace(0, 2*pi, num_azimuths + 1);
    azimuths = azimuths(1:end-1);
    num_vertical = 2 * double(include_pure_vertical);
    directions = zeros(3, num_azimuths * numel(vertical_ratios) + num_vertical);
    idx = 0;
    for z_ratio = vertical_ratios
        for az = azimuths
            dir = [cos(az); sin(az); z_ratio];
            idx = idx + 1;
            directions(:, idx) = dir / norm(dir);
        end
    end
    if include_pure_vertical
        directions(:, idx+1) = [0; 0; 1];
        directions(:, idx+2) = [0; 0; -1];
    end
end

function trial = runImpulseTrial(delta_v_mag, impulse_dir, xEq, uEq, ocp_solver, ...
        sim_solver, N, num_steps, upper_bounds, lower_bounds, umax, mass, ...
        inertia_vec, contact_radius, contact_height_offset)
    x_current = applyContactImpulse(xEq, delta_v_mag, impulse_dir, mass, ...
        inertia_vec, contact_radius, contact_height_offset);
    resetNmpcSolver(ocp_solver);
    initializeNmpcTrajectory(ocp_solver, x_current, xEq, uEq, N);

    trial.success = true;
    trial.failure_code = 0; % 0=success, 1=bounds/nonfinite, 2=exception, 3=QP failure
    trial.failure_step = 0;
    trial.failure_state = NaN(length(xEq), 1);
    trial.status_values = zeros(num_steps, 1);
    trial.nonzero_status_count = 0;
    trial.exception_count = 0;
    status_log_idx = 0;

    for step = 1:num_steps
        try
            ocp_solver.set('constr_x0', x_current);
            ocp_solver.solve();

            status = ocp_solver.get('status');
            status_log_idx = status_log_idx + 1;
            trial.status_values(status_log_idx) = status;
            if status ~= 0
                trial.success = false;
                trial.failure_code = 3;
                trial.failure_step = step;
                trial.failure_state = x_current;
                trial.nonzero_status_count = trial.nonzero_status_count + 1;
                break;
            end

            u_applied = ocp_solver.get('u', 0);
            u_applied = max(min(u_applied, umax), -umax);

            sim_solver.set('x', x_current);
            sim_solver.set('u', u_applied);
            sim_solver.solve();
            x_next = sim_solver.get('xn');

            shiftNmpcTrajectory(ocp_solver, xEq, uEq, N);
        catch ME
            trial.success = false;
            trial.failure_code = 2;
            trial.failure_step = step;
            trial.failure_state = x_current;
            trial.exception_count = 1;
            fprintf('  Solver/simulation failed at step %d: %s\n', step, ME.message);
            break;
        end

        if any(x_next > upper_bounds) || any(x_next < lower_bounds) || ...
                any(isnan(x_next)) || any(isinf(x_next))
            trial.success = false;
            trial.failure_code = 1;
            trial.failure_step = step;
            trial.failure_state = x_next;
            break;
        end

        x_current = x_next;
    end

    trial.status_values = trial.status_values(1:status_log_idx);
end

function x_impulsed = applyContactImpulse(xEq, delta_v_mag, impulse_dir, mass, ...
        inertia_vec, contact_radius, contact_height_offset)
    x_impulsed = xEq;
    delta_v = delta_v_mag * impulse_dir;
    impulse = mass * delta_v;

    lateral = impulse_dir(1:2);
    if norm(lateral) > 1e-9
        lateral_unit = lateral / norm(lateral);
    else
        lateral_unit = [1; 0];
    end

    if abs(impulse_dir(3)) > 0.95
        contact_point = [contact_radius * lateral_unit; 0];
    else
        contact_point = [-contact_radius * lateral_unit; contact_height_offset];
    end

    angular_impulse = cross(contact_point, impulse);
    delta_omega = angular_impulse ./ inertia_vec;

    x_impulsed(6:8) = x_impulsed(6:8) + delta_v;
    x_impulsed(9:10) = x_impulsed(9:10) + delta_omega(1:2);
end

function text = passFailText(success)
    if success
        text = 'recover';
    else
        text = 'fail';
    end
end

function resetNmpcSolver(ocp_solver)
    try
        ocp_solver.reset();
    catch
        try
            ocp_solver.set('reset', 1);
        catch
            % Some acados MATLAB interfaces do not expose reset. In that case,
            % the trajectory initialization below still overwrites x and u.
        end
    end
end

function initializeNmpcTrajectory(ocp_solver, x_current, xEq, uEq, N)
    % Linear interpolation from the tested initial condition to equilibrium.
    for k = 0:N
        alpha_k = k / N;
        ocp_solver.set('x', (1 - alpha_k) * x_current + alpha_k * xEq, k);
    end
    for k = 0:N-1
        ocp_solver.set('u', uEq, k);
    end
end

function shiftNmpcTrajectory(ocp_solver, xEq, uEq, N)
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
end
