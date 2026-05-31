%% noiseRSM_reducedOrder.m
% =========================================================================
%  Response Surface Methodology (RSM) study of measurement-noise tolerance
%  for the reduced-order (10-state) NMPC controller.
%
%  Following supervisor's recommendation: instead of changing one noise
%  factor at a time (OVAT), we use a Box-Behnken design over four noise
%  factors and fit a second-order response surface.
%
%  Factors (independent variables, in log10 space):
%    f1 = log10(sigma_pos)   -- position noise std [m]
%    f2 = log10(sigma_ori)   -- orientation noise std [rad]
%    f3 = log10(sigma_vel)   -- linear-velocity noise std [m/s]
%    f4 = log10(sigma_ang)   -- angular-velocity noise std [rad/s]
%
%  Coded levels (-1, 0, +1) map to (center - delta, center, center + delta)
%  in log10 space. Center matches the supervisor-question values:
%      sigma_pos = 1e-4,  sigma_ori = 1e-3,
%      sigma_vel = 1e-3,  sigma_ang = 1e-2
%
%  Response variables (per simulation):
%    y_rms      : RMS deviation from equilibrium over the settling window
%    y_maxpos   : peak |position deviation| over the run
%    y_diverged : 1 if simulation diverged, 0 otherwise
%
%  Author: Marius Jullum Faanes (RSM extension)
% =========================================================================

%% --- PROJECT SETUP ---
% Force a clean rebuild of the acados solvers. Persistent solver objects
% from previous (possibly crashed) MATLAB sessions can carry NaN internal
% state that is impossible to recover from.
clear ocp_solver sim_solver;
clearvars; clc;

acados_root  = 'C:\Users\mariu\acados';
project_root = 'C:\Users\mariu\noise_testing';

setenv('ACADOS_SOURCE_DIR',        acados_root);
setenv('ENV_ACADOS_INSTALL_DIR',   acados_root);
setenv('ACADOS_INSTALL_DIR',       acados_root);

addpath(fullfile(acados_root, 'interfaces', 'acados_matlab_octave'));
addpath(fullfile(acados_root, 'external',   'jsonlab'));
addpath(fullfile(acados_root, 'external',   'casadi-matlab'));

addpath(genpath(fullfile(project_root, 'model_implementations')));
addpath(genpath(fullfile(project_root, 'system_parameters')));
addpath(genpath(fullfile(project_root, 'utilities')));

import casadi.*

%% --- CONTROLLER SELECTION ---
% Pick which controller to test: 'nmpc', 'rl100', or 'rl1000'.
controller_choices = {'nmpc', 'rl100', 'rl1000'};
controller_type = controller_choices{1};  % next test: NMPC

% Pick plant integration backend: 'acados', 'ode15s', or 'ode15s_full'
% ('ode15' is accepted as an alias for ode15s).
% The RL agents were developed with MATLAB ODE integration, so 'ode15s' is
% useful for checking whether integrator mismatch explains a failure.
plant_integrator_choices = {'acados', 'ode15s', 'ode15s_full'};
plant_integrator = plant_integrator_choices{1};  % use acados for NMPC

% Pick the plant/model parameter set. NMPC results in this script originally
% used V4 with the fast-model solenoid radius correction. RL training may
% have used a nominal ODE model, so this is a useful compatibility check.
plant_parameter_choices = {'v4_corrected', 'v4_nominal', 'v2_corrected', 'v2_nominal'};
plant_parameter_set = plant_parameter_choices{1};

% RL agents can live either in project_root/agents or directly in project_root.
switch lower(controller_type)
    case 'nmpc'
        agent_paths = {};
    case 'rl100'
        agent_paths = { ...
            fullfile(project_root, 'agents', 'L2N128_1.mat'), ...
            fullfile(project_root, 'L2N128_1.mat')};
    case 'rl1000'
        agent_paths = { ...
            fullfile(project_root, 'agents', 'HF_1000Hz_1.mat'), ...
            fullfile(project_root, 'HF_1000Hz_1.mat')};
    otherwise
        error('Unsupported controller_type "%s". Use "nmpc", "rl100", or "rl1000".', controller_type);
end

% Use 'state' when the agent was trained on absolute reduced state directly.
% Use 'error' when the agent was trained around zero as x - xEq.
agent_observation = 'state';

% RL convention switches. The developer note used the saved agent directly;
% 'greedy' is useful for deterministic PPO evaluation, but 'agent' matches
% the original call more closely.
rl_policy_mode = 'agent';      % 'agent' or 'greedy'
rl_action_sign = 1;            % set to -1 if the training plant used opposite current sign
rl_action_order = 1:4;         % reorder currents if the training plant used another coil order

use_nmpc = strcmpi(controller_type, 'nmpc');
use_rl   = any(strcmpi(controller_type, {'rl100','rl1000'}));

agent = [];
controller_name = 'nmpc';
agent_path = '';
if use_rl
    existing_agent_paths = agent_paths(cellfun(@(p) exist(p, 'file') == 2, agent_paths));
    if isempty(existing_agent_paths)
        error('controller_type is "%s", but none of the configured agent_paths exist.', controller_type);
    end
    agent_path = existing_agent_paths{1};
    loaded_data = load(agent_path, 'agent');
    if ~isfield(loaded_data, 'agent')
        error('Agent file "%s" does not contain a variable named "agent".', agent_path);
    end
    switch lower(rl_policy_mode)
        case 'agent'
            agent = loaded_data.agent;
        case 'greedy'
            agent = getGreedyPolicy(loaded_data.agent);
        otherwise
            error('Unsupported rl_policy_mode "%s". Use "agent" or "greedy".', rl_policy_mode);
    end
    [~, controller_name, ~] = fileparts(agent_path);
    fprintf('\n--- Loaded RL agent: %s ---\n', agent_path);
    fprintf('--- RL policy mode: %s ---\n', rl_policy_mode);
else
    fprintf('\n--- Using NMPC controller ---\n');
end
output_tag = regexprep(controller_name, '[^\w-]', '_');
if use_nmpc
    output_suffix = '';
else
    output_suffix = ['_' output_tag];
end

%% --- MODEL SETUP (reduced order: 10 states) ---
nx_full = 12;
nx      = 10;
nu      = 4;

x_full = SX.sym('x_full', nx_full);
u_sym  = SX.sym('u', nu);

switch lower(plant_parameter_set)
    case {'v4_corrected','v4_nominal'}
        parameters_maggy_V4;
    case {'v2_corrected','v2_nominal'}
        parameters_maggy_V2;
    otherwise
        error('Unsupported plant_parameter_set "%s".', plant_parameter_set);
end
paramsFast = params;
if contains(lower(plant_parameter_set), 'corrected')
    correctionFactorFast   = computeSolenoidRadiusCorrectionFactor(params, 'fast');
    paramsFast.solenoids.r = correctionFactorFast * paramsFast.solenoids.r;
end

f_expl_full = maglevSystemDynamicsCasADi(x_full, u_sym, paramsFast);
f_func_full = casadi.Function('f_full', {x_full, u_sym}, {f_expl_full});

%% --- EQUILIBRIUM ---
fprintf('--- Searching for equilibrium z with u = 0 ---\n');
z_var       = SX.sym('z_eq');
u_zero      = zeros(nu, 1);
x_eq_sym    = [0; 0; z_var; zeros(9,1)];
accel       = f_func_full(x_eq_sym, u_zero);
fz_residual = accel(9);
nlp         = struct('x', z_var, 'f', fz_residual^2);
solver_eq   = nlpsol('solver_eq', 'ipopt', nlp, ...
    struct('ipopt', struct('print_level', 0)));
sol     = solver_eq('x0', 0.030, 'lbx', 0.015, 'ubx', 0.060);
zEq_cas = full(sol.x);
uEq     = zeros(nu, 1);
xEq     = [0; 0; zEq_cas; zeros(7,1)];
fprintf('Equilibrium: z = %.6f m\n', zEq_cas);

%% --- REDUCED DYNAMICS ---
x_r           = SX.sym('x_r', nx);
xdot_r        = SX.sym('xdot_r', nx);
x_full_from_r = [x_r(1:5); 0; x_r(6:8); x_r(9:10); 0];
dx_full       = f_func_full(x_full_from_r, u_sym);
f_expl_r      = [dx_full(1:5); dx_full(7:11)];
f_func_r      = casadi.Function('f_reduced', {x_r, u_sym}, {f_expl_r});

%% --- OCP SETUP (same as workingSimulatorReducedOrder.m) ---
N      = 10;
Tf     = 0.05;
dt_mpc = Tf / N;
switch lower(controller_type)
    case 'rl100'
        dt_mpc = 1/100;
    case 'rl1000'
        dt_mpc = 1/1000;
end

Q = diag([1e2, 1e2, 1e3, ...
          1e3, 1e3, ...
          1e1*0.3, 1e1*0.3, 1e1*0.3, ...
          1e1*0.3, 1e1*0.3]);
R = eye(nu) * 0.1;

ocp = AcadosOcp();
ocp.model.name        = 'maglev_nmpc_reduced';
ocp.model.x           = x_r;
ocp.model.u           = u_sym;
ocp.model.xdot        = xdot_r;
ocp.model.f_impl_expr = xdot_r - f_expl_r;

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
ocp.constraints.lbu   = -1 * ones(nu,1);
ocp.constraints.ubu   =  1 * ones(nu,1);

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

if use_nmpc
    if ~exist('ocp_solver', 'var') || ~isvalid(ocp_solver)
        fprintf('\n--- Building acados OCP solver (reduced, nx=%d) ---\n', nx);
        ocp_solver = AcadosOcpSolver(ocp);
    else
        fprintf('\n--- Reusing OCP solver ---\n');
    end
else
    fprintf('\n--- Skipping OCP solver build for RL controller ---\n');
    ocp_solver = [];
end

use_acados_sim = strcmpi(plant_integrator, 'acados');
use_ode15s_sim = any(strcmpi(plant_integrator, {'ode15s','ode15'}));
use_ode15s_full_sim = strcmpi(plant_integrator, 'ode15s_full');
if ~(use_acados_sim || use_ode15s_sim || use_ode15s_full_sim)
    error('Unsupported plant_integrator "%s". Use "acados", "ode15s", or "ode15s_full".', plant_integrator);
end

if use_acados_sim
    if ~exist('sim_solver', 'var') || ~isvalid(sim_solver)
        fprintf('\n--- Building acados sim solver (reduced, nx=%d) ---\n', nx);
        sim = AcadosSim();
        sim.model.name        = 'maglev_sim_reduced';
        sim.model.x           = x_r;
        sim.model.u           = u_sym;
        sim.model.xdot        = xdot_r;
        sim.model.f_impl_expr = xdot_r - f_expl_r;
        sim.solver_options.Tsim            = dt_mpc;
        sim.solver_options.integrator_type = 'IRK';
        sim.solver_options.num_stages      = 1;
        sim.solver_options.num_steps       = 1;
        sim_solver = AcadosSimSolver(sim);
    else
        fprintf('\n--- Reusing sim solver ---\n');
    end
else
    fprintf('\n--- Using %s plant integration ---\n', plant_integrator);
    sim_solver = [];
end

%% =========================================================================
%  RSM EXPERIMENTAL DESIGN
%  =========================================================================
%  Factor center (matches supervisor-question baseline) and spread in log10:
%      step_per_unit = how many decades one coded unit represents
%      A coded level of +/-1 thus scales the noise by 10^(+/-step_per_unit).
% =========================================================================

% --- Factor metadata ---
% Two previous runs bracketed the controller's noise tolerance:
%   center = supervisor's baseline (1e-4,1e-3,1e-3,1e-2) -> all 9 diverged
%   center = supervisor/10        (1e-5,1e-4,1e-4,1e-3) -> all converged
% The stability threshold therefore lies inside that decade. Place the
% RSM center HALFWAY between the two on the log10 scale and use a wide
% step so the design straddles the boundary: -1 corner sits in the safe
% region (1e-5 etc), +1 corner sits in the unsafe region (supervisor's).
factor_names   = {'sigma_pos','sigma_ori','sigma_vel','sigma_ang'};
log_center     = [-4.5, -3.5, -3.5, -2.5];   % geometric mean of the two extremes
step_per_unit  = 0.5;                          % +-1 unit  =>  factor 10^0.5 ~ 3.16x
% --> Coded -1: (1e-5,1e-4,1e-4,1e-3)  = 1/10 of supervisor   (expected: safe)
% --> Coded  0: (3e-5,3e-4,3e-4,3e-3)  = midpoint              (borderline)
% --> Coded +1: (1e-4,1e-3,1e-3,1e-2)  = supervisor's baseline (expected: unsafe)

% --- Box-Behnken design for k = 4 factors (27 runs incl. 3 centers) ---
%   Standard Box-Behnken: each pair of factors visits {-1,-1},{-1,+1},
%   {+1,-1},{+1,+1} while the other factors stay at 0.
pairs = nchoosek(1:4, 2);
BB    = [];
for p = 1:size(pairs,1)
    row = zeros(4, 4);     % 4 combos per pair
    combos = [-1 -1; -1 +1; +1 -1; +1 +1];
    for c = 1:4
        row(c, pairs(p,1)) = combos(c,1);
        row(c, pairs(p,2)) = combos(c,2);
    end
    BB = [BB; row];        %#ok<AGROW>
end
BB = [BB; zeros(3,4)];     % 3 center-point replicates
n_design = size(BB,1);
fprintf('\n--- Box-Behnken design: %d design points (k=4 factors) ---\n', n_design);

% --- Monte Carlo replicates per design point (different RNG seeds) ---
n_replicates = 3;          % set higher for more statistical power

%% --- SIMULATION CONFIG ---
sim_duration = 7.5;         % seconds; keeps physical run length comparable across controller rates
sim_steps  = round(sim_duration / dt_mpc);
settle_idx = round(0.5 * sim_steps); % use last 50% of run for RMS metric

% Start the test exactly at equilibrium for the NMPC case.
% perturbation_scale = 0.05;
% dx_dir = [0.005; -0.008; 0.010; 0.15; -0.10; ...
%           0.01;  -0.01;  0.0;   0.2;  -0.3];
x0_perturbed = xEq;

% Map factor index -> state indices (reduced model)
%   1 -> sigma_pos applied to states [1 2 3]   (x, y, z)
%   2 -> sigma_ori applied to states [4 5]     (alpha, beta)
%   3 -> sigma_vel applied to states [6 7 8]   (vx, vy, vz)
%   4 -> sigma_ang applied to states [9 10]    (wx, wy)
state_groups = {1:3, 4:5, 6:8, 9:10};

% Optional weighting so the RMS error treats positions and angles fairly
% (1 m comparable to 1 rad). Position units: m, angles: rad, velocities: m/s, rad/s.
err_weight = ones(nx,1);
err_weight(1:3) = 1.0;       % positions in m -- baseline
err_weight(4:5) = 0.05;      % de-weight rad (1 rad = 57 deg)
err_weight(6:8) = 0.1;       % m/s
err_weight(9:10) = 0.05;     % rad/s

% --- TIGHT divergence bounds (applied BEFORE each OCP solve) ---
% The single most important change in this script: we declare the run
% diverged as soon as the *true* plant state leaves a physically sane box,
% and break BEFORE the next OCP solve. Consequence: the QP solver never
% sees pathological states (10^4 rad/s etc.), HPIPM internal state stays
% clean, and we only need reset() between replicates -- no rebuilds.
div_pos_xy = 0.024;     % m   (state constraint is +-25mm, leave 1mm slack)
div_pos_z  = 0.020;     % m   (around zEq)
div_angle  = 0.30;      % rad (~17 deg)
div_lin_v  = 0.50;      % m/s
div_ang_v  = 5.0;       % rad/s

% The tight box above protects the NMPC solver from pathological states.
% RL policies do not feed a QP solver, and these agents were trained with
% wider observation limits, especially angular rates (up to +-50 rad/s).
if use_rl
    div_pos_xy = 0.08;   % m
    div_pos_z  = 0.08;   % m around zEq
    div_angle  = 1.00;   % rad
    div_lin_v  = 5.00;   % m/s
    div_ang_v  = 50.0;   % rad/s
end

%% --- BASELINE: noise-free reference ---
% A noiseless closed-loop run, identical otherwise to the noisy ones.
% Establishes (a) that the controller works in this config, (b) a floor RMS
% to compare noisy runs against. If this diverges we abort the sweep.
fprintf('\n--- Baseline noise-free run ---\n');
rng(0,'twister');
if use_nmpc
    try
        ocp_solver.reset();
    catch
    end
end
x_current = x0_perturbed;
x_full_current = reducedToFullState(x_current);
if use_nmpc
    for k = 0:N
        ak = k/N;
        ocp_solver.set('x', (1-ak)*x_current + ak*xEq, k);
    end
    for k = 0:N-1
        ocp_solver.set('u', uEq, k);
    end
end
plot_x_base = zeros(nx, sim_steps+1);
plot_x_base(:,1) = x_current;
baseline_diverged = false;
baseline_last_step = sim_steps;
baseline_div_reason = '';
for i = 1:sim_steps
    if any(~isfinite(x_current))
        baseline_div_reason = 'nonfinite state';
    elseif max(abs(x_current(1:2))) > div_pos_xy
        baseline_div_reason = 'xy position bound';
    elseif abs(x_current(3) - zEq_cas) > div_pos_z
        baseline_div_reason = 'z position bound';
    elseif max(abs(x_current(4:5))) > div_angle
        baseline_div_reason = 'angle bound';
    elseif max(abs(x_current(6:8))) > div_lin_v
        baseline_div_reason = 'linear velocity bound';
    elseif max(abs(x_current(9:10))) > div_ang_v
        baseline_div_reason = 'angular velocity bound';
    end
    if ~isempty(baseline_div_reason)
        baseline_diverged = true;
        baseline_last_step = i;
        break;
    end

    u_a = getNoiseTestAction(controller_type, ocp_solver, agent, x_current, xEq, ...
        agent_observation, rl_action_sign, rl_action_order);
    [x_current, x_full_current] = stepNoiseTestPlant(plant_integrator, sim_solver, f_func_r, ...
        x_current, x_full_current, u_a, dt_mpc, paramsFast);
    plot_x_base(:, i+1) = x_current;
    % Warm-start shift
    if use_nmpc
        x_traj = ocp_solver.get('x');  u_traj = ocp_solver.get('u');
        for k = 0:N
            if k < N, ocp_solver.set('x', x_traj(:,k+2), k); else, ocp_solver.set('x', xEq, k); end
        end
        for k = 0:N-1
            if k < N-1, ocp_solver.set('u', u_traj(:,k+2), k); else, ocp_solver.set('u', uEq, k); end
        end
    end
end
baseline_n_samples = max(1, baseline_last_step);
baseline_metric_start = min(settle_idx+1, baseline_n_samples);
err_base = (plot_x_base(:, baseline_metric_start:baseline_n_samples) - xEq) .* err_weight;
y_rms_base = sqrt(mean(sum(err_base.^2, 1)));
pos_dev_base = max(sqrt(sum((plot_x_base(1:3,1:baseline_n_samples) - xEq(1:3)).^2, 1)));
fprintf('  baseline (%s):  diverged=%d  step=%d  y_rms=%.4e   max |pos-eq|=%.4e m\n', ...
    controller_name, baseline_diverged, baseline_last_step, y_rms_base, pos_dev_base);
if baseline_diverged
    fprintf('  baseline divergence reason: %s\n', baseline_div_reason);
    fprintf('  state at stop: p=[%+.4e %+.4e %+.4e], ang=[%+.4e %+.4e], v=[%+.4e %+.4e %+.4e], w=[%+.4e %+.4e]\n', ...
        x_current(1), x_current(2), x_current(3), x_current(4), x_current(5), ...
        x_current(6), x_current(7), x_current(8), x_current(9), x_current(10));
end
if baseline_diverged || any(~isfinite(plot_x_base(:, max(1, baseline_last_step)))) || abs(x_current(3)-zEq_cas) > 0.005
    error('Baseline noise-free run failed for controller "%s" (%s); do not proceed with noise sweep.', ...
        controller_name, controller_type);
end

%% =========================================================================
%  OFAT (one-factor-at-a-time) SCREENING SWEEP
%  -------------------------------------------------------------------------
%  Vary ONE noise factor at a time over 2-3 decades (the other three at
%  zero) and record stability + RMS. Directly answers:
%     "Which state group is the controller most sensitive to?"
%  The supervisor's note: OFAT is "not the sharpest knife in DOE" -- meaning
%  it can't see INTERACTIONS between factors -- but it is the standard tool
%  for FACTOR SCREENING and threshold-finding, which the RSM below is not
%  the right tool for (RSM assumes the response surface is smooth, not
%  binary). OFAT + RSM is the textbook combination.
% =========================================================================
fprintf('\n--- OFAT screening sweep ---\n');

ofat_n_levels = 9;
ofat_n_reps   = 3;
% Sigma levels per factor (log10 spaced over ~3 decades, centred near the
% threshold we already located between 10^-5 and 10^-4 for sigma_pos).
ofat_log_min = [-6.0, -5.0, -5.0, -4.0];   % well below the expected threshold
ofat_log_max = [-3.0, -2.0, -2.0, -1.0];   % well above (supervisor's value +1 decade)

ofat = table('Size',[0, 6], ...
    'VariableTypes',{'double','double','double','double','double','double'}, ...
    'VariableNames',{'factor','level_idx','sigma','rep','y_rms','y_diverged'});

t_ofat = tic;
for f = 1:4
    log_levels = linspace(ofat_log_min(f), ofat_log_max(f), ofat_n_levels);
    fprintf('  Factor %d (%s):\n', f, factor_names{f});
    for li = 1:ofat_n_levels
        sigma_lvl = 10^log_levels(li);
        % Only factor f has noise; others = 0
        sigma_vec = zeros(nx,1);
        sigma_vec(state_groups{f}) = sigma_lvl;

        ndiv = 0;
        rms_acc = nan(ofat_n_reps,1);
        for r = 1:ofat_n_reps
            rng(7000 + 100*f + 10*li + r, 'twister');
            if use_nmpc
                try
                    ocp_solver.reset();
                catch
                end
            end
            x_current = x0_perturbed;
            x_full_current = reducedToFullState(x_current);
            if use_nmpc
                for k = 0:N
                    ak = k/N;
                    ocp_solver.set('x', (1-ak)*x_current + ak*xEq, k);
                end
                for k = 0:N-1
                    ocp_solver.set('u', uEq, k);
                end
            end

            plot_x_o = zeros(nx, sim_steps+1);
            plot_x_o(:,1) = x_current;
            div = false;
            for i = 1:sim_steps
                if any(~isfinite(x_current)) || ...
                   max(abs(x_current(1:2)))    > div_pos_xy || ...
                   abs(x_current(3) - zEq_cas) > div_pos_z  || ...
                   max(abs(x_current(4:5)))    > div_angle  || ...
                   max(abs(x_current(6:8)))    > div_lin_v  || ...
                   max(abs(x_current(9:10)))   > div_ang_v
                    div = true; break;
                end
                x_meas = x_current + sigma_vec .* randn(nx,1);
                u_a = getNoiseTestAction(controller_type, ocp_solver, agent, x_meas, xEq, ...
                    agent_observation, rl_action_sign, rl_action_order);
                [x_current, x_full_current] = stepNoiseTestPlant(plant_integrator, sim_solver, f_func_r, ...
                    x_current, x_full_current, u_a, dt_mpc, paramsFast);
                plot_x_o(:, i+1) = x_current;
                if use_nmpc
                    x_traj = ocp_solver.get('x'); u_traj = ocp_solver.get('u');
                    for k = 0:N
                        if k < N, ocp_solver.set('x', x_traj(:,k+2), k); else, ocp_solver.set('x', xEq, k); end
                    end
                    for k = 0:N-1
                        if k < N-1, ocp_solver.set('u', u_traj(:,k+2), k); else, ocp_solver.set('u', uEq, k); end
                    end
                end
            end
            if div
                ndiv = ndiv + 1;
                rms_acc(r) = NaN;
            else
                err = (plot_x_o(:, settle_idx+1:end) - xEq) .* err_weight;
                rms_acc(r) = sqrt(mean(sum(err.^2, 1)));
            end
            ofat = [ofat; {f, li, sigma_lvl, r, rms_acc(r), double(div)}]; %#ok<AGROW>
        end
        fprintf('    sigma=%.2e  div=%d/%d   y_rms=%s\n', ...
            sigma_lvl, ndiv, ofat_n_reps, ...
            sprintf('%.3e ', rms_acc));
    end
end
fprintf('OFAT sweep done in %.1f s.\n', toc(t_ofat));

% --- Per-factor 50%-divergence threshold (linear interp on P(div) vs log10 sigma)
ofat_thresholds = nan(4,1);
fprintf('\n  Per-factor stability thresholds (P(div)=0.5):\n');
for f = 1:4
    rows = ofat.factor == f;
    lvls = unique(ofat.sigma(rows));
    p_div = zeros(numel(lvls),1);
    for li = 1:numel(lvls)
        m = rows & ofat.sigma == lvls(li);
        p_div(li) = mean(ofat.y_diverged(m));
    end
    % find first level where p_div crosses 0.5
    idx = find(p_div >= 0.5, 1, 'first');
    if isempty(idx) || idx == 1
        fprintf('    %-10s : P(div) never reaches 0.5 -- sweep range too low.\n', factor_names{f});
    else
        % linear interp in log10(sigma) space
        x1 = log10(lvls(idx-1)); x2 = log10(lvls(idx));
        y1 = p_div(idx-1);       y2 = p_div(idx);
        log_thr = x1 + (0.5 - y1)/(y2 - y1) * (x2 - x1);
        ofat_thresholds(f) = 10^log_thr;
        fprintf('    %-10s : sigma_threshold = %.3e\n', factor_names{f}, ofat_thresholds(f));
    end
end

% --- OFAT plot: per-factor sigma vs y_rms (left) and P(div) (right)
fig_ofat = figure('Name','OFAT sensitivity per factor','Position',[80 80 1200 800]);
for f = 1:4
    subplot(2,2,f); hold on; grid on;
    rows = ofat.factor == f;
    lvls = unique(ofat.sigma(rows));
    rms_med = nan(numel(lvls),1); rms_p25 = rms_med; rms_p75 = rms_med;
    p_div   = zeros(numel(lvls),1);
    for li = 1:numel(lvls)
        m = rows & ofat.sigma == lvls(li);
        rr = ofat.y_rms(m & ~isnan(ofat.y_rms));
        if ~isempty(rr)
            rms_med(li) = median(rr);
            rms_p25(li) = quantile(rr, 0.25);
            rms_p75(li) = quantile(rr, 0.75);
        end
        p_div(li) = mean(ofat.y_diverged(m));
    end
    yyaxis left;
    set(gca,'XScale','log','YScale','log');
    plot(lvls, rms_med, '-o', 'LineWidth', 1.4); hold on;
    plot(lvls, rms_p25, ':', 'LineWidth', 1.0);
    plot(lvls, rms_p75, ':', 'LineWidth', 1.0);
    ylabel('y\_rms (survivors only)');
    yyaxis right;
    plot(lvls, p_div, '-s', 'LineWidth', 1.4);
    set(gca,'XScale','log');
    ylim([-0.05 1.05]);
    ylabel('P(diverged)');
    if ~isnan(ofat_thresholds(f))
        xline(ofat_thresholds(f), '--k', sprintf('thr=%.2e', ofat_thresholds(f)));
    end
    xlabel(sprintf('%s', factor_names{f}), 'Interpreter','none');
    title(sprintf('OFAT: factor %d (%s)', f, factor_names{f}), 'Interpreter','none');
end
sgtitle('Per-factor noise sensitivity (others at zero noise)');
savefig(fig_ofat, ['ofat_sensitivity' output_suffix '.fig']);

%% --- RUN THE DESIGNED EXPERIMENT ---
n_runs   = n_design * n_replicates;
results  = table('Size',[n_runs, 11], ...
    'VariableTypes', {'double','double','double','double','double','double',...
                      'double','double','double','double','double'}, ...
    'VariableNames', {'design','rep','f1','f2','f3','f4', ...
                      'sigma_pos','sigma_ori','sigma_vel','sigma_ang','seed'});
y_rms      = zeros(n_runs,1);
y_maxpos   = zeros(n_runs,1);
y_diverged = zeros(n_runs,1);

run_idx = 0;
fprintf('\n--- Starting RSM sweep: %d design pts x %d reps = %d runs ---\n', ...
    n_design, n_replicates, n_runs);
fprintf('Controller under test: %s (%s), plant=%s, dt = %.4g s, sim_steps = %d\n', ...
    controller_name, controller_type, plant_integrator, dt_mpc, sim_steps);
fprintf('Plant parameter set: %s\n', plant_parameter_set);
t_sweep_start = tic;

for d = 1:n_design
    coded   = BB(d,:);
    log_lvl = log_center + step_per_unit * coded;
    sigma   = 10.^log_lvl;

    fprintf('\n[Design %2d/%2d] coded=[%+d %+d %+d %+d]  sigma=[%.2e %.2e %.2e %.2e]\n', ...
        d, n_design, coded(1),coded(2),coded(3),coded(4), ...
        sigma(1),sigma(2),sigma(3),sigma(4));

    for r = 1:n_replicates
        run_idx = run_idx + 1;
        seed    = 1000*d + r;
        rng(seed, 'twister');

        sigma_vec = zeros(nx,1);
        for g = 1:4
            sigma_vec(state_groups{g}) = sigma(g);
        end

        % --- Solver reset + clean trajectory init ---
        if use_nmpc
            try
                ocp_solver.reset();
            catch
            end
        end
        x_current = x0_perturbed;
        x_full_current = reducedToFullState(x_current);
        if use_nmpc
            for k = 0:N
                ak = k/N;
                ocp_solver.set('x', (1-ak)*x_current + ak*xEq, k);
            end
            for k = 0:N-1
                ocp_solver.set('u', uEq, k);
            end
        end

        plot_x      = zeros(nx, sim_steps+1);
        plot_x(:,1) = x_current;
        diverged    = false;
        last_step   = sim_steps;

        for i = 1:sim_steps
            % --- EARLY EXIT: true state outside sanity box -> stop now,
            % BEFORE the OCP gets fed a pathological constr_x0. This is
            % what keeps the HPIPM internal state clean across replicates.
            if any(~isfinite(x_current)) || ...
               max(abs(x_current(1:2)))     > div_pos_xy || ...
               abs(x_current(3) - zEq_cas)  > div_pos_z  || ...
               max(abs(x_current(4:5)))     > div_angle  || ...
               max(abs(x_current(6:8)))     > div_lin_v  || ...
               max(abs(x_current(9:10)))    > div_ang_v
                diverged  = true;
                last_step = i;
                break;
            end

            % --- Inject measurement noise on the state passed to the OCP ---
            x_meas = x_current + sigma_vec .* randn(nx,1);

            u_applied = getNoiseTestAction(controller_type, ocp_solver, agent, x_meas, xEq, ...
                agent_observation, rl_action_sign, rl_action_order);

            % --- True plant integrated from true state, with controller u ---
            [x_next, x_full_current] = stepNoiseTestPlant(plant_integrator, sim_solver, f_func_r, ...
                x_current, x_full_current, u_applied, dt_mpc, paramsFast);

            % --- Warm-start shift ---
            if use_nmpc
                x_traj = ocp_solver.get('x');
                u_traj = ocp_solver.get('u');
                for k = 0:N
                    if k < N
                        ocp_solver.set('x', x_traj(:, k+2), k);
                    else
                        ocp_solver.set('x', xEq, k);
                    end
                end
                for k = 0:N-1
                    if k < N-1
                        ocp_solver.set('u', u_traj(:, k+2), k);
                    else
                        ocp_solver.set('u', uEq, k);
                    end
                end
            end

            plot_x(:, i+1) = x_next;
            x_current      = x_next;
        end

        % --- Response metrics ---
        if diverged
            y_diverged(run_idx) = 1;
            % Penalise divergence with a large but finite RMS in log10 space.
            y_rms(run_idx)    = 1.0;
            y_maxpos(run_idx) = 1.0;
        else
            err = (plot_x(:, settle_idx+1:end) - xEq) .* err_weight;
            y_rms(run_idx) = sqrt(mean(sum(err.^2, 1)));
            % Max position deviation FROM equilibrium (bug fix: subtract xEq).
            pos_dev = plot_x(1:3, 2:end) - xEq(1:3);
            y_maxpos(run_idx) = max(sqrt(sum(pos_dev.^2, 1)));
        end

        results(run_idx, :) = {d, r, coded(1), coded(2), coded(3), coded(4), ...
                               sigma(1), sigma(2), sigma(3), sigma(4), seed};

        fprintf('  rep %d/%d (seed=%d):  diverged=%d  step=%d  y_rms=%.4e  max|p-eq|=%.3e\n', ...
            r, n_replicates, seed, diverged, last_step, y_rms(run_idx), y_maxpos(run_idx));
    end
end

t_sweep = toc(t_sweep_start);
fprintf('\n--- Sweep finished in %.1f s (%.2f s / run avg) ---\n', t_sweep, t_sweep/n_runs);

results.y_rms      = y_rms;
results.y_maxpos   = y_maxpos;
results.y_diverged = y_diverged;

%% --- SAVE RAW DATA ---
rsm_data = struct();
rsm_data.results       = results;
rsm_data.controller_type = controller_type;
rsm_data.controller_name = controller_name;
rsm_data.agent_path    = agent_path;
rsm_data.agent_observation = agent_observation;
rsm_data.rl_policy_mode = rl_policy_mode;
rsm_data.rl_action_sign = rl_action_sign;
rsm_data.rl_action_order = rl_action_order;
rsm_data.plant_integrator = plant_integrator;
rsm_data.plant_parameter_set = plant_parameter_set;
rsm_data.factor_names  = factor_names;
rsm_data.log_center    = log_center;
rsm_data.step_per_unit = step_per_unit;
rsm_data.design_matrix = BB;
rsm_data.n_replicates  = n_replicates;
rsm_data.sim_steps     = sim_steps;
rsm_data.err_weight    = err_weight;
rsm_data.y_rms_base    = y_rms_base;       % noise-free RMS floor for reference
rsm_data.div_bounds    = struct('pos_xy',div_pos_xy,'pos_z',div_pos_z, ...
                                'angle',div_angle,'lin_v',div_lin_v,'ang_v',div_ang_v);
rsm_data.ofat          = ofat;
rsm_data.ofat_thresholds = ofat_thresholds;

results_file = ['rsm_noise_results' output_suffix '.mat'];
save(results_file, '-struct', 'rsm_data');
fprintf('Raw data saved to %s\n', results_file);

%% =========================================================================
%  RESPONSE SURFACE FIT (second-order polynomial with interactions)
% =========================================================================
fprintf('\n--- Fitting RSM (quadratic) to log10(y_rms) ---\n');
% Use log(y_rms) so the surface is well-conditioned (response spans decades).
results.log_y_rms = log10(max(results.y_rms, 1e-8));

% IMPORTANT: only fit the continuous response on SURVIVING runs.
% Diverged runs are penalised with y_rms=1.0 -- including them would make
% the quadratic surface fit the binary penalty rather than the smooth
% noise-induced RMS. Divergence is handled separately by the logistic fit.
surv = results.y_diverged == 0;
n_surv = sum(surv);
fprintf('  Fitting on %d surviving runs (of %d total).\n', n_surv, height(results));
if n_surv < 15
    warning('Only %d surviving runs -- quadratic RSM (15 coefficients) is rank-deficient. Shift log_center smaller and rerun.', n_surv);
end

mdl = fitlm(results(surv,:), ...
    ['log_y_rms ~ f1 + f2 + f3 + f4 + ' ...
     'f1:f2 + f1:f3 + f1:f4 + f2:f3 + f2:f4 + f3:f4 + ' ...
     'f1^2 + f2^2 + f3^2 + f4^2']);

disp(mdl);
fprintf('R^2 = %.4f   adjusted R^2 = %.4f\n', mdl.Rsquared.Ordinary, mdl.Rsquared.Adjusted);

% Logistic-style fit of P(diverged) -- where the boundary lies
fprintf('\n--- Fitting logistic regression for divergence probability ---\n');
mdl_div = fitglm(results, ...
    ['y_diverged ~ f1 + f2 + f3 + f4 + ' ...
     'f1:f2 + f1:f3 + f1:f4 + f2:f3 + f2:f4 + f3:f4 + ' ...
     'f1^2 + f2^2 + f3^2 + f4^2'], ...
    'Distribution','binomial','Link','logit');
disp(mdl_div);

%% --- CONTOUR PLOTS: pairwise effects (other factors held at center 0) ---
fig1 = figure('Name','RSM contours: log10(y_rms)','Position',[60 60 1200 800]);
pair_list = nchoosek(1:4, 2);
[fg, ~]   = meshgrid(linspace(-1,1,40), linspace(-1,1,40));
gx        = linspace(-1,1,40);
[GX, GY]  = meshgrid(gx, gx);

for k = 1:size(pair_list,1)
    subplot(2,3,k);
    i1 = pair_list(k,1);
    i2 = pair_list(k,2);

    pred = table();
    fvals = zeros(numel(GX), 4);
    fvals(:, i1) = GX(:);
    fvals(:, i2) = GY(:);
    pred.f1 = fvals(:,1);
    pred.f2 = fvals(:,2);
    pred.f3 = fvals(:,3);
    pred.f4 = fvals(:,4);
    z = predict(mdl, pred);
    Z = reshape(z, size(GX));

    contourf(GX, GY, Z, 15, 'LineColor','none'); hold on;
    colormap(parula);
    colorbar;
    % Overlay design points (coded)
    scatter(results.(sprintf('f%d',i1)), results.(sprintf('f%d',i2)), ...
        25, 'w', 'filled', 'MarkerEdgeColor','k');
    xlabel(sprintf('coded %s', factor_names{i1}), 'Interpreter','none');
    ylabel(sprintf('coded %s', factor_names{i2}), 'Interpreter','none');
    title(sprintf('log10(y\\_rms): %s vs %s', factor_names{i1}, factor_names{i2}), 'Interpreter','tex');
end
sgtitle('Response surface contours -- pairwise (other factors = center)');
savefig(fig1, ['rsm_contours_yrms' output_suffix '.fig']);

%% --- DIVERGENCE-PROBABILITY CONTOURS ---
fig2 = figure('Name','RSM contours: P(diverged)','Position',[60 60 1200 800]);
for k = 1:size(pair_list,1)
    subplot(2,3,k);
    i1 = pair_list(k,1);
    i2 = pair_list(k,2);

    fvals = zeros(numel(GX), 4);
    fvals(:, i1) = GX(:);
    fvals(:, i2) = GY(:);
    pred = table(fvals(:,1),fvals(:,2),fvals(:,3),fvals(:,4), ...
        'VariableNames',{'f1','f2','f3','f4'});
    z = predict(mdl_div, pred);
    Z = reshape(z, size(GX));

    contourf(GX, GY, Z, linspace(0,1,11), 'LineColor','none'); hold on;
    clim([0 1]); colorbar;
    contour(GX, GY, Z, [0.5 0.5], 'r', 'LineWidth', 2);  % 50% failure boundary
    scatter(results.(sprintf('f%d',i1)), results.(sprintf('f%d',i2)), ...
        25, 'w', 'filled', 'MarkerEdgeColor','k');
    xlabel(sprintf('coded %s', factor_names{i1}), 'Interpreter','none');
    ylabel(sprintf('coded %s', factor_names{i2}), 'Interpreter','none');
    title(sprintf('P(diverged): %s vs %s', factor_names{i1}, factor_names{i2}), 'Interpreter','tex');
end
sgtitle('Divergence probability -- red line = 50% failure boundary');
savefig(fig2, ['rsm_contours_pdiv' output_suffix '.fig']);

%% --- MAIN-EFFECTS PLOT (Pareto-style) ---
fig3 = figure('Name','RSM main effects','Position',[100 100 900 500]);
coefs = mdl.Coefficients;
% Drop intercept for bar plot
mask = ~strcmp(coefs.Row, '(Intercept)');
[srt, ord] = sort(abs(coefs.Estimate(mask)), 'descend');
names_eff  = coefs.Row(mask);
estimates  = coefs.Estimate(mask);

barh(estimates(ord)); hold on; grid on;
set(gca, 'YTick', 1:numel(ord), 'YTickLabel', names_eff(ord), 'YDir','reverse');
xlabel('Coefficient (log10 y\_rms)');
title('Standardised effects on log10(RMS tracking error)');
savefig(fig3, ['rsm_main_effects' output_suffix '.fig']);

%% --- TEXT REPORT: success rate per design + tolerances ---
fprintf('\n--- Summary ---\n');
fprintf('  Baseline (no noise) y_rms = %.4e\n', y_rms_base);
n_total_success = sum(results.y_diverged == 0);
fprintf('  Successful runs: %d / %d (%.0f%%)\n', n_total_success, height(results), 100*n_total_success/height(results));

% Per-design success rate (one row per design point)
fprintf('\n  Per design-point divergence rate:\n');
fprintf('   d  | coded f1 f2 f3 f4 | sigma_pos sigma_ori sigma_vel sigma_ang | #div\n');
for d = 1:n_design
    mask = results.design == d;
    nd   = sum(mask);
    div  = sum(results.y_diverged(mask));
    rr   = find(mask, 1, 'first');
    fprintf('  %3d | %+d %+d %+d %+d | %.2e %.2e %.2e %.2e | %d/%d\n', d, ...
        results.f1(rr), results.f2(rr), results.f3(rr), results.f4(rr), ...
        results.sigma_pos(rr), results.sigma_ori(rr), results.sigma_vel(rr), results.sigma_ang(rr), ...
        div, nd);
end

fprintf('\n--- Single-factor tolerance estimates (P(diverged) = 0.5) ---\n');
fprintf('     (other three factors held at coded 0 = baseline)\n');
for f = 1:4
    coded_grid = linspace(-1.5, 2.0, 200)';
    fvals      = zeros(numel(coded_grid), 4);
    fvals(:,f) = coded_grid;
    pred = table(fvals(:,1),fvals(:,2),fvals(:,3),fvals(:,4), ...
        'VariableNames',{'f1','f2','f3','f4'});
    pdiv = predict(mdl_div, pred);
    idx  = find(pdiv > 0.5, 1, 'first');
    if isempty(idx)
        fprintf('  %-10s : controller survives across entire tested range\n', factor_names{f});
    else
        coded_50 = coded_grid(idx);
        sigma_50 = 10^(log_center(f) + step_per_unit * coded_50);
        fprintf('  %-10s : 50%%-failure boundary at coded=%.2f  ->  sigma = %.3e\n', ...
            factor_names{f}, coded_50, sigma_50);
    end
end

fprintf('\nDone. Edit n_replicates / step_per_unit / log_center to refine.\n');

function u = getNoiseTestAction(controller_type, ocp_solver, agent, x_meas, xEq, ...
    agent_observation, rl_action_sign, rl_action_order)
%GETNOISETESTACTION Compute bounded controller input for the noise tests.
    switch lower(controller_type)
        case 'nmpc'
            ocp_solver.set('constr_x0', x_meas);
            ocp_solver.solve();
            u = ocp_solver.get('u', 0);

        case {'rl100','rl1000'}
            switch lower(agent_observation)
                case 'state'
                    x_for_agent = x_meas;
                case 'error'
                    x_for_agent = x_meas - xEq;
                otherwise
                    error('Unsupported agent_observation "%s". Use "state" or "error".', agent_observation);
            end

            u = getAction(agent, x_for_agent);
            if iscell(u)
                u = u{1};
            end
            if isa(u, 'dlarray')
                u = extractdata(u);
            end

        otherwise
            error('Unsupported controller_type "%s". Use "nmpc", "rl100", or "rl1000".', controller_type);
    end

    u = double(u(:));
    if numel(u) ~= 4
        error('Controller returned %d actions, expected 4.', numel(u));
    end
    if ~strcmpi(controller_type, 'nmpc')
        if numel(rl_action_order) ~= 4 || any(sort(rl_action_order(:))' ~= 1:4)
            error('rl_action_order must be a permutation of 1:4.');
        end
        u = rl_action_sign * u(rl_action_order);
    end
    u = max(min(u, 1), -1);
end

function [x_next, x_full_next] = stepNoiseTestPlant(plant_integrator, sim_solver, f_func_r, ...
    x_current, x_full_current, u_applied, dt, paramsFast)
%STEPNOISETESTPLANT Propagate the reduced-order plant by one controller step.
    switch lower(plant_integrator)
        case 'acados'
            sim_solver.set('x', x_current);
            sim_solver.set('u', u_applied);
            sim_solver.solve();
            x_next = sim_solver.get('xn');
            x_full_next = reducedToFullState(x_next);

        case {'ode15s','ode15'}
            ode_rhs = @(~, x) full(f_func_r(x, u_applied));
            ode_opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
            [~, x_traj] = ode15s(ode_rhs, [0 dt], x_current, ode_opts);
            x_next = x_traj(end, :)';
            x_full_next = reducedToFullState(x_next);

        case 'ode15s_full'
            ode_rhs = @(~, x) maglevSystemDynamics(x, u_applied, paramsFast, 'fast');
            ode_opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
            [~, x_traj] = ode15s(ode_rhs, [0 dt], x_full_current, ode_opts);
            x_full_next = x_traj(end, :)';
            x_next = fullToReducedState(x_full_next);

        otherwise
            error('Unsupported plant_integrator "%s". Use "acados", "ode15s", or "ode15s_full".', plant_integrator);
    end
end

function x_full = reducedToFullState(x_r)
    x_full = [x_r(1:5); 0; x_r(6:8); x_r(9:10); 0];
end

function x_r = fullToReducedState(x_full)
    x_r = [x_full(1:5); x_full(7:11)];
end
