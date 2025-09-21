import os
import json
import datetime

import numpy as np
import optuna

import sys
sys.path.append('/Users/fmjp/Desktop/lab/simu/tvgti')

from exog_sparse_sim.data_gen import generate_piecewise_Y_with_exog
from exog_sparse_sim.models.pp_exog import PPExogenousSEM
from models.tvgti_pc.time_varing_sem import TimeVaryingSEM as PCSEM


def tune_piecewise_all_methods(
    N: int = 20,
    T: int = 1000,
    sparsity: float = 0.7,
    max_weight: float = 0.5,
    std_e: float = 0.05,
    K: int = 4,
    tuning_trials: int = 30,
    tuning_runs_per_trial: int = 5,
    seed: int = 3,
):
    np.random.seed(seed)
    penalty_value = 1e6
    T_tune = min(T, 400)

    # 既定ハイパラ（fallback）
    best = {
        'pp': {'r': 50, 'q': 5, 'rho': 1e-3, 'mu_lambda': 0.05},
        'pc': {'lambda_reg': 1e-3, 'alpha': 1e-2, 'beta': 1e-2, 'gamma': 0.9, 'P': 1, 'C': 1},
        'co': {'beta_co': 0.02},
        'sgd': {'beta_sgd': 0.0269},
    }

    # PP objective
    def objective_pp(trial):
        r_suggested = best['pp']['r']
        q_suggested = best['pp']['q']
        rho_suggested = trial.suggest_float('rho', 1e-6, 1e-1, log=True)
        mu_lambda_suggested = trial.suggest_float('mu_lambda', 1e-4, 1.0, log=True)
        errs = []
        for _ in range(tuning_runs_per_trial):
            S_ser, B_tr, U_gen, Y_gen = generate_piecewise_Y_with_exog(
                N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
                s_type='random', b_min=0.5, b_max=1.0, u_dist='uniform01')
            S0 = np.zeros((N, N)); b0 = np.ones(N)
            model = PPExogenousSEM(N, S0, b0, r=r_suggested, q=q_suggested, rho=rho_suggested, mu_lambda=mu_lambda_suggested)
            S_hat_list, _ = model.run(Y_gen, U_gen)
            errs.append(np.linalg.norm(S_hat_list[-1] - S_ser[-1], ord='fro'))
        return float(np.mean(errs))

    # PC objective
    def objective_pc(trial):
        # 広めに探索（ユーザ要望によりステップサイズの上限を拡大）
        lambda_reg_suggested = trial.suggest_float('lambda_reg', 1e-5, 1e-2, log=True)
        alpha_suggested = trial.suggest_float('alpha', 1e-4, 2e-1, log=True)
        beta_suggested = trial.suggest_float('beta_pc', 1e-4, 3e-1, log=True)
        gamma_suggested = trial.suggest_float('gamma', 0.85, 0.999)
        P_suggested = trial.suggest_int('P', 0, 2)
        C_suggested = trial.suggest_categorical('C', [1, 2, 5])
        errs = []
        for _ in range(tuning_runs_per_trial):
            S_ser, B_tr, U_gen, Y_gen = generate_piecewise_Y_with_exog(
                N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
                s_type='random', b_min=0.5, b_max=1.0, u_dist='uniform01')
            # 短縮系列で評価
            X = Y_gen[:, :T_tune]
            S_ser = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            try:
                pc = PCSEM(N, S0_pc, lambda_reg_suggested, alpha_suggested, beta_suggested, gamma_suggested, P_suggested, C_suggested, show_progress=False, name='pc_baseline')
                estimates_pc, _ = pc.run(X)
                err_ts = [np.linalg.norm(estimates_pc[t] - S_ser[t], ord='fro') for t in range(len(S_ser))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    # CO objective
    def objective_co(trial):
        beta_co_suggested = trial.suggest_float('beta_co', 1e-5, 1e0, log=True)
        gamma_suggested = trial.suggest_float('gamma', 0.85, 0.999)
        C_suggested = trial.suggest_categorical('C', [1, 2, 5])
        errs = []
        for _ in range(tuning_runs_per_trial):
            S_ser, B_tr, U_gen, Y_gen = generate_piecewise_Y_with_exog(
                N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
                s_type='random', b_min=0.5, b_max=1.0, u_dist='uniform01')
            X = Y_gen[:, :T_tune]
            S_ser = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            lambda_reg = best['pc']['lambda_reg']
            alpha = best['pc']['alpha']
            P = 0
            try:
                co = PCSEM(N, S0_pc, lambda_reg, alpha, beta_co_suggested, gamma_suggested, P, C_suggested, show_progress=False, name='co_baseline')
                estimates_co, _ = co.run(X)
                err_ts = [np.linalg.norm(estimates_co[t] - S_ser[t], ord='fro') for t in range(len(S_ser))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    # SGD objective
    def objective_sgd(trial):
        beta_sgd_suggested = trial.suggest_float('beta_sgd', 1e-5, 1e0, log=True)
        errs = []
        for _ in range(tuning_runs_per_trial):
            S_ser, B_tr, U_gen, Y_gen = generate_piecewise_Y_with_exog(
                N=N, T=T, sparsity=sparsity, max_weight=max_weight, std_e=std_e, K=K,
                s_type='random', b_min=0.5, b_max=1.0, u_dist='uniform01')
            X = Y_gen[:, :T_tune]
            S_ser = S_ser[:T_tune]
            S0_pc = np.zeros((N, N))
            lambda_reg = best['pc']['lambda_reg']
            alpha = best['pc']['alpha']
            gamma = 0.0
            P = 0
            C = best['pc']['C']
            try:
                sgd = PCSEM(N, S0_pc, lambda_reg, alpha, beta_sgd_suggested, gamma, P, C, show_progress=False, name='sgd_baseline')
                estimates_sgd, _ = sgd.run(X)
                err_ts = [np.linalg.norm(estimates_sgd[t] - S_ser[t], ord='fro') for t in range(len(S_ser))]
                mean_err = float(np.mean(err_ts))
                if not np.isfinite(mean_err):
                    mean_err = penalty_value
                errs.append(mean_err)
            except Exception:
                errs.append(penalty_value)
        return float(np.mean(errs))

    # Run studies
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_pp, n_trials=tuning_trials)
    best['pp']['rho'] = study.best_params.get('rho', best['pp']['rho'])
    best['pp']['mu_lambda'] = study.best_params.get('mu_lambda', best['pp']['mu_lambda'])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_pc, n_trials=tuning_trials)
    best['pc']['lambda_reg'] = study.best_params.get('lambda_reg', best['pc']['lambda_reg'])
    best['pc']['alpha'] = study.best_params.get('alpha', best['pc']['alpha'])
    best['pc']['beta'] = study.best_params.get('beta_pc', best['pc']['beta'])
    best['pc']['gamma'] = study.best_params.get('gamma', best['pc']['gamma'])
    best['pc']['P'] = study.best_params.get('P', best['pc']['P'])
    best['pc']['C'] = study.best_params.get('C', best['pc']['C'])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_co, n_trials=tuning_trials)
    best['co']['beta_co'] = study.best_params.get('beta_co', best['co']['beta_co'])
    best['pc']['gamma'] = study.best_params.get('gamma', best['pc']['gamma'])
    best['pc']['C'] = study.best_params.get('C', best['pc']['C'])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_sgd, n_trials=tuning_trials)
    best['sgd']['beta_sgd'] = study.best_params.get('beta_sgd', best['sgd']['beta_sgd'])

    return best


def main():
    best = tune_piecewise_all_methods()
    # 保存
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    today_str = datetime.datetime.now().strftime('%y%m%d')
    save_dir = os.path.join('./result', today_str)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'piecewise_best_hyperparams_{timestamp}.json')
    with open(out_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f'Saved best hyperparams to: {out_path}')


if __name__ == '__main__':
    main()


