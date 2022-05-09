import sys
import yaml

import optuna

from run import run


def suggest_params(trial: optuna.trial.BaseTrial, sample_spaces: dict):
    params = {}

    for variable_name, variable_specs in sample_spaces.items():
        if variable_specs['type'] == 'categorical':
            params[variable_name] = trial.suggest_categorical(variable_name, **variable_specs['sample_space'])
        elif variable_specs['type'] == 'int':
            params[variable_name] = trial.suggest_int(variable_name, **variable_specs['sample_space'])
        elif variable_specs['type'] == 'float':
            params[variable_name] = trial.suggest_float(variable_name, **variable_specs['sample_space'])
        else:
            raise NameError('Type of name {} cannot be suggested'.format(variable_specs['type']))

    return params


def objective(trial, config_dict: dict):
    run_params = get_run_params_from_config(config_dict, trial)

    return run(**run_params)


def get_run_params_from_config(config_dict, trial):
    run_params = {}

    fixed_params = config_dict['fixed_params']
    optimizable_params = config_dict['optimizable_params']

    run_params['data_path'] = fixed_params['data_path']
    run_params['dist_type'] = fixed_params['dist_type']
    run_params['loss_type'] = fixed_params['loss_type']
    run_params['model_type'] = fixed_params['model_type']

    for params_class in ['data_params', 'loss_params', 'model_params', 'optimizer_params', 'trainer_params']:
        run_params[params_class] = fixed_params[params_class].copy() if params_class in fixed_params else {}
        if params_class in optimizable_params:
            run_params[params_class].update(suggest_params(trial, optimizable_params[params_class]))

    return run_params


def run_study(config_dict, n_trials=10):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, config_dict), n_trials=n_trials)


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    run_study(config)
