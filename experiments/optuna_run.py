import sys
import os
from typing import Union

import yaml
import argparse

import optuna

sys.path.append(os.path.join(sys.path[0], '..'))

from run import run

parser = argparse.ArgumentParser(description='Run experiments with Optuna hyperparameter optimization')
parser.add_argument('configuration_file', type=str)
parser.add_argument('--n_trials', type=int)
parser.add_argument('--debug', action='store_true')


def suggest_params(trial: optuna.trial.BaseTrial, sample_spaces: dict):
    params = {}

    for variable_name, variable_specs in sample_spaces.items():
        if variable_specs['type'] == 'categorical':
            params[variable_name] = trial.suggest_categorical(variable_name, **variable_specs['sample_space'])
        elif variable_specs['type'] == 'int':
            params[variable_name] = trial.suggest_int(variable_name, **variable_specs['sample_space'])
        elif variable_specs['type'] == 'float':
            params[variable_name] = trial.suggest_float(variable_name, **variable_specs['sample_space'])
        elif variable_specs['type'] == 'power':
            params[variable_name] = suggest_power(trial, variable_name, **variable_specs['sample_space'])
        elif variable_specs['type'] == 'list':
            params[variable_name] = suggest_list(trial, variable_name, **variable_specs['sample_space'])
        else:
            raise NameError('Type of name {} cannot be suggested'.format(variable_specs['type']))

    return params


def suggest_power(trial: optuna.trial.BaseTrial,
                  name: str,
                  min_exp: int,
                  max_exp: int,
                  base: Union[int, float],
                  include_zero: bool = False):
    if include_zero:
        assert min_exp == 0, "If 0 should be included in power-of-two sampling, the lowest exponent should be 0 too"
        min_exp = -1

    exponential = trial.suggest_int(name=name + "_exponent", low=min_exp, high=max_exp)

    if exponential == -1:
        return 0

    return base**exponential


def suggest_list(trial: optuna.trial.BaseTrial,
                 name: str,
                 element_type: str,
                 sample_space: dict,
                 max_size: int,
                 ignore_if_zero: bool):
    lst = []

    for i in range(max_size):
        sample_name = name + '_{}'.format(i)
        if element_type == 'categorical':
            lst.append(trial.suggest_categorical(sample_name, **sample_space))
        elif element_type == 'int':
            lst.append(trial.suggest_int(sample_name, **sample_space))
        elif element_type == 'float':
            lst.append(trial.suggest_float(sample_name, **sample_space))
        elif element_type == 'power':
            lst.append(suggest_power(trial, sample_name, **sample_space))
        else:
            raise NameError('Type of name {} cannot be suggested'.format(element_type))

    if ignore_if_zero:
        return [s for s in lst if s != 0]

    return lst


def objective(trial, config_dict: dict, debug: bool):
    run_params = get_run_params_from_config(config_dict, trial)
    save_path = os.path.join(os.path.dirname(__file__), 'optuna_runs', config_dict['name'],
                             'trial_{}'.format(trial.number))

    return run(**run_params, save_path=save_path, debug=debug)


def get_run_params_from_config(config_dict, trial):
    run_params = {}

    fixed_params = config_dict['fixed_params']
    optimizable_params = config_dict['optimizable_params']

    run_params['data_sets'] = fixed_params['data_sets']
    run_params['dist_type'] = fixed_params['dist_type']
    run_params['loss_type'] = fixed_params['loss_type']
    run_params['model_type'] = fixed_params['model_type']
    run_params['early_stopping_params'] = fixed_params['early_stopping_params']
    run_params['test_params'] = fixed_params['test_params']
    run_params['seed'] = fixed_params['seed']

    for params_class in ['data_params', 'loss_params', 'model_params', 'optimizer_params', 'trainer_params']:
        run_params[params_class] = fixed_params[params_class].copy() if params_class in fixed_params else {}
        if params_class in optimizable_params:
            run_params[params_class].update(suggest_params(trial, optimizable_params[params_class]))

    return run_params


def run_study(config_dict, n_trials, debug):
    storage = 'sqlite:///{}/optuna_runs/{}.db'.format(sys.path[0], config['name'])
    study = optuna.create_study(storage=storage,
                                study_name=config_dict['name'],
                                direction='maximize',
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, config_dict, debug), n_trials=n_trials, catch=(ValueError,))


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.configuration_file, 'r') as f:
        config = yaml.safe_load(f)

    run_study(config, n_trials=args.n_trials, debug=args.debug)
