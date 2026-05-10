import ast
import configparser
import os


def _str_to_bool(value):
    return value.strip().lower() in ['true', '1', 'yes']


def _resolve_path(config_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(config_dir, path_value))


def load_and_validate_config(config_path):
    parser = configparser.ConfigParser()
    read_files = parser.read(config_path)
    if len(read_files) == 0:
        raise FileNotFoundError(f'Could not read config file: {config_path}')

    config_dir = os.path.dirname(os.path.abspath(config_path))

    required_sections = ['Model', 'Data', 'Training', 'Loss', 'Settings']
    for section in required_sections:
        if section not in parser:
            raise KeyError(f'Missing required section: [{section}]')

    model_required = [
        'gnn_model_name',
        'inter_irreps_node',
        'inter_irreps_edge',
        'num_layer',
        'lmax',
        'max_radius',
        'in_dim',
        'irreps_E',
        'fcn_len',
        'number_of_basis',
        'orbital_types',
        'spin_flag',
    ]
    data_required = [
        'data_dir',
        'csv_path',
        'pkl_path',
        'train_idx_path',
        'test_idx_path',
        'mean_energy',
        'rms_f',
        'rms_tb',
    ]
    training_required = ['max_iter', 'lr', 'gamma', 'batch_size']
    loss_required = ['weight_energy', 'weight_forces', 'weight_tb']
    settings_required = ['device']

    for key in model_required:
        if key not in parser['Model']:
            raise KeyError(f'Missing required [Model] key: {key}')
    for key in data_required:
        if key not in parser['Data']:
            raise KeyError(f'Missing required [Data] key: {key}')
    for key in training_required:
        if key not in parser['Training']:
            raise KeyError(f'Missing required [Training] key: {key}')
    for key in loss_required:
        if key not in parser['Loss']:
            raise KeyError(f'Missing required [Loss] key: {key}')
    for key in settings_required:
        if key not in parser['Settings']:
            raise KeyError(f'Missing required [Settings] key: {key}')

    model_cfg = {
        'gnn_model_name': parser['Model']['gnn_model_name'],
        'inter_irreps_node': parser['Model']['inter_irreps_node'],
        'inter_irreps_edge': parser['Model']['inter_irreps_edge'],
        'num_layer': int(parser['Model']['num_layer']),
        'lmax': int(parser['Model']['lmax']),
        'max_radius': float(parser['Model']['max_radius']),
        'in_dim': int(parser['Model']['in_dim']),
        'irreps_E': parser['Model']['irreps_E'],
        'fcn_len': int(parser['Model']['fcn_len']),
        'number_of_basis': int(parser['Model']['number_of_basis']),
        'orbital_types': ast.literal_eval(parser['Model']['orbital_types']),
        'spin_flag': _str_to_bool(parser['Model']['spin_flag']),
    }

    data_cfg = {
        'data_dir': _resolve_path(config_dir, parser['Data']['data_dir']),
        'csv_path': _resolve_path(config_dir, parser['Data']['csv_path']),
        'pkl_path': _resolve_path(config_dir, parser['Data']['pkl_path']),
        'train_idx_path': _resolve_path(config_dir, parser['Data']['train_idx_path']),
        'test_idx_path': _resolve_path(config_dir, parser['Data']['test_idx_path']),
        'mean_energy': float(parser['Data']['mean_energy']),
        'rms_f': float(parser['Data']['rms_f']),
        'rms_tb': float(parser['Data']['rms_tb']),
    }

    training_cfg = {
        'max_iter': int(parser['Training']['max_iter']),
        'lr': float(parser['Training']['lr']),
        'gamma': float(parser['Training']['gamma']),
        'batch_size': int(parser['Training']['batch_size']),
        'weight_ratio': float(parser['Training'].get('weight_ratio', '1.0')),
    }

    loss_cfg = {
        'weight_energy': float(parser['Loss']['weight_energy']),
        'weight_forces': float(parser['Loss']['weight_forces']),
        'weight_tb': float(parser['Loss']['weight_tb']),
    }

    settings_cfg = {
        'device': parser['Settings']['device'],
    }

    if model_cfg['spin_flag']:
        raise ValueError('PLAN_01 requires spin_flag=False.')
    if training_cfg['batch_size'] != 1:
        raise ValueError('PLAN_01 requires batch_size=1.')

    file_keys = ['csv_path', 'train_idx_path', 'test_idx_path']
    for key in file_keys:
        if not os.path.isfile(data_cfg[key]):
            raise FileNotFoundError(f'Missing required file for {key}: {data_cfg[key]}')

    if not os.path.isdir(data_cfg['data_dir']):
        raise FileNotFoundError(f'Missing data_dir: {data_cfg["data_dir"]}')

    return {
        'model': model_cfg,
        'data': data_cfg,
        'training': training_cfg,
        'loss': loss_cfg,
        'settings': settings_cfg,
    }
