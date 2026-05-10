import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg

from tqdm import tqdm

from utils.utils_config import load_and_validate_config
from utils.utils_data import (
    build_atom_encoders,
    build_converter_list,
    build_orbital_edges_and_codes,
    graph_build,
    load_data,
)
from utils.utils_eval import compute_mae_summary, evaluate_datasets, save_eval_results_npz
from utils.utils_irreps import run_irreps_parity_checks
from utils.utils_model import e3net


warnings.filterwarnings('ignore')
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
tqdm.pandas()


if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gnn_config_mos2.ini')

cfg = load_and_validate_config(config_path)

model_cfg = cfg['model']
data_cfg = cfg['data']
training_cfg = cfg['training']
settings_cfg = cfg['settings']

if torch.cuda.is_available():
    device = settings_cfg['device']
else:
    device = 'cpu'

print(f'Using config: {config_path}')
print(f'Using device: {device}')
print(f"weight_ratio from ini: {training_cfg['weight_ratio']}")


type_encoding, _, am_onehot = build_atom_encoders(dtype=default_dtype)
orbital_edges, pair_to_code = build_orbital_edges_and_codes(model_cfg['orbital_types'])
converter_list = build_converter_list(
    orbital_edges=orbital_edges,
    spin_flag=model_cfg['spin_flag'],
    dtype=default_dtype,
    device='cpu',
)

run_irreps_parity_checks(
    orbital_edges=orbital_edges,
    spin_flag=model_cfg['spin_flag'],
    dtype=default_dtype,
    device='cpu',
)

if os.path.isfile(data_cfg['pkl_path']):
    print(f"Loading cached PKL data from: {data_cfg['pkl_path']}")
    data_all = pd.read_pickle(data_cfg['pkl_path'])
else:
    print('Loading CSV data...')
    data_all = load_data(data_cfg['csv_path'])
    print('Building graph objects from CSV...')
    data_all['data'] = data_all.progress_apply(
        lambda row: graph_build(
            temp_data=row,
            pair_to_code=pair_to_code,
            converter_list=converter_list,
            type_encoding=type_encoding,
            am_onehot=am_onehot,
            r_max=model_cfg['max_radius'],
        ),
        axis=1,
    )

    pkl_dir = os.path.dirname(data_cfg['pkl_path'])
    if pkl_dir:
        os.makedirs(pkl_dir, exist_ok=True)
    data_all.to_pickle(data_cfg['pkl_path'])
    print(f"Saved rebuilt PKL snapshot to: {data_cfg['pkl_path']}")

with open(data_cfg['test_idx_path'], 'r') as f:
    idx_test = [int(i.strip()) for i in f.readlines() if i.strip()]
with open(data_cfg['train_idx_path'], 'r') as f:
    idx_train = [int(i.strip()) for i in f.readlines() if i.strip()]


data_train = tg.loader.DataLoader(
    data_all.iloc[idx_train]['data'].values,
    batch_size=1,
)
data_test = tg.loader.DataLoader(
    data_all.iloc[idx_test]['data'].values,
    batch_size=1,
)


gnn_model = e3net(
    in_dim=model_cfg['in_dim'],
    irreps_intermediate_node=model_cfg['inter_irreps_node'],
    irreps_intermediate_edge=model_cfg['inter_irreps_edge'],
    orbital_edges=orbital_edges,
    irreps_E=model_cfg['irreps_E'],
    lmax=model_cfg['lmax'],
    num_layer=model_cfg['num_layer'],
    max_radius=model_cfg['max_radius'],
    fcn_len=model_cfg['fcn_len'],
    number_of_basis=model_cfg['number_of_basis'],
    spin_flag=model_cfg['spin_flag'],
)

state_path = model_cfg['gnn_model_name'] + '.torch'
if not os.path.isfile(state_path):
    raise FileNotFoundError(f'Missing trained checkpoint: {state_path}')

checkpoint = torch.load(state_path, map_location=torch.device(device))
gnn_model.load_state_dict(checkpoint['state'])

train_results, test_results = evaluate_datasets(
    gnn_model=gnn_model,
    dataloader_train=data_train,
    dataloader_test=data_test,
    device=device,
    converter_list=converter_list,
    mean_energy=data_cfg['mean_energy'],
    rms_force=data_cfg['rms_f'],
    rms_tb_data=data_cfg['rms_tb'],
)

script_dir = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(script_dir, 'eval_exports')
os.makedirs(export_dir, exist_ok=True)

config_stem = os.path.splitext(os.path.basename(config_path))[0]
export_prefix = f"{model_cfg['gnn_model_name']}_{config_stem}"
plot_data_path = os.path.join(export_dir, f'{export_prefix}.npz')
metrics_path = os.path.join(export_dir, f'{export_prefix}_mae.csv')

save_eval_results_npz('eval_results.npz', train_results, test_results)
save_eval_results_npz(plot_data_path, train_results, test_results)

mae_summary = compute_mae_summary(train_results, test_results)

metrics_df = pd.DataFrame([
    {
        'split': 'train',
        'mae_energy_meV_per_atom': mae_summary['train']['energy'] * 1000.0,
        'mae_force_meV_per_Angstrom': mae_summary['train']['force'] * 1000.0,
        'mae_tb_meV': mae_summary['train']['tb'] * 1000.0,
    },
    {
        'split': 'test',
        'mae_energy_meV_per_atom': mae_summary['test']['energy'] * 1000.0,
        'mae_force_meV_per_Angstrom': mae_summary['test']['force'] * 1000.0,
        'mae_tb_meV': mae_summary['test']['tb'] * 1000.0,
    },
])
metrics_df.to_csv(metrics_path, index=False)

print(
    f'Train MAE Energy: {mae_summary["train"]["energy"] * 1000:.3f} meV, '
    f'Train MAE Force: {mae_summary["train"]["force"] * 1000:.3f} meV/Angstrom, '
    f'Train MAE TB: {mae_summary["train"]["tb"] * 1000:.3f} meV'
)
print(
    f'Test MAE Energy: {mae_summary["test"]["energy"] * 1000:.3f} meV, '
    f'Test MAE Force: {mae_summary["test"]["force"] * 1000:.3f} meV/Angstrom, '
    f'Test MAE TB: {mae_summary["test"]["tb"] * 1000:.3f} meV'
)
print(f'Saved plot data: {plot_data_path}')
print(f'Saved MAE summary: {metrics_path}')
print('Saved eval_results.npz (training + test)')
