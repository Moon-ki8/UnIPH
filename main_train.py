#%%
import os
import sys
import warnings

import matplotlib.pyplot as plt
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
from utils.utils_device import resolve_device
from utils.utils_irreps import run_irreps_parity_checks
from utils.utils_model import e3net, train


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
loss_cfg = cfg['loss']
settings_cfg = cfg['settings']

device = resolve_device(settings_cfg['device'])

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

with open(data_cfg['train_idx_path'], 'r') as f:
    idx_train = [int(i.strip()) for i in f.readlines() if i.strip()]
with open(data_cfg['test_idx_path'], 'r') as f:
    idx_valid = [int(i.strip()) for i in f.readlines() if i.strip()]
with open(data_cfg['test_idx_path'], 'r') as f:
    idx_test = [int(i.strip()) for i in f.readlines() if i.strip()]


data_train = tg.loader.DataLoader(
    data_all.iloc[idx_train]['data'].values,
    batch_size=training_cfg['batch_size'],
    shuffle=True,
)
data_valid = tg.loader.DataLoader(
    data_all.iloc[idx_valid]['data'].values,
    batch_size=training_cfg['batch_size'],
)

print(f'Train/Valid/Test sizes: {len(idx_train)}/{len(idx_valid)}/{len(idx_test)}')


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

print(gnn_model)

opt = torch.optim.AdamW(gnn_model.parameters(), lr=training_cfg['lr'], weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode='min',
    factor=training_cfg['gamma'],
    patience=20,
)
loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.L1Loss()

train(
    model=gnn_model,
    optimizer=opt,
    dataloader_train=data_train,
    dataloader_valid=data_valid,
    loss_fn=loss_fn,
    loss_fn_mae=loss_fn_mae,
    run_name=model_cfg['gnn_model_name'],
    loss_weights=loss_cfg,
    max_iter=training_cfg['max_iter'],
    scheduler=scheduler,
    device=device,
    rms_f=data_cfg['rms_f'],
    rms_tb=data_cfg['rms_tb'],
)

history = torch.load(model_cfg['gnn_model_name'] + '.torch', map_location=device)['history']
steps = [d['step'] + 1 for d in history]
loss_train = [d['train']['loss'] for d in history]
loss_valid = [d['valid']['loss'] for d in history]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(steps, loss_train, 'o-', label='Training')
ax.plot(steps, loss_valid, 'o-', label='Validation')
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.set_yscale('log')
ax.legend(frameon=False)
fig.savefig('convergence.png', dpi=300)
print('Saved convergence.png')
