import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import degree

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear

from tqdm import tqdm

from utils.utils_irreps import E3TensorDecomp


bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


class e3LayerNorm(nn.Module):
    def __init__(self, irreps_in, eps=1e-5, affine=True, normalization='component', subtract_mean=True, divide_norm=False):
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        self.eps = eps

        if affine:
            ib = 0
            iw = 0
            weight_slices = []
            bias_slices = []
            for mul, ir in irreps_in:
                if ir.is_scalar():
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones([iw]))
            self.bias = nn.Parameter(torch.zeros([ib]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        assert normalization in ['component', 'norm']
        self.normalization = normalization

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            self.weight.data.fill_(1)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        if batch is None:
            batch = torch.full([x.shape[0]], 0, dtype=torch.int64).to(x.device)

        batch_size = int(batch.max()) + 1
        batch_degree = degree(batch, batch_size, dtype=torch.int64).clamp_(min=1).to(dtype=x.dtype)

        out = []
        idx = 0
        for index, (mul, ir) in enumerate(self.irreps_in):
            field = x[:, idx: idx + mul * ir.dim].reshape(-1, mul, ir.dim)

            if self.subtract_mean or ir.l == 0:
                mean = scatter(field, batch, dim=0, dim_size=batch_size, reduce='add').mean(dim=1, keepdim=True)
                mean = mean / batch_degree[:, None, None]
                field = field - mean[batch]

            if self.divide_norm or ir.l == 0:
                norm = scatter(field.abs().pow(2), batch, dim=0, dim_size=batch_size, reduce='mean').mean(dim=[1, 2], keepdim=True)
                if self.normalization == 'norm':
                    norm = norm * ir.dim
                field = field / (norm.sqrt()[batch] + self.eps)

            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]
            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[index]]
                field = field + bias[None, :, None]

            out.append(field.reshape(-1, mul * ir.dim))
            idx += mul * ir.dim

        return torch.cat(out, dim=-1)


class E3conv(torch.nn.Module):
    def __init__(self, irreps_node_in, irreps_edge_fea, irreps_out, irreps_edge_attr, fcn_len):
        super().__init__()
        self.irreps_node_in = Irreps(irreps_node_in)
        self.irreps_edge_fea = Irreps(irreps_edge_fea)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_intermediate = Irreps(irreps_out)

        self.irreps_in = self.irreps_node_in + self.irreps_node_in + self.irreps_edge_fea

        act = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        irreps_gated = Irreps((mul, ir) for mul, ir in self.irreps_intermediate if ir.l > 0)
        num_gated = int(irreps_gated.num_irreps)
        irreps_gates = Irreps(f'{int(num_gated / 2)}x0o+{int(num_gated / 2)}x0e')
        irreps_scalars = Irreps((mul, ir) for mul, ir in self.irreps_intermediate if ir.l == 0)

        self.gate = Gate(
            irreps_scalars,
            [act[ir.p] for _, ir in irreps_scalars],
            irreps_gates,
            [act_gates[ir.p] for _, ir in irreps_gates],
            irreps_gated,
        )

        self.fctp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            self.gate.irreps_in,
            shared_weights=False,
        )
        self.fcn = FullyConnectedNet([fcn_len, fcn_len, self.fctp.weight_numel], torch.nn.functional.silu)
        self.lin = Linear(irreps_in=self.fctp.irreps_out, irreps_out=self.fctp.irreps_out, biases=False)

    def forward(self, node_in_i, node_in_j, edge_fea, edge_attr, edge_len_embedded):
        weight = self.fcn(edge_len_embedded)
        x = torch.cat((node_in_i, node_in_j, edge_fea), dim=-1)
        x = self.fctp(x, edge_attr, weight)
        x = self.lin(x)
        x = self.gate(x)
        return x


class NodeUpdate(torch.nn.Module):
    def __init__(self, irreps_node_in, irreps_edge_fea, irreps_out, irreps_edge_attr, fcn_len):
        super().__init__()
        self.e3conv = E3conv(irreps_node_in, irreps_edge_fea, irreps_out, irreps_edge_attr, fcn_len)
        self.norm = e3LayerNorm(irreps_out)

    def forward(self, node_fea, edge_src, edge_dst, edge_fea, edge_attr, edge_len_embedded):
        edge_update = self.e3conv(node_fea[edge_src], node_fea[edge_dst], edge_fea, edge_attr, edge_len_embedded)
        x = scatter(edge_update, edge_src, dim=0, dim_size=node_fea.shape[0], reduce='sum')
        return x


class EdgeUpdate(torch.nn.Module):
    def __init__(self, irreps_node_in, irreps_edge_fea, irreps_out, irreps_edge_attr, fcn_len):
        super().__init__()
        self.e3conv = E3conv(irreps_node_in, irreps_edge_fea, irreps_out, irreps_edge_attr, fcn_len)
        self.norm = e3LayerNorm(irreps_out)

    def forward(self, node_fea, edge_src, edge_dst, edge_fea, edge_attr, edge_len_embedded):
        edge_update = self.e3conv(node_fea[edge_src], node_fea[edge_dst], edge_fea, edge_attr, edge_len_embedded)
        return edge_update


class e3net(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        irreps_intermediate_node,
        irreps_intermediate_edge,
        orbital_edges,
        irreps_E,
        lmax,
        fcn_len,
        num_layer,
        max_radius,
        number_of_basis,
        spin_flag=False,
    ):
        super().__init__()
        self.irreps_edge_in = Irreps(f'{int(number_of_basis / 2)}x0o+{int(number_of_basis / 2)}x0e')
        self.irreps_node_in = self.irreps_edge_in
        self.irreps_intermediate_node = Irreps(irreps_intermediate_node)
        self.irreps_intermediate_edge = Irreps(irreps_intermediate_edge)
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.fcn_len = fcn_len
        self.spin_flag = spin_flag

        self.irreps_tbs = []
        for values in orbital_edges:
            orbital_types_left = values[2]
            orbital_types_right = values[3]
            out_js_list = [(l1, l2) for l1 in orbital_types_left for l2 in orbital_types_right]
            converter = E3TensorDecomp(
                None,
                out_js_list,
                default_dtype_torch=default_dtype,
                spinful=spin_flag,
                device_torch='cpu',
            )
            self.irreps_tbs.append(Irreps(converter.required_irreps_out))

        self.irreps_E = Irreps(irreps_E)
        self.em = torch.nn.Linear(in_dim, fcn_len)

        self.node_updates = torch.nn.ModuleList([])
        self.edge_updates = torch.nn.ModuleList([])
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                irreps_node_in_temp = self.irreps_node_in
                irreps_edge_in_temp = self.irreps_edge_in
            else:
                irreps_node_in_temp = self.irreps_intermediate_node
                irreps_edge_in_temp = self.irreps_intermediate_edge

            node_update = NodeUpdate(
                irreps_node_in=irreps_node_in_temp,
                irreps_edge_fea=irreps_edge_in_temp,
                irreps_out=self.irreps_intermediate_node,
                irreps_edge_attr=self.irreps_edge_attr,
                fcn_len=self.fcn_len,
            )
            edge_update = EdgeUpdate(
                irreps_node_in=self.irreps_intermediate_node,
                irreps_edge_fea=irreps_edge_in_temp,
                irreps_out=self.irreps_intermediate_edge,
                irreps_edge_attr=self.irreps_edge_attr,
                fcn_len=self.fcn_len,
            )

            self.node_updates.append(node_update)
            self.edge_updates.append(edge_update)

        self.lin_E = Linear(
            irreps_in=self.irreps_intermediate_node,
            irreps_out=self.irreps_intermediate_node,
            biases=False,
        )
        self.sc_E = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_intermediate_node,
            irreps_in2=self.irreps_intermediate_node,
            irreps_out=self.irreps_E,
        )

        self.sc_tbs = torch.nn.ModuleList([])
        for idx in range(len(orbital_edges)):
            sc_tb = FullyConnectedTensorProduct(
                irreps_in1=self.irreps_intermediate_edge,
                irreps_in2=self.irreps_intermediate_edge,
                irreps_out=self.irreps_tbs[idx],
            )
            self.sc_tbs.append(sc_tb)

    def _tb_by_type(self, edge_fea, edge_type):
        tb_list = []
        for idx, sc_tb in enumerate(self.sc_tbs):
            mask = edge_type == idx
            if int(mask.sum()) == 0:
                tb_list.append(edge_fea.new_zeros((0, sc_tb.irreps_out.dim)))
            else:
                tb = sc_tb(edge_fea[mask], edge_fea[mask])
                tb_list.append(tb)
        return tb_list

    def forward(self, data, flag):
        x = F.relu(self.em(data['x']))
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        pos = data['pos']
        edge_shift_vec = data['edge_shift_vec']

        edge_vec = pos[edge_dst] - pos[edge_src] + edge_shift_vec
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='gaussian',
            cutoff=False,
        ).mul(self.number_of_basis ** 0.5)

        node_fea = x
        edge_fea = edge_length_embedded

        if flag in ['TRAIN', 'TB', 'IP']:
            for node_update, edge_update in zip(self.node_updates, self.edge_updates):
                node_fea = node_update(node_fea, edge_src, edge_dst, edge_fea, edge_sh, edge_length_embedded)
                edge_fea = edge_update(node_fea, edge_src, edge_dst, edge_fea, edge_sh, edge_length_embedded)

        if flag == 'TB':
            return self._tb_by_type(edge_fea, data.edge_type)

        node_fea = self.lin_E(node_fea)
        energy = self.sc_E(node_fea, node_fea)

        if flag == 'IP':
            forces = -torch.autograd.grad(
                outputs=energy.sum(),
                inputs=data['pos'],
                grad_outputs=torch.ones_like(energy.sum()),
                create_graph=False,
                allow_unused=False,
            )[0]
            return energy, forces

        tb_list = self._tb_by_type(edge_fea, data.edge_type)
        forces = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=data['pos'],
            grad_outputs=torch.ones_like(energy.sum()),
            create_graph=True,
            allow_unused=False,
        )[0]
        return energy, forces, tb_list


def _compute_tb_loss(tb_data_pred_list, tb_data_ref, edge_type, loss_fn, loss_fn_mae, device):
    if torch.is_tensor(edge_type):
        edge_type_list = edge_type.detach().cpu().tolist()
    else:
        edge_type_list = list(edge_type)

    if len(tb_data_pred_list) == 1:
        ref_list = [ref.reshape(-1).to(device).to(default_dtype) for ref in tb_data_ref]
        ref = torch.stack(ref_list, dim=0)
        pred = tb_data_pred_list[0]
        return loss_fn(pred, ref), loss_fn_mae(pred, ref)

    loss_tb_data = torch.tensor(0.0, dtype=default_dtype, device=device)
    loss_tb_data_mae = torch.tensor(0.0, dtype=default_dtype, device=device)
    n_types_with_data = 0

    for idx, pred in enumerate(tb_data_pred_list):
        ref_list = []
        for ref, idx_type in zip(tb_data_ref, edge_type_list):
            if idx == int(idx_type):
                ref_list.append(ref.reshape(-1).to(device).to(default_dtype))
        if len(ref_list) > 0:
            ref = torch.stack(ref_list, dim=0)
            loss_tb_data = loss_tb_data + loss_fn(ref, pred)
            loss_tb_data_mae = loss_tb_data_mae + loss_fn_mae(ref, pred)
            n_types_with_data += 1

    if n_types_with_data == 0:
        return torch.tensor(0.0, dtype=default_dtype, device=device), torch.tensor(0.0, dtype=default_dtype, device=device)

    loss_tb_data = loss_tb_data / n_types_with_data
    loss_tb_data_mae = loss_tb_data_mae / n_types_with_data
    return loss_tb_data, loss_tb_data_mae


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def evaluate(model, dataloader, loss_fn, loss_fn_mae, device, loss_weights):
    model.eval()

    total_loss_cumulative = 0.0
    total_loss_cumulative_mae = 0.0
    e_loss_cumulative_mae = 0.0
    f_loss_cumulative_mae = 0.0
    tb_loss_cumulative_mae = 0.0

    for d in dataloader:
        d.to(device)
        energy_pred, forces_pred, tb_data_pred_list = model(d, flag='TRAIN')
        num_atom = energy_pred.shape[0]
        energy_pred = energy_pred.sum()

        energy_ref = d.energy.to(default_dtype).sum()
        forces_ref = d.forces.to(default_dtype)

        loss_energy = loss_fn(energy_pred / num_atom, energy_ref / num_atom)
        loss_forces = loss_fn(forces_pred, forces_ref)
        loss_energy_mae = loss_fn_mae(energy_pred / num_atom, energy_ref / num_atom)
        loss_forces_mae = loss_fn_mae(forces_pred, forces_ref)

        loss_tb_data, loss_tb_data_mae = _compute_tb_loss(
            tb_data_pred_list=tb_data_pred_list,
            tb_data_ref=d.tb_data,
            edge_type=d.edge_type,
            loss_fn=loss_fn,
            loss_fn_mae=loss_fn_mae,
            device=device,
        )

        total_loss = (
            loss_weights['weight_energy'] * loss_energy
            + loss_weights['weight_forces'] * loss_forces
            + loss_weights['weight_tb'] * loss_tb_data
        )
        total_loss_mae = (
            loss_weights['weight_energy'] * loss_energy_mae
            + loss_weights['weight_forces'] * loss_forces_mae
            + loss_weights['weight_tb'] * loss_tb_data_mae
        )

        total_loss_cumulative += total_loss.detach().item()
        total_loss_cumulative_mae += total_loss_mae.detach().item()
        e_loss_cumulative_mae += loss_energy_mae.detach().item()
        f_loss_cumulative_mae += loss_forces_mae.detach().item()
        tb_loss_cumulative_mae += loss_tb_data_mae.detach().item()

    n_batch = len(dataloader)
    return (
        total_loss_cumulative / n_batch,
        total_loss_cumulative_mae / n_batch,
        e_loss_cumulative_mae / n_batch,
        f_loss_cumulative_mae / n_batch,
        tb_loss_cumulative_mae / n_batch,
    )


def train(
    model,
    optimizer,
    dataloader_train,
    dataloader_valid,
    loss_fn,
    loss_fn_mae,
    run_name,
    loss_weights,
    max_iter=101,
    scheduler=None,
    device='cpu',
    rms_f=1.0,
    rms_tb=1.0,
):
    model.to(device)

    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    try:
        state = torch.load(run_name + '.torch', map_location=device)
        model.load_state_dict(state['state'])
        history = state['history']
        s0 = history[-1]['step'] + 1
    except Exception:
        history = []
        s0 = 0

    for step in range(max_iter):
        model.train()
        total_loss = None
        total_loss_mae = None
        epoch_loss_cumulative = 0.0
        epoch_loss_count = 0

        for _, d in tqdm(enumerate(dataloader_train), total=len(dataloader_train), bar_format=bar_format):
            d.to(device)
            energy_pred, forces_pred, tb_data_pred_list = model(d, flag='TRAIN')
            num_atom = energy_pred.shape[0]
            energy_pred = energy_pred.sum()

            energy_ref = d.energy.to(default_dtype).sum()
            forces_ref = d.forces.to(default_dtype)

            loss_energy = loss_fn(energy_pred / num_atom, energy_ref / num_atom)
            loss_forces = loss_fn(forces_pred, forces_ref)
            loss_energy_mae = loss_fn_mae(energy_pred / num_atom, energy_ref / num_atom)
            loss_forces_mae = loss_fn_mae(forces_pred, forces_ref)

            loss_tb_data, loss_tb_data_mae = _compute_tb_loss(
                tb_data_pred_list=tb_data_pred_list,
                tb_data_ref=d.tb_data,
                edge_type=d.edge_type,
                loss_fn=loss_fn,
                loss_fn_mae=loss_fn_mae,
                device=device,
            )

            total_loss = (
                loss_weights['weight_energy'] * loss_energy
                + loss_weights['weight_forces'] * loss_forces
                + loss_weights['weight_tb'] * loss_tb_data
            )
            total_loss_mae = (
                loss_weights['weight_energy'] * loss_energy_mae
                + loss_weights['weight_forces'] * loss_forces_mae
                + loss_weights['weight_tb'] * loss_tb_data_mae
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss_cumulative += total_loss.detach().item()
            epoch_loss_count += 1

        epoch_avg_loss = epoch_loss_cumulative / max(epoch_loss_count, 1)

        wall = time.time() - start_time
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, dataloader_valid, loss_fn, loss_fn_mae, device, loss_weights)
            train_avg_loss = evaluate(model, dataloader_train, loss_fn, loss_fn_mae, device, loss_weights)

            history.append(
                {
                    'step': s0 + step,
                    'wall': wall,
                    'batch': {
                        'loss': total_loss.item(),
                        'mean_abs': total_loss_mae.item(),
                    },
                    'valid': {
                        'loss': valid_avg_loss[0],
                        'mean_abs': valid_avg_loss[1],
                        'e_mae': valid_avg_loss[2] * rms_f,
                        'f_mae': valid_avg_loss[3] * rms_f,
                        'tb_mae': valid_avg_loss[4] * rms_tb,
                    },
                    'train': {
                        'loss': train_avg_loss[0],
                        'mean_abs': train_avg_loss[1],
                        'e_mae': train_avg_loss[2] * rms_f,
                        'f_mae': train_avg_loss[3] * rms_f,
                        'tb_mae': train_avg_loss[4] * rms_tb,
                    },
                }
            )

            results = {
                'history': history,
                'state': model.state_dict(),
            }

            print(
                f"Iteration {step + 1:4d}   "
                f"train loss = {train_avg_loss[1]:8.4f}   "
                f"valid loss = {valid_avg_loss[1]:8.4f}   "
                f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}"
            )
            print(
                f"TRAIN loss   "
                f"Energy loss = {train_avg_loss[2] * rms_f:9.5f} eV  "
                f"Force loss = {train_avg_loss[3] * rms_f:9.5f} eV/Angstrom  "
                f"TB loss = {train_avg_loss[4] * rms_tb:9.5f} eV  "
            )
            print(
                f"VALID loss  "
                f"Energy loss = {valid_avg_loss[2] * rms_f:9.5f} eV  "
                f"Force loss = {valid_avg_loss[3] * rms_f:9.5f} eV/Angstrom  "
                f"TB loss = {valid_avg_loss[4] * rms_tb:9.5f} eV  "
            )

            with open(run_name + '.torch', 'wb') as f:
                torch.save(results, f)

        if scheduler is not None:
            prev_lrs = [group['lr'] for group in optimizer.param_groups]
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_avg_loss)
            else:
                scheduler.step()
            new_lrs = [group['lr'] for group in optimizer.param_groups]
            if any(not math.isclose(prev_lr, new_lr, rel_tol=0.0, abs_tol=1e-15) for prev_lr, new_lr in zip(prev_lrs, new_lrs)):
                lr_updates = ', '.join(
                    f'{prev_lr:.6e} -> {new_lr:.6e}'
                    for prev_lr, new_lr in zip(prev_lrs, new_lrs)
                )
                print(f'Learning rate changed at iteration {step + 1:4d}: {lr_updates}')
