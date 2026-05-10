import ast
import numpy as np
import pandas as pd
import torch
import torch_geometric as tg

from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from tqdm import tqdm

from utils.utils_irreps import E3TensorDecomp, wiki2e3nn, e3nn2wiki


tqdm.pandas()
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


def _get_converter_device(converter):
    if hasattr(converter, 'device'):
        return converter.device
    if hasattr(converter, 'wms') and len(converter.wms) > 0 and torch.is_tensor(converter.wms[0]):
        return converter.wms[0].device
    return 'cpu'


def build_atom_encoders(dtype=default_dtype):
    type_encoding = {}
    specie_am = []
    for z_num in range(1, 119):
        specie = Atom(z_num)
        type_encoding[specie.symbol] = z_num - 1
        specie_am.append(specie.mass)
    type_onehot = torch.eye(len(type_encoding), dtype=dtype)
    am_onehot = torch.diag(torch.tensor(specie_am, dtype=dtype))
    return type_encoding, type_onehot, am_onehot


def _parse_tb_data(tb_value):
    raw = ast.literal_eval(tb_value) if isinstance(tb_value, str) else tb_value
    parsed = []
    for hop in raw:
        hop_list = list(hop)
        if len(hop_list) < 6:
            raise ValueError(f'Invalid TB record length: {len(hop_list)}')

        rx = int(float(hop_list[0]))
        ry = int(float(hop_list[1]))
        rz = int(float(hop_list[2]))
        atom_i = int(float(hop_list[3]))
        atom_j = int(float(hop_list[4]))
        hop_payload = hop_list[5]

        if isinstance(hop_payload, (int, float, np.floating, np.integer)):
            hop_payload = float(hop_payload)
        else:
            hop_payload = np.array(hop_payload, dtype=np.float64).tolist()

        if len(hop_list) >= 7:
            parsed.append([rx, ry, rz, atom_i, atom_j, hop_payload, float(hop_list[6])])
        else:
            parsed.append([rx, ry, rz, atom_i, atom_j, hop_payload])
    return parsed


def load_data(filename):
    temp_data = pd.read_csv(filename)

    temp_data['structure'] = temp_data['structure'].apply(ast.literal_eval).progress_map(lambda x: Atoms.fromdict(x))
    temp_data['formula'] = temp_data['structure'].map(lambda x: x.get_chemical_formula())
    temp_data['species'] = temp_data['structure'].map(lambda x: list(set(x.get_chemical_symbols())))

    temp_data['tb_data'] = temp_data['tb_data'].apply(_parse_tb_data)
    temp_data['forces'] = temp_data['forces'].apply(ast.literal_eval).apply(np.array)

    return temp_data


def build_orbital_edges_and_codes(orbital_types):
    orbital_edges = []
    for left_symbol in orbital_types:
        for right_symbol in orbital_types:
            orbital_edges.append([
                left_symbol,
                right_symbol,
                orbital_types[left_symbol],
                orbital_types[right_symbol],
            ])

    pair_to_code = {
        (left_symbol, right_symbol): idx
        for idx, (left_symbol, right_symbol, _, _) in enumerate(orbital_edges)
    }
    return orbital_edges, pair_to_code


def build_converter_list(orbital_edges, spin_flag=False, dtype=default_dtype, device='cpu'):
    converter_list = []
    for values in orbital_edges:
        orbital_types_left = values[2]
        orbital_types_right = values[3]
        out_js_list = [(l1, l2) for l1 in orbital_types_left for l2 in orbital_types_right]
        converter = E3TensorDecomp(
            None,
            out_js_list,
            default_dtype_torch=dtype,
            spinful=spin_flag,
            device_torch=device,
        )
        converter_list.append(converter)
    return converter_list


def _match_tb_records_to_edges(tb_data, edge_src, edge_dst, edge_shift):
    tb_data_ordered = []
    for edge_i, edge_j, edge_s in zip(edge_src, edge_dst, edge_shift):
        found = False
        for tb_temp in tb_data:
            i_mx = int(tb_temp[3])
            j_mx = int(tb_temp[4])
            s_mx = np.array(tb_temp[0:3], dtype=int)
            if edge_i == i_mx and edge_j == j_mx and np.array_equal(edge_s, s_mx):
                tb_data_ordered.append(tb_temp[5])
                found = True
                break
        if not found:
            raise ValueError(f'Pair not found: i={edge_i}, j={edge_j}, s={edge_s}')

    return tb_data_ordered


def h_tb_to_e3nn(temp_data, edge_src, edge_dst, edge_shift, edge_type, converter_list, device='cpu'):
    tb_data = temp_data.tb_data
    tb_data_ordered = _match_tb_records_to_edges(tb_data, edge_src, edge_dst, edge_shift)

    tb_1d_list = []
    for tb_raw, edge_t in zip(tb_data_ordered, edge_type):
        converter = converter_list[int(edge_t)]
        converter_device = _get_converter_device(converter)

        if isinstance(tb_raw, (int, float, np.floating, np.integer)):
            tb_tensor = torch.tensor([[float(tb_raw)]], dtype=default_dtype, device=converter_device)
        else:
            tb_tensor = torch.tensor(np.array(tb_raw, dtype=np.float64), dtype=default_dtype, device=converter_device).reshape(1, -1)

        tb_wiki_1d = converter.get_net_out(tb_tensor)
        tb_e3nn_1d = wiki2e3nn(tb_wiki_1d.squeeze(), converter, device=converter_device)
        tb_1d_list.append(torch.cat(tb_e3nn_1d).to('cpu'))

    return tb_1d_list


def graph_build(
    temp_data,
    pair_to_code,
    converter_list,
    type_encoding,
    am_onehot,
    r_max,
):
    symbols = list(temp_data.structure.symbols).copy()
    edge_src, edge_dst, edge_shift = neighbor_list(
        'ijS', a=temp_data.structure, cutoff=r_max, self_interaction=True
    )

    edge_type = [pair_to_code[(symbols[i], symbols[j])] for i, j in zip(edge_src, edge_dst)]
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    positions = torch.from_numpy(temp_data.structure.positions.copy()).to(default_dtype)
    lattice = torch.from_numpy(temp_data.structure.cell.array.copy()).unsqueeze(0).to(default_dtype)
    energy = temp_data.energy
    forces = temp_data.forces

    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_shift_vec = torch.einsum(
        'ni,nij->nj',
        torch.tensor(edge_shift, dtype=default_dtype),
        lattice[edge_batch],
    )

    tb_1d_list = h_tb_to_e3nn(
        temp_data=temp_data,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_shift=edge_shift,
        edge_type=edge_type,
        converter_list=converter_list,
        device='cpu',
    )

    data_graph = tg.data.Data(
        x=am_onehot[[type_encoding[specie] for specie in symbols]],
        pos=positions,
        lattice=lattice,
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_shift_vec=edge_shift_vec.to(default_dtype),
        forces=torch.from_numpy(forces),
        energy=torch.tensor([energy], dtype=default_dtype),
        tb_data=tb_1d_list,
        edge_type=edge_type,
    )
    data_graph.pos.requires_grad_(True)
    data_graph.energy.requires_grad_(True)
    return data_graph


def graph_build_single(temp_data, pair_to_code, r_max, type_encoding, am_onehot):
    symbols = list(temp_data.symbols).copy()
    edge_src, edge_dst, edge_shift = neighbor_list(
        'ijS', a=temp_data, cutoff=r_max, self_interaction=True
    )

    edge_type = [pair_to_code[(symbols[i], symbols[j])] for i, j in zip(edge_src, edge_dst)]
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    positions = torch.from_numpy(temp_data.positions.copy()).to(default_dtype)
    lattice = torch.from_numpy(temp_data.cell.array.copy()).unsqueeze(0).to(default_dtype)

    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_shift_vec = torch.einsum(
        'ni,nij->nj',
        torch.tensor(edge_shift, dtype=default_dtype),
        lattice[edge_batch],
    )

    data_graph = tg.data.Data(
        x=am_onehot[[type_encoding[specie] for specie in symbols]],
        pos=positions,
        lattice=lattice,
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_shift_vec=edge_shift_vec,
        edge_type=edge_type,
    )
    data_graph.pos.requires_grad_(True)
    return data_graph


def extract_edgewise_tb_from_type_outputs(tb_type_outputs, edge_type):
    edgewise_outputs = []
    if len(tb_type_outputs) == 0:
        return edgewise_outputs

    if torch.is_tensor(edge_type):
        edge_type_list = edge_type.detach().cpu().tolist()
    else:
        edge_type_list = list(edge_type)

    counters = [0 for _ in range(len(tb_type_outputs))]
    for edge_t in edge_type_list:
        idx_local = counters[int(edge_t)]
        edgewise_outputs.append(tb_type_outputs[int(edge_t)][idx_local])
        counters[int(edge_t)] += 1

    return edgewise_outputs


def h_e3nn_to_tb(single_graph, tb_data_edgewise, converter_list, device='cpu'):
    edge_type = single_graph.edge_type
    if torch.is_tensor(edge_type):
        edge_type_list = edge_type.detach().cpu().tolist()
    else:
        edge_type_list = list(edge_type)

    wiki_1d_all = []
    for edge_t, tb_data_e3nn in zip(edge_type_list, tb_data_edgewise):
        converter = converter_list[int(edge_t)]
        converter_device = _get_converter_device(converter)
        wiki_1d = e3nn2wiki(tb_data_e3nn, converter, device=converter_device)
        wiki_1d_all.append(wiki_1d)

    h_1d_w90 = []
    for edge_t, wiki_1d in zip(edge_type_list, wiki_1d_all):
        converter = converter_list[int(edge_t)]
        converter_device = _get_converter_device(converter)
        h_1d = converter.get_h(wiki_1d.unsqueeze(0).to(converter_device))
        h_1d_w90.append(h_1d.squeeze(0))

    return h_1d_w90
