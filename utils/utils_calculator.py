from pathlib import Path
import sys

import torch
from ase.calculators.calculator import Calculator, all_changes


UNIPH_ROOT = Path(__file__).resolve().parents[1]
if str(UNIPH_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIPH_ROOT))

from utils.utils_config import load_and_validate_config
from utils.utils_data import (
    build_atom_encoders,
    build_converter_list,
    build_orbital_edges_and_codes,
    extract_edgewise_tb_from_type_outputs,
    graph_build_single,
    h_e3nn_to_tb,
)
from utils.utils_model import e3net


torch.set_default_dtype(torch.float64)


def _select_device(device=None):
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _normalize_legacy_state_dict(state):
    if "sc_TB.weight" not in state or "sc_tbs.0.weight" in state:
        return state

    state = dict(state)
    state["sc_tbs.0.weight"] = state.pop("sc_TB.weight")
    state["sc_tbs.0.output_mask"] = state.pop("sc_TB.output_mask")
    return state


def load_uniph_model(config_path, checkpoint_path=None, device=None):
    device = _select_device(device)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = UNIPH_ROOT / config_path

    cfg = load_and_validate_config(config_path, require_data_files=False)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    type_encoding, _, am_onehot = build_atom_encoders(dtype=torch.float64)
    orbital_edges, pair_to_code = build_orbital_edges_and_codes(model_cfg["orbital_types"])
    converter_list = build_converter_list(
        orbital_edges=orbital_edges,
        spin_flag=model_cfg["spin_flag"],
        dtype=torch.float64,
        device=device,
    )

    model = e3net(
        in_dim=model_cfg["in_dim"],
        irreps_intermediate_node=model_cfg["inter_irreps_node"],
        irreps_intermediate_edge=model_cfg["inter_irreps_edge"],
        orbital_edges=orbital_edges,
        irreps_E=model_cfg["irreps_E"],
        lmax=model_cfg["lmax"],
        num_layer=model_cfg["num_layer"],
        max_radius=model_cfg["max_radius"],
        fcn_len=model_cfg["fcn_len"],
        number_of_basis=model_cfg["number_of_basis"],
        spin_flag=model_cfg["spin_flag"],
    ).to(device)

    if checkpoint_path is None:
        checkpoint_path = UNIPH_ROOT / f"{model_cfg['gnn_model_name']}.torch"
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = UNIPH_ROOT / checkpoint_path

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(_normalize_legacy_state_dict(checkpoint["state"]))
    model.eval()

    return {
        "device": device,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "model": model,
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
        "type_encoding": type_encoding,
        "am_onehot": am_onehot,
        "orbital_edges": orbital_edges,
        "pair_to_code": pair_to_code,
        "converter_list": converter_list,
    }


def load_graphene_model(device=None, checkpoint_path=None):
    return load_uniph_model(
        config_path="gnn_config_graphene.ini",
        checkpoint_path=checkpoint_path,
        device=device,
    )


def load_mos2_model(device=None, checkpoint_path=None):
    return load_uniph_model(
        config_path="gnn_config_mos2.ini",
        checkpoint_path=checkpoint_path,
        device=device,
    )


def build_graph(atoms, context):
    graph = graph_build_single(
        temp_data=atoms,
        pair_to_code=context["pair_to_code"],
        r_max=context["model_cfg"]["max_radius"],
        type_encoding=context["type_encoding"],
        am_onehot=context["am_onehot"],
    )
    return graph.to(context["device"])


def predict_ip(atoms, context):
    graph = build_graph(atoms, context)
    energy_norm, forces_norm = context["model"](graph, flag="IP")
    energy_per_atom = energy_norm.reshape(-1) * context["data_cfg"]["rms_f"] + context["data_cfg"]["mean_energy"]
    forces = forces_norm * context["data_cfg"]["rms_f"]
    total_energy = float(energy_per_atom.sum().detach().cpu().item())
    return total_energy, energy_per_atom.detach().cpu().numpy(), forces.detach().cpu().numpy()


def predict_tb(atoms, context):
    graph = build_graph(atoms, context)
    with torch.no_grad():
        tb_by_type = context["model"](graph, flag="TB")
    tb_edgewise = extract_edgewise_tb_from_type_outputs(tb_by_type, graph.edge_type.detach().cpu())
    hopping_mats = h_e3nn_to_tb(
        single_graph=graph,
        tb_data_edgewise=tb_edgewise,
        converter_list=context["converter_list"],
        device=context["device"],
    )
    hopping_mats = [
        (hop * context["data_cfg"]["rms_tb"]).detach().cpu().numpy()
        for hop in hopping_mats
    ]
    return graph, hopping_mats


class UnIPHCalculator(Calculator):
    implemented_properties = ["energy", "energies", "forces"]

    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self.context = context

    def calculate(self, atoms, properties=("energy", "energies", "forces"), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        total_energy, energies, forces = predict_ip(atoms, self.context)
        self.results["energy"] = total_energy
        self.results["energies"] = energies
        self.results["forces"] = forces
