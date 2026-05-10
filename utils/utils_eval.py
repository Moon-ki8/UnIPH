import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18
ANNOTATION_FONTSIZE = 18


def plot_comparisons_2d(
    ax,
    pred_train,
    ref_train,
    pred_test,
    ref_test,
    title,
    xlabel='Reference',
    ylabel='Predicted',
    mae_annotation='',
    tick_format='%.3f',
):
    ax.scatter(ref_train, pred_train, marker='o', alpha=0.7, color='tab:blue', label='Training')
    ax.scatter(ref_test, pred_test, marker='o', alpha=0.7, color='tab:orange', label='Test')
    max_val = max(np.max(pred_train), np.max(ref_train), np.max(pred_test), np.max(ref_test))
    min_val = min(np.min(pred_train), np.min(ref_train), np.min(pred_test), np.min(ref_test))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--')
    margin = (max_val - min_val) * 0.1
    ax.set_xlim([min_val - margin, max_val + margin])
    ax.set_ylim([min_val - margin, max_val + margin])
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
    ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format))
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc='upper left')
    if mae_annotation:
        ax.text(
            0.98,
            0.02,
            mae_annotation,
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=ANNOTATION_FONTSIZE,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
        )


def evaluate_model_on_dataset(model, dataloader, device, converter_list):
    import torch

    from utils.utils_data import extract_edgewise_tb_from_type_outputs, h_e3nn_to_tb

    energy_preds = []
    energy_refs = []
    force_preds = []
    force_refs = []
    tb_data_preds = []
    tb_data_refs = []
    num_atoms = []

    model.eval()
    model.to(device)

    for batch in dataloader:
        batch = batch.to(device)
        energy_pred, forces_pred, tb_data_pred_by_type = model(batch, flag='TRAIN')

        tb_data_pred_edgewise = extract_edgewise_tb_from_type_outputs(
            tb_data_pred_by_type,
            batch.edge_type.detach().cpu(),
        )

        ref_tb = h_e3nn_to_tb(
            single_graph=batch,
            tb_data_edgewise=batch.tb_data,
            converter_list=converter_list,
            device=device,
        )
        pred_tb = h_e3nn_to_tb(
            single_graph=batch,
            tb_data_edgewise=tb_data_pred_edgewise,
            converter_list=converter_list,
            device=device,
        )

        ref_tb_1d = torch.cat([ref.reshape(-1) for ref in ref_tb]).detach().cpu().numpy()
        pred_tb_1d = torch.cat([pred.reshape(-1) for pred in pred_tb]).detach().cpu().numpy()

        n_atoms = energy_pred.shape[0]
        energy_pred = energy_pred.sum()

        energy_ref = batch.energy.double().to(device)
        forces_ref = batch.forces.double().to(device)

        energy_preds.append(energy_pred.detach().cpu().numpy())
        energy_refs.append(energy_ref.detach().cpu().numpy())
        force_preds.append(forces_pred.detach().cpu().numpy())
        force_refs.append(forces_ref.detach().cpu().numpy())
        tb_data_preds.append(pred_tb_1d)
        tb_data_refs.append(ref_tb_1d)
        num_atoms.append(n_atoms)

    return energy_preds, energy_refs, force_preds, force_refs, tb_data_preds, tb_data_refs, num_atoms


def _convert_outputs_to_plot_arrays(
    energy_preds,
    energy_refs,
    force_preds,
    force_refs,
    tb_data_preds,
    tb_data_refs,
    num_atoms,
    mean_energy,
    rms_force,
    rms_tb_data,
):
    energy_preds = np.array(energy_preds).flatten()
    energy_refs = np.array(energy_refs).flatten()

    force_preds_mag = np.concatenate([np.linalg.norm(pred, axis=1) for pred in force_preds])
    force_refs_mag = np.concatenate([np.linalg.norm(ref, axis=1) for ref in force_refs])

    tb_data_preds_flat = np.concatenate(tb_data_preds)
    tb_data_refs_flat = np.concatenate(tb_data_refs)

    num_atoms = np.array(num_atoms)
    energy_pred_total = energy_preds * rms_force + mean_energy * num_atoms
    energy_ref_total = energy_refs * rms_force + mean_energy * num_atoms
    force_pred_plot = force_preds_mag * rms_force
    force_ref_plot = force_refs_mag * rms_force
    tb_pred_plot = tb_data_preds_flat * rms_tb_data
    tb_ref_plot = tb_data_refs_flat * rms_tb_data

    energy_pred = energy_pred_total / num_atoms
    energy_ref = energy_ref_total / num_atoms

    return {
        'energy_pred': energy_pred,
        'energy_ref': energy_ref,
        'force_pred': force_pred_plot,
        'force_ref': force_ref_plot,
        'tb_pred': tb_pred_plot,
        'tb_ref': tb_ref_plot,
        'num_atoms': num_atoms,
    }


def evaluate_datasets(
    gnn_model,
    dataloader_train,
    dataloader_test,
    device,
    converter_list,
    mean_energy=0.0,
    rms_force=1.0,
    rms_tb_data=1.0,
):
    train_outputs = evaluate_model_on_dataset(
        gnn_model,
        dataloader_train,
        device,
        converter_list,
    )
    test_outputs = evaluate_model_on_dataset(
        gnn_model,
        dataloader_test,
        device,
        converter_list,
    )

    train_results = _convert_outputs_to_plot_arrays(
        *train_outputs,
        mean_energy=mean_energy,
        rms_force=rms_force,
        rms_tb_data=rms_tb_data,
    )
    test_results = _convert_outputs_to_plot_arrays(
        *test_outputs,
        mean_energy=mean_energy,
        rms_force=rms_force,
        rms_tb_data=rms_tb_data,
    )

    return train_results, test_results


def compute_mae_summary(train_results, test_results):
    return {
        'train': {
            'energy': np.mean(np.abs(train_results['energy_pred'] - train_results['energy_ref'])),
            'force': np.mean(np.abs(train_results['force_pred'] - train_results['force_ref'])),
            'tb': np.mean(np.abs(train_results['tb_pred'] - train_results['tb_ref'])),
        },
        'test': {
            'energy': np.mean(np.abs(test_results['energy_pred'] - test_results['energy_ref'])),
            'force': np.mean(np.abs(test_results['force_pred'] - test_results['force_ref'])),
            'tb': np.mean(np.abs(test_results['tb_pred'] - test_results['tb_ref'])),
        },
    }


def save_eval_results_npz(output_path, train_results, test_results):
    np.savez(
        output_path,
        train_energy_pred=train_results['energy_pred'],
        train_energy_ref=train_results['energy_ref'],
        train_force_pred=train_results['force_pred'],
        train_force_ref=train_results['force_ref'],
        train_tb_pred=train_results['tb_pred'],
        train_tb_ref=train_results['tb_ref'],
        train_num_atoms=train_results['num_atoms'],
        test_energy_pred=test_results['energy_pred'],
        test_energy_ref=test_results['energy_ref'],
        test_force_pred=test_results['force_pred'],
        test_force_ref=test_results['force_ref'],
        test_tb_pred=test_results['tb_pred'],
        test_tb_ref=test_results['tb_ref'],
        test_num_atoms=test_results['num_atoms'],
    )


def load_eval_results_npz(npz_path):
    arrays = np.load(npz_path)
    train_results = {
        'energy_pred': arrays['train_energy_pred'],
        'energy_ref': arrays['train_energy_ref'],
        'force_pred': arrays['train_force_pred'],
        'force_ref': arrays['train_force_ref'],
        'tb_pred': arrays['train_tb_pred'],
        'tb_ref': arrays['train_tb_ref'],
        'num_atoms': arrays['train_num_atoms'],
    }
    test_results = {
        'energy_pred': arrays['test_energy_pred'],
        'energy_ref': arrays['test_energy_ref'],
        'force_pred': arrays['test_force_pred'],
        'force_ref': arrays['test_force_ref'],
        'tb_pred': arrays['test_tb_pred'],
        'tb_ref': arrays['test_tb_ref'],
        'num_atoms': arrays['test_num_atoms'],
    }
    return train_results, test_results


def plot_eval_results(train_results, test_results, output_path='comparison.png'):
    test_mae_energy = np.mean(np.abs(test_results['energy_pred'] - test_results['energy_ref']))
    test_mae_force = np.mean(np.abs(test_results['force_pred'] - test_results['force_ref']))
    test_mae_tb = np.mean(np.abs(test_results['tb_pred'] - test_results['tb_ref']))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_comparisons_2d(
        axes[0],
        train_results['energy_pred'],
        train_results['energy_ref'],
        test_results['energy_pred'],
        test_results['energy_ref'],
        title='Energy (eV/atom)',
        mae_annotation=f'Test MAE: {test_mae_energy * 1000:.2f} meV/atom',
        tick_format='%.2f',
    )
    plot_comparisons_2d(
        axes[1],
        train_results['force_pred'],
        train_results['force_ref'],
        test_results['force_pred'],
        test_results['force_ref'],
        title='Force Magnitude (eV/Angstrom)',
        mae_annotation=f'Test MAE: {test_mae_force * 1000:.2f} meV/Angstrom',
    )
    plot_comparisons_2d(
        axes[2],
        train_results['tb_pred'],
        train_results['tb_ref'],
        test_results['tb_pred'],
        test_results['tb_ref'],
        title='TB Parameters (eV)',
        mae_annotation=f'Test MAE: {test_mae_tb * 1000:.2f} meV',
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    return {
        'test': {
            'energy': test_mae_energy,
            'force': test_mae_force,
            'tb': test_mae_tb,
        },
    }


def plot_eval_results_from_npz(npz_path, output_path='comparison.png'):
    train_results, test_results = load_eval_results_npz(npz_path)
    return plot_eval_results(train_results, test_results, output_path=output_path)


def plotgraph(
    gnn_model,
    dataloader_train,
    dataloader_test,
    device,
    converter_list,
    mean_energy=0.0,
    rms_force=1.0,
    rms_tb_data=1.0,
):
    train_results, test_results = evaluate_datasets(
        gnn_model,
        dataloader_train,
        dataloader_test,
        device,
        converter_list,
        mean_energy=mean_energy,
        rms_force=rms_force,
        rms_tb_data=rms_tb_data,
    )
    save_eval_results_npz('eval_results.npz', train_results, test_results)
    mae_summary = plot_eval_results(train_results, test_results, output_path='comparison.png')

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

    return train_results, test_results
