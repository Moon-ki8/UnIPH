import torch
from e3nn import o3
from e3nn.o3 import Irreps, wigner_3j


def flt2cplx(dtype):
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    return dtype


def irreps_from_l1l2(l1, l2, mul=1, spinful=False, no_parity=False):
    irreps_out = []
    for l in range(abs(l1 - l2), l1 + l2 + 1):
        irreps_out.append((mul, (l, 1 if l % 2 == 1 and not no_parity else 1)))
    irreps_out = Irreps(irreps_out)

    if spinful:
        irreps_x1 = []
        for _, ir in irreps_out:
            irreps_l = []
            for l in range(abs(ir.l - 1), ir.l + 2):
                if l >= 0:
                    irreps_l.append((mul, (l, ir.p)))
            irreps_x1.append(Irreps(irreps_l))
        return irreps_out, irreps_out, irreps_x1

    return irreps_out, irreps_out, None


def sort_irreps(irreps):
    irreps_in = irreps
    irreps_out = irreps.sort()

    inds = torch.zeros(irreps_in.dim, dtype=torch.long)
    i_in = 0
    i_out = 0
    for _, ir_in in irreps_in:
        for _, ir_out in irreps_out:
            if ir_in == ir_out:
                dim = ir_in.dim
                inds[i_out:i_out + dim] = torch.arange(i_in, i_in + dim)
                i_in += dim
                i_out += dim
                break
            i_out += ir_out.dim
        i_out = 0

    class Sort:
        def __init__(self, indices, irreps_sorted):
            self.inds = indices
            self.irreps_out = irreps_sorted

        def __call__(self, x):
            return x[..., self.inds]

        def inverse(self, x):
            y = torch.zeros_like(x)
            y[..., self.inds] = x
            return y

    return Sort(inds, irreps_out)


class E3TensorDecomp:
    def __init__(
        self,
        net_irreps_out,
        out_js_list,
        default_dtype_torch=torch.float32,
        spinful=False,
        no_parity=False,
        if_sort=False,
        device_torch='cpu',
    ):
        if spinful:
            default_dtype_torch = flt2cplx(default_dtype_torch)
        self.dtype = default_dtype_torch
        self.spinful = spinful
        self.device = device_torch
        self.out_js_list = out_js_list
        if net_irreps_out is not None:
            net_irreps_out = Irreps(net_irreps_out)

        required_irreps_out = Irreps(None)
        in_slices = [0]
        wms = []
        h_slices = [0]
        wms_h = []

        if spinful:
            in_slices_sp = []
            h_slices_sp = []
            wms_sp = []
            wms_sp_h = []

        for h_l1, h_l2 in out_js_list:
            mul = 1
            _, required_irreps_out_single, required_irreps_x1 = irreps_from_l1l2(
                h_l1, h_l2, mul, spinful, no_parity=no_parity
            )
            required_irreps_out += required_irreps_out_single

            if spinful:
                in_slice_sp = [0, required_irreps_out_single.dim]
                h_slice_sp = [0]
                wm_sp = [None]
                wm_sp_h = []

                for (_, ir), ir_times_1 in zip(required_irreps_out_single, required_irreps_x1):
                    required_irreps_out += ir_times_1
                    in_slice_sp.append(in_slice_sp[-1] + ir_times_1.dim)
                    h_slice_sp.append(h_slice_sp[-1] + ir.dim)

                    wm_irx1 = []
                    wm_irx1_h = []
                    for _ in ir_times_1:
                        wm_irx1.append(
                            wigner_3j(ir.l, 1, _[1].l, dtype=default_dtype_torch, device=device_torch)
                        )
                        wm_irx1_h.append(
                            wigner_3j(_[1].l, ir.l, 1, dtype=default_dtype_torch, device=device_torch)
                            * (2 * _[1].l + 1)
                        )

                    wm_irx1 = torch.cat(wm_irx1, dim=-1)
                    wm_sp.append(wm_irx1)
                    wm_irx1_h = torch.cat(wm_irx1_h, dim=0)
                    wm_sp_h.append(wm_irx1_h)

            in_slices.append(required_irreps_out.dim)
            h_slices.append(h_slices[-1] + (2 * h_l1 + 1) * (2 * h_l2 + 1))

            if spinful:
                in_slices_sp.append(in_slice_sp)
                h_slices_sp.append(h_slice_sp)

            wm = []
            wm_h = []
            for _, ir in required_irreps_out_single:
                wm.append(wigner_3j(h_l1, h_l2, ir.l, dtype=default_dtype_torch, device=device_torch))
                wm_h.append(
                    wigner_3j(ir.l, h_l1, h_l2, dtype=default_dtype_torch, device=device_torch)
                    * (2 * ir.l + 1)
                )

            wm = torch.cat(wm, dim=-1)
            wm_h = torch.cat(wm_h, dim=0)
            wms.append(wm)
            wms_h.append(wm_h)

            if spinful:
                wms_sp.append(wm_sp)
                wms_sp_h.append(wm_sp_h)

        if spinful:
            required_irreps_out = required_irreps_out + required_irreps_out

        if net_irreps_out is not None:
            if if_sort:
                assert net_irreps_out == required_irreps_out.sort().irreps.simplify(), (
                    f'requires {required_irreps_out.sort().irreps.simplify()} but got {net_irreps_out}'
                )
            else:
                assert net_irreps_out == required_irreps_out, (
                    f'requires {required_irreps_out} but got {net_irreps_out}'
                )

        self.in_slices = in_slices
        self.wms = wms
        self.h_slices = h_slices
        self.wms_h = wms_h

        if spinful:
            self.in_slices_sp = in_slices_sp
            self.h_slices_sp = h_slices_sp
            self.wms_sp = wms_sp
            self.wms_sp_h = wms_sp_h

        if spinful:
            sqrt2 = 1.4142135623730951
            self.oyzx2spin = torch.tensor(
                [[1, 0, 1, 0], [0, -1j, 0, 1], [0, 1j, 0, 1], [1, 0, -1, 0]],
                dtype=default_dtype_torch,
                device=device_torch,
            ) / sqrt2

        self.sort = sort_irreps(required_irreps_out) if if_sort else None
        self.required_irreps_out = self.sort.irreps_out if self.sort is not None else required_irreps_out

    def get_h(self, net_out):
        if self.sort is not None:
            net_out = self.sort.inverse(net_out)

        if self.spinful:
            half_len = int(net_out.shape[-1] // 2)
            re = net_out[..., :half_len]
            im = net_out[..., half_len:]
            net_out = re + 1j * im

        out = []
        for idx in range(len(self.out_js_list)):
            in_slice = slice(self.in_slices[idx], self.in_slices[idx + 1])
            net_out_block = net_out[..., in_slice]

            if self.spinful:
                h_block = []
                for jdx in range(len(self.wms_sp[idx])):
                    in_slice_sp = slice(self.in_slices_sp[idx][jdx], self.in_slices_sp[idx][jdx + 1])
                    if jdx == 0:
                        h_block.append(net_out_block[..., in_slice_sp].unsqueeze(-1))
                    else:
                        h_block.append(torch.einsum('jkl,il->ijk', self.wms_sp[idx][jdx], net_out_block[..., in_slice_sp]))

                h_block = torch.cat([h_block[0], torch.cat(h_block[1:], dim=-2)], dim=-1)
                h_block = torch.einsum('imn,klm,jn->ijkl', h_block, self.wms[idx], self.oyzx2spin)
                out.append(h_block.reshape(net_out.shape[0], 4, -1))
            else:
                h_block = torch.sum(self.wms[idx][None, :, :, :] * net_out_block[..., None, None, :], dim=-1)
                out.append(h_block.reshape(net_out.shape[0], -1))

        return torch.cat(out, dim=-1)

    def get_net_out(self, h):
        out = []
        for idx in range(len(self.out_js_list)):
            h_slice = slice(self.h_slices[idx], self.h_slices[idx + 1])
            l1, l2 = self.out_js_list[idx]

            if self.spinful:
                h_block = h[..., h_slice].reshape(-1, 4, 2 * l1 + 1, 2 * l2 + 1)
                h_block = torch.einsum('ilmn,jmn,kl->ijk', h_block, self.wms_h[idx], self.oyzx2spin.T.conj())
                net_out_block = [h_block[..., :, 0]]
                for jdx in range(len(self.wms_sp_h[idx])):
                    h_slice_sp = slice(self.h_slices_sp[idx][jdx], self.h_slices_sp[idx][jdx + 1])
                    net_out_block.append(
                        torch.einsum('jlm,ilm->ij', self.wms_sp_h[idx][jdx], h_block[..., h_slice_sp, 1:])
                    )
                net_out_block = torch.cat(net_out_block, dim=-1)
                out.append(net_out_block)
            else:
                h_block = h[..., h_slice].reshape(-1, 2 * l1 + 1, 2 * l2 + 1)
                net_out_block = torch.sum(self.wms_h[idx][None, :, :, :] * h_block[..., None, :, :], dim=(-1, -2))
                out.append(net_out_block)

        out = torch.cat(out, dim=-1)

        if self.spinful:
            out = torch.cat([out.real, out.imag], dim=-1)

        if self.sort is not None:
            out = self.sort(out)

        return out


# Backward-compatible alias used in legacy code.
E3TensorDecomp.get_H = E3TensorDecomp.get_h


def wiki2e3nn(wiki_1d, converter, device='cpu'):
    irreps_out = Irreps(converter.required_irreps_out)
    change_of_coord = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    wiki_1d = wiki_1d.squeeze()
    if wiki_1d.ndim == 0:
        wiki_1d = wiki_1d.unsqueeze(0)
    dim0 = 0
    e3nn_1d = []
    for irrep in irreps_out:
        dim1 = irrep.dim + dim0
        temp_irrep = wiki_1d[dim0:dim1]
        d_mat = o3.Irrep(irrep.ir).D_from_matrix(change_of_coord).to(device)
        temp_irreps = d_mat @ temp_irrep.to(device)
        dim0 = dim1
        e3nn_1d.append(temp_irreps)
    return e3nn_1d


def e3nn2wiki(e3nn_1d, converter, device='cpu'):
    irreps_out = Irreps(converter.required_irreps_out)
    change_of_coord = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    e3nn_1d = e3nn_1d.squeeze()
    if e3nn_1d.ndim == 0:
        e3nn_1d = e3nn_1d.unsqueeze(0)
    dim0 = 0
    wiki_1d = []
    for irrep in irreps_out:
        dim1 = irrep.dim + dim0
        temp_irrep = e3nn_1d[dim0:dim1]
        d_mat = o3.Irrep(irrep.ir).D_from_matrix(change_of_coord).to(device)
        temp_irreps = d_mat.inverse() @ temp_irrep.to(device)
        dim0 = dim1
        wiki_1d.append(temp_irreps)
    return torch.cat(wiki_1d)


def run_irreps_parity_checks(orbital_edges, spin_flag=False, dtype=torch.float64, device='cpu', atol=1e-8, rtol=1e-6):
    if spin_flag:
        raise ValueError('PLAN_01 requires spin_flag=False.')

    for values in orbital_edges:
        left_ls = values[2]
        right_ls = values[3]
        out_js_list = [(l1, l2) for l1 in left_ls for l2 in right_ls]
        converter = E3TensorDecomp(
            None,
            out_js_list,
            default_dtype_torch=dtype,
            spinful=False,
            device_torch=device,
        )

        total_h_dim = 0
        for l1, l2 in out_js_list:
            total_h_dim += (2 * l1 + 1) * (2 * l2 + 1)

        h_input = torch.randn(1, total_h_dim, dtype=dtype, device=device)
        net_out = converter.get_net_out(h_input)
        h_roundtrip = converter.get_h(net_out)
        if not torch.allclose(h_input, h_roundtrip, atol=atol, rtol=rtol):
            raise ValueError(f'H->net_out->H round-trip failed for orbital pair {values[0]}-{values[1]}')

        wiki_random = torch.randn(converter.required_irreps_out.dim, dtype=dtype, device=device)
        e3nn_parts = wiki2e3nn(wiki_random, converter, device=device)
        wiki_roundtrip = e3nn2wiki(torch.cat(e3nn_parts), converter, device=device)
        if not torch.allclose(wiki_random, wiki_roundtrip, atol=atol, rtol=rtol):
            raise ValueError(f'wiki->e3nn->wiki round-trip failed for orbital pair {values[0]}-{values[1]}')

        if net_out.shape[-1] != converter.required_irreps_out.dim:
            raise ValueError(f'Irreps dimension mismatch for orbital pair {values[0]}-{values[1]}')

    return True
