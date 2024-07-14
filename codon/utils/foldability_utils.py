import esm
import numpy as np
import torch
from openfold.np.residue_constants import restypes
from tmtools import tm_align
from tqdm import tqdm


def run_foldability(atom37, aatype, device='cuda'):
    print('Loading ESMFold model for foldability evaluation')
    torch.cuda.empty_cache()
    esmf_model = esm.pretrained.esmfold_v1().eval().to(device)
    results = {'tm_score': [], 'rmsd': []}

    for bb, aa in tqdm(zip(atom37, aatype)):
        seq = ''.join([restypes[i] for i in aa])
        seq = seq.replace('X', 'A')

        with torch.no_grad():
            output = esmf_model.infer(seq)

        out_ca_pos = output['positions'][-1].squeeze()[:,1].cpu().numpy()

        _, tm_score = get_tm_score(bb[..., 1, :], out_ca_pos, seq, seq)
        rmsd = get_aligned_rmsd(bb[..., 1, :], out_ca_pos)
        results['tm_score'].append(float(tm_score))
        results['rmsd'].append(float(rmsd))

    del esmf_model
    torch.cuda.empty_cache()
    return results

def get_aligned_rmsd(pos_1, pos_2):
    aligned_pos_1 = rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))

def get_tm_score(pos_1, pos_2, seq_1, seq_2):
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2

def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected