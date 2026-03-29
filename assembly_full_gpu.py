"""
Assembly Index — FULL GPU Implementation
=========================================
Everything on GPU. No CPU graph search.

Strategy (counting revolution approach):
1. Molecule -> adjacency tensor + bond type tensor (GPU)
2. For each pair of bonds: check if same type -> duplicate counting (GPU tensor ops)
3. For each triple of bonds: check connectivity + canonical form (GPU)
4. Savings from duplicates -> MA estimate (GPU)

Key: represent fragment matching as MATRIX OPERATIONS.
Two bonds are "same type" iff their atom-pair signatures match.
This is a COMPARISON MATRIX computed in one tensor op.
"""
import torch
import numpy as np
import time
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smiles_to_tensors(smiles, max_atoms=40, max_bonds=60):
    """Convert SMILES to GPU tensors.
    Returns:
        atom_types: (max_atoms,) int tensor
        bond_matrix: (max_atoms, max_atoms) int tensor (bond type: 0=none, 1=single, 2=double, 3=triple)
        bond_endpoints: (max_bonds, 2) int tensor
        bond_types: (max_bonds,) int tensor
        n_atoms, n_bonds: actual counts
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    na = mol.GetNumAtoms()
    nb = mol.GetNumBonds()
    if na > max_atoms or nb > max_bonds or nb == 0:
        return None

    atom_types = torch.zeros(max_atoms, dtype=torch.long)
    bond_matrix = torch.zeros(max_atoms, max_atoms, dtype=torch.long)
    bond_endpoints = torch.zeros(max_bonds, 2, dtype=torch.long)
    bond_types_vec = torch.zeros(max_bonds, dtype=torch.long)

    # Atom type encoding
    atom_map = {'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'P': 8, 'I': 9, 'B': 10}
    for i, a in enumerate(mol.GetAtoms()):
        atom_types[i] = atom_map.get(a.GetSymbol(), 11)

    for i, b in enumerate(mol.GetBonds()):
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = int(b.GetBondTypeAsDouble())
        bond_matrix[a1, a2] = bt
        bond_matrix[a2, a1] = bt
        bond_endpoints[i, 0] = a1
        bond_endpoints[i, 1] = a2
        bond_types_vec[i] = bt

    return {
        'atom_types': atom_types,
        'bond_matrix': bond_matrix,
        'bond_endpoints': bond_endpoints,
        'bond_types': bond_types_vec,
        'n_atoms': na,
        'n_bonds': nb,
    }


def compute_ma_gpu(mol_tensors):
    """Compute Assembly Index entirely on GPU using tensor operations.

    Method:
    1. Bond signature = (min_atom_type, bond_type, max_atom_type) -> integer code
    2. Count duplicate bond signatures -> single-bond savings
    3. For connected pairs: 2-bond fragment signature -> count duplicates
    4. For connected triples: 3-bond fragment signature -> count duplicates
    5. Total savings = sum of all fragment-level savings
    6. MA = n_bonds - 1 - total_savings
    """
    at = mol_tensors['atom_types'].to(device)
    be = mol_tensors['bond_endpoints'].to(device)
    bt = mol_tensors['bond_types'].to(device)
    bm = mol_tensors['bond_matrix'].to(device)
    nb = mol_tensors['n_bonds']

    if nb <= 1:
        return nb

    # ========================================
    # LEVEL 1: Single-bond duplicate counting
    # ========================================
    # Bond signature: encode as (min_atom, bond_type, max_atom) -> single int
    a1_types = at[be[:nb, 0]]  # atom type of first endpoint
    a2_types = at[be[:nb, 1]]  # atom type of second endpoint
    min_at = torch.minimum(a1_types, a2_types)
    max_at = torch.maximum(a1_types, a2_types)

    # Unique signature per bond: min_atom * 1000 + bond_type * 100 + max_atom
    bond_sigs = min_at * 1000 + bt[:nb] * 100 + max_at  # (nb,)

    # Count duplicates: for each unique signature, count occurrences
    unique_sigs, counts = torch.unique(bond_sigs, return_counts=True)
    # Savings from single-bond duplicates: (count - 1) * (1 - 1) = 0
    # Single bonds have size=1, so savings = 0 per duplicate
    savings_1 = 0  # single bonds never save anything

    # ========================================
    # LEVEL 2: Two-bond connected fragment duplicates
    # ========================================
    # Two bonds are connected if they share an atom
    # For each pair (i,j) where i<j: check if they share an endpoint
    # Fragment signature = sorted pair of bond signatures

    savings_2 = torch.tensor(0.0, device=device)

    if nb >= 2:
        # Build adjacency between bonds: bonds i and j are adjacent if they share an atom
        # be[:nb] has shape (nb, 2)
        # Shared atom: be[i,0]==be[j,0] or be[i,0]==be[j,1] or be[i,1]==be[j,0] or be[i,1]==be[j,1]

        e = be[:nb]  # (nb, 2)
        # Expand for pairwise comparison
        e1 = e.unsqueeze(1).expand(-1, nb, -1)  # (nb, nb, 2)
        e2 = e.unsqueeze(0).expand(nb, -1, -1)  # (nb, nb, 2)

        # Check all 4 endpoint combinations
        shared = ((e1[:,:,0] == e2[:,:,0]) | (e1[:,:,0] == e2[:,:,1]) |
                  (e1[:,:,1] == e2[:,:,0]) | (e1[:,:,1] == e2[:,:,1]))
        # Remove diagonal and lower triangle
        mask = torch.triu(shared, diagonal=1)  # (nb, nb) upper triangle, True where bonds are adjacent

        if mask.any():
            # Get pairs of adjacent bonds
            pairs_i, pairs_j = torch.where(mask)  # indices of adjacent bond pairs

            # 2-bond fragment signature: sorted pair of bond sigs
            sig_i = bond_sigs[pairs_i]
            sig_j = bond_sigs[pairs_j]
            frag2_sig = torch.minimum(sig_i, sig_j) * 100000 + torch.maximum(sig_i, sig_j)

            # Count duplicate 2-bond fragments
            if len(frag2_sig) > 0:
                unique_f2, counts_f2 = torch.unique(frag2_sig, return_counts=True)
                # Savings: for each fragment appearing k times, non-overlapping copies save (2-1)*(k'-1)
                # k' <= k (non-overlapping constraint), approximate as k//2 pairs
                dup_mask = counts_f2 >= 2
                if dup_mask.any():
                    dup_counts = counts_f2[dup_mask]
                    # Each duplicate 2-bond fragment saves 1 step (size-1=1) per extra copy
                    # Approximate non-overlapping: min(count//2, count-1) copies
                    n_copies = torch.clamp(dup_counts - 1, min=0)
                    savings_2 = n_copies.sum().float()

    # ========================================
    # LEVEL 3: Three-bond connected fragment duplicates
    # ========================================
    savings_3 = torch.tensor(0.0, device=device)

    if nb >= 3 and nb <= 30:  # only for small molecules (combinatorial explosion)
        # For 3-bond fragments: find all connected triples
        # A triple (i,j,k) is connected if each pair shares an atom
        # Use bond adjacency matrix
        if mask.shape[0] >= 3:
            adj_bonds = mask.float()  # bond-bond adjacency (upper tri)
            adj_full = adj_bonds + adj_bonds.t()  # make symmetric

            # Find triangles in bond adjacency graph = connected 3-bond fragments
            # A^2[i,k] > 0 means i and k have a common neighbor bond j
            # A[i,k] AND A^2[i,k] > 0 means i,k adjacent AND have common neighbor = triangle
            adj2 = adj_full @ adj_full
            triangles = adj_full * adj2  # element-wise: nonzero where triangle exists
            triangles = torch.triu(triangles, diagonal=1)

            if triangles.any():
                tri_i, tri_k = torch.where(triangles > 0)
                # For each (i,k) that form a triangle edge, the middle bond j satisfies:
                # adj[i,j] and adj[j,k]
                # Fragment signature for 3-bond: sorted triple of bond sigs
                sig_ik_min = torch.minimum(bond_sigs[tri_i], bond_sigs[tri_k])
                sig_ik_max = torch.maximum(bond_sigs[tri_i], bond_sigs[tri_k])
                # Approximate 3-fragment signature (ignoring middle bond for speed)
                frag3_sig = sig_ik_min * 10000000 + sig_ik_max

                unique_f3, counts_f3 = torch.unique(frag3_sig, return_counts=True)
                dup_mask3 = counts_f3 >= 2
                if dup_mask3.any():
                    n_copies3 = torch.clamp(counts_f3[dup_mask3] - 1, min=0)
                    savings_3 = (n_copies3 * 2).sum().float()  # size-1=2 per copy

    # ========================================
    # COMPUTE MA
    # ========================================
    total_savings = savings_2 + savings_3
    # Cap savings at naive-1 (can't save more than total steps)
    total_savings = torch.clamp(total_savings, max=float(nb - 1))
    ma = nb - 1 - int(total_savings.item())
    return max(ma, 1)


def batch_ma_full_gpu(smiles_list):
    """Process batch of molecules — ALL computation on GPU."""
    results = []
    for smi in smiles_list:
        tensors = smiles_to_tensors(smi)
        if tensors is None:
            continue
        ma = compute_ma_gpu(tensors)
        results.append((smi, ma, tensors['n_bonds']))
    return results


# ================================================================
# MAIN
# ================================================================
print("=" * 60)
print("ASSEMBLY INDEX — FULL GPU")
print(f"Device: {device}")
print("=" * 60)

# Validation molecules
known = [
    ("Ethanol", "CCO", 1),
    ("Benzene", "c1ccccc1", 4),
    ("Naphthalene", "c1ccc2ccccc2c1", 6),
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", 9),
    ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", 11),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", 10),
    ("Glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", 7),
]

print(f"\n{'Name':<16} {'CPU MA':>7} {'GPU MA':>7} {'Match':>6}")
print("-" * 42)

for name, smi, expected_ma in known:
    tensors = smiles_to_tensors(smi)
    if tensors is None:
        print(f"{name:<16} ERROR")
        continue
    gpu_ma = compute_ma_gpu(tensors)
    match = "OK" if abs(gpu_ma - expected_ma) <= 2 else "DIFF"
    print(f"{name:<16} {expected_ma:>7} {gpu_ma:>7} {match:>6}")

# Batch benchmark
print(f"\n--- BATCH BENCHMARK ---")
rings = ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1cc[nH]c1", "c1ccoc1", "c1ccsc1"]
fgs = ["O", "N", "F", "Cl", "C(=O)O", "C(=O)N", "C#N", "OC", "NC"]
links = ["", "C", "CC", "CCC"]

all_smiles = set()
for r in rings:
    for fg in fgs:
        for lnk in links:
            m = Chem.MolFromSmiles(r + lnk + fg)
            if m: all_smiles.add(Chem.MolToSmiles(m))
    for r2 in rings:
        for lnk in links[:2]:
            m = Chem.MolFromSmiles(r + lnk + r2)
            if m: all_smiles.add(Chem.MolToSmiles(m))

all_smiles = list(all_smiles)
print(f"{len(all_smiles)} molecules")

t0 = time.time()
results = batch_ma_full_gpu(all_smiles)
elapsed = time.time() - t0

ma_vals = [r[1] for r in results]
print(f"Processed: {len(results)} in {elapsed:.2f}s ({len(results)/elapsed:.0f} mol/s)")
print(f"MA range: {min(ma_vals)}-{max(ma_vals)}, mean={np.mean(ma_vals):.2f}")

# Compare with CPU exact on subset
from assembly_tensorized import compute_ma_cpu

print(f"\n--- GPU vs CPU COMPARISON ---")
print(f"{'SMILES':<40} {'CPU':>5} {'GPU':>5} {'Match':>6}")
print("-" * 60)
n_match = 0
n_close = 0
n_total = 0
for smi, gpu_ma, nb in results[:50]:
    mol = Chem.MolFromSmiles(smi)
    cpu_ma = compute_ma_cpu(mol, timeout=2.0)
    exact = gpu_ma == cpu_ma
    close = abs(gpu_ma - cpu_ma) <= 2
    if exact: n_match += 1
    if close: n_close += 1
    n_total += 1
    if not close:
        print(f"  {smi:<38} {cpu_ma:>5} {gpu_ma:>5} {'DIFF':>6}")

print(f"\nExact match: {n_match}/{n_total} ({100*n_match/n_total:.0f}%)")
print(f"Within +/-2: {n_close}/{n_total} ({100*n_close/n_total:.0f}%)")
print(f"\nNote: GPU uses approximate fragment matching (level 1-3)")
print(f"CPU uses exact branch-and-bound (level 1-8)")
print(f"Difference is expected for complex molecules with large reusable fragments.")
