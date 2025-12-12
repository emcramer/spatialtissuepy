"""
Generate sample spatial tissue data for testing and examples.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_clustered_tissue(
    n_cells: int = 1000,
    n_clusters: int = 5,
    fov_size: float = 1000.0,
    cell_types: list = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic clustered tissue data.
    
    Parameters
    ----------
    n_cells : int
        Total number of cells.
    n_clusters : int
        Number of spatial clusters.
    fov_size : float
        Field of view size (square).
    cell_types : list
        List of cell types. If None, uses default.
    seed : int
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with x, y, cell_type columns.
    """
    if cell_types is None:
        cell_types = ['Tumor', 'T_cell', 'Macrophage', 'Stromal', 'Endothelial']
    
    np.random.seed(seed)
    
    # Generate cluster centers
    cluster_centers = np.random.rand(n_clusters, 2) * fov_size
    cluster_sizes = np.random.dirichlet(np.ones(n_clusters)) * n_cells
    cluster_sizes = cluster_sizes.astype(int)
    cluster_sizes[-1] = n_cells - cluster_sizes[:-1].sum()  # Ensure total is correct
    
    # Generate cells around clusters
    all_coords = []
    all_types = []
    
    for i, (center, size) in enumerate(zip(cluster_centers, cluster_sizes)):
        # Each cluster has a dominant cell type
        dominant_type = cell_types[i % len(cell_types)]
        
        # Cluster spread
        spread = fov_size * 0.1  # 10% of FOV
        
        coords = np.random.randn(size, 2) * spread + center
        coords = np.clip(coords, 0, fov_size)
        
        # Assign cell types (80% dominant, 20% random)
        types = []
        for _ in range(size):
            if np.random.rand() < 0.8:
                types.append(dominant_type)
            else:
                types.append(np.random.choice(cell_types))
        
        all_coords.append(coords)
        all_types.extend(types)
    
    coords = np.vstack(all_coords)
    
    return pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'cell_type': all_types
    })


def generate_multisample_data(
    n_samples: int = 3,
    cells_per_sample: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate multi-sample spatial data.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    cells_per_sample : int
        Cells per sample.
    seed : int
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with x, y, cell_type, sample_id columns.
    """
    np.random.seed(seed)
    
    all_dfs = []
    for i in range(n_samples):
        df = generate_clustered_tissue(
            n_cells=cells_per_sample,
            seed=seed + i
        )
        df['sample_id'] = f'sample_{i+1:02d}'
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)


if __name__ == '__main__':
    # Generate and save sample data
    output_dir = Path(__file__).parent
    
    # Single sample
    single = generate_clustered_tissue(n_cells=1000, seed=42)
    single.to_csv(output_dir / 'sample_tissue.csv', index=False)
    print(f"Generated sample_tissue.csv with {len(single)} cells")
    
    # Multi-sample
    multi = generate_multisample_data(n_samples=3, cells_per_sample=500, seed=42)
    multi.to_csv(output_dir / 'multi_sample_tissue.csv', index=False)
    print(f"Generated multi_sample_tissue.csv with {len(multi)} cells")
