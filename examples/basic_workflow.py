"""
Basic workflow example for spatialtissuepy.

This script demonstrates the core functionality of the package.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for local development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatialtissuepy import SpatialTissueData
from spatialtissuepy.core.cell import Cell


def main():
    print("=" * 60)
    print("spatialtissuepy - Basic Workflow Example")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Create data from scratch
    # -------------------------------------------------------------------------
    print("\n1. Creating SpatialTissueData from arrays...")
    
    np.random.seed(42)
    n_cells = 500
    
    # Generate random coordinates
    coords = np.random.rand(n_cells, 2) * 1000  # 1000x1000 µm FOV
    
    # Generate cell types
    cell_types = np.random.choice(
        ['Tumor', 'T_cell', 'Macrophage', 'Stromal', 'Endothelial'],
        n_cells,
        p=[0.4, 0.2, 0.15, 0.15, 0.1]  # Tumor-dominated
    )
    
    # Generate marker expression
    markers = pd.DataFrame({
        'CD3': np.random.rand(n_cells),
        'CD8': np.random.rand(n_cells),
        'CD68': np.random.rand(n_cells),
        'PanCK': np.random.rand(n_cells),
    })
    
    # Create SpatialTissueData object
    data = SpatialTissueData(
        coordinates=coords,
        cell_types=cell_types,
        markers=markers,
        metadata={'tissue': 'synthetic', 'experiment': 'demo'}
    )
    
    print(f"   Created: {data}")
    print(f"   Cell type counts:\n{data.cell_type_counts}")
    
    # -------------------------------------------------------------------------
    # 2. Explore the data
    # -------------------------------------------------------------------------
    print("\n2. Exploring the data...")
    
    print(f"   Number of cells: {data.n_cells}")
    print(f"   Number of dimensions: {data.n_dims}")
    print(f"   Number of cell types: {data.n_cell_types}")
    print(f"   Spatial bounds: {data.bounds}")
    print(f"   Spatial extent: {data.extent}")
    print(f"   Marker names: {data.marker_names}")
    
    # -------------------------------------------------------------------------
    # 3. Access individual cells
    # -------------------------------------------------------------------------
    print("\n3. Accessing individual cells...")
    
    cell = data.get_cell(0)
    print(f"   First cell: {cell}")
    print(f"   Coordinates: {cell.coordinates}")
    print(f"   CD3 expression: {cell.get_marker('CD3'):.3f}")
    
    # -------------------------------------------------------------------------
    # 4. Spatial queries
    # -------------------------------------------------------------------------
    print("\n4. Performing spatial queries...")
    
    # Find cells near center of FOV
    center = np.array([500, 500])
    
    # Radius query
    nearby_indices = data.query_radius(center, radius=100)
    print(f"   Cells within 100µm of center: {len(nearby_indices)}")
    
    # KNN query
    distances, knn_indices = data.query_knn(center, k=10)
    print(f"   10 nearest neighbors to center:")
    print(f"   - Distances: {distances[:5].round(2)}...")
    print(f"   - Indices: {knn_indices[:5]}...")
    
    # -------------------------------------------------------------------------
    # 5. Subset data
    # -------------------------------------------------------------------------
    print("\n5. Subsetting data...")
    
    # By cell type
    t_cells = data.subset(cell_types=['T_cell'])
    print(f"   T cells only: {t_cells.n_cells} cells")
    
    # By multiple types
    immune_cells = data.subset(cell_types=['T_cell', 'Macrophage'])
    print(f"   Immune cells: {immune_cells.n_cells} cells")
    
    # By indices
    subset_indices = data.get_cells_by_type('Tumor')[:50]
    tumor_subset = data.subset(indices=subset_indices)
    print(f"   First 50 tumor cells: {tumor_subset.n_cells} cells")
    
    # -------------------------------------------------------------------------
    # 6. Iterate over cells
    # -------------------------------------------------------------------------
    print("\n6. Iterating over cells...")
    
    tumor_count = 0
    total_cd3 = 0
    for cell in data.iter_cells():
        if cell.cell_type == 'Tumor':
            tumor_count += 1
        total_cd3 += cell.get_marker('CD3', default=0)
    
    print(f"   Tumor cells counted: {tumor_count}")
    print(f"   Mean CD3 expression: {total_cd3 / data.n_cells:.3f}")
    
    # -------------------------------------------------------------------------
    # 7. Export data
    # -------------------------------------------------------------------------
    print("\n7. Exporting data...")
    
    # To DataFrame
    df = data.to_dataframe()
    print(f"   DataFrame shape: {df.shape}")
    print(f"   DataFrame columns: {list(df.columns)}")
    
    # To CSV (in a temp location for demo)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        data.to_csv(f.name)
        print(f"   Saved to: {f.name}")
    
    # -------------------------------------------------------------------------
    # 8. Multi-sample data
    # -------------------------------------------------------------------------
    print("\n8. Working with multi-sample data...")
    
    # Create multi-sample data
    sample_ids = np.array(['sample_A'] * 250 + ['sample_B'] * 250)
    multi_data = SpatialTissueData(
        coordinates=coords,
        cell_types=cell_types,
        sample_ids=sample_ids,
        markers=markers
    )
    
    print(f"   Multi-sample data: {multi_data}")
    print(f"   Is multi-sample: {multi_data.is_multisample}")
    print(f"   Number of samples: {multi_data.n_samples}")
    print(f"   Sample IDs: {multi_data.sample_ids_unique}")
    
    # Iterate over samples
    for sample_id, sample_data in multi_data.iter_samples():
        print(f"   {sample_id}: {sample_data.n_cells} cells, "
              f"{sample_data.n_cell_types} types")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
