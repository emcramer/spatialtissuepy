"""
Low-level parsers for PhysiCell output files.

This module handles the actual parsing of XML and MAT files from PhysiCell,
extracting cell data, metadata, and microenvironment information.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# XML Parsing
# -----------------------------------------------------------------------------

@dataclass
class PhysiCellMetadata:
    """Container for PhysiCell simulation metadata."""
    time: float
    time_units: str
    space_units: str
    runtime: float
    program_name: str
    program_version: str
    domain_min: Tuple[float, float, float]
    domain_max: Tuple[float, float, float]
    substrate_names: List[str]
    cell_type_names: List[str]
    cell_type_ids: List[int]
    extra: Dict[str, Any]


def parse_physicell_xml(xml_path: Path) -> PhysiCellMetadata:
    """
    Parse a PhysiCell output XML file for metadata.
    
    Parameters
    ----------
    xml_path : Path
        Path to the output*.xml file.
        
    Returns
    -------
    PhysiCellMetadata
        Parsed metadata from the XML file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract time
    time_elem = root.find('.//current_time')
    time = float(time_elem.text) if time_elem is not None else 0.0
    time_units = time_elem.get('units', 'min') if time_elem is not None else 'min'
    
    # Extract runtime
    runtime_elem = root.find('.//current_runtime')
    runtime = float(runtime_elem.text) if runtime_elem is not None else 0.0
    
    # Extract space units
    space_units = 'micron'  # Default
    mesh_elem = root.find('.//mesh')
    if mesh_elem is not None:
        space_units = mesh_elem.get('units', 'micron')
    
    # Extract program info - check both old and new XML structures
    program_name = 'PhysiCell'
    program_version = 'unknown'
    
    # New format: software/n and software/version
    software_elem = root.find('.//software')
    if software_elem is not None:
        name_elem = software_elem.find('n')
        ver_elem = software_elem.find('version')
        if name_elem is not None and name_elem.text:
            program_name = name_elem.text
        if ver_elem is not None and ver_elem.text:
            program_version = ver_elem.text
    
    # Old format: program/name and program/version
    if program_version == 'unknown':
        program_elem = root.find('.//program')
        if program_elem is not None:
            name_elem = program_elem.find('name')
            ver_elem = program_elem.find('version')
            if name_elem is not None:
                program_name = name_elem.text or 'PhysiCell'
            if ver_elem is not None:
                program_version = ver_elem.text or 'unknown'
    
    # Extract domain bounds from mesh bounding_box
    domain_min = (-500.0, -500.0, -10.0)
    domain_max = (500.0, 500.0, 10.0)
    
    bbox_elem = root.find('.//bounding_box')
    if bbox_elem is not None and bbox_elem.text:
        try:
            coords = [float(x) for x in bbox_elem.text.split()]
            if len(coords) == 6:
                domain_min = (coords[0], coords[1], coords[2])
                domain_max = (coords[3], coords[4], coords[5])
        except (ValueError, IndexError):
            pass
    
    # Fall back to domain element if bounding_box not found
    if domain_min == (-500.0, -500.0, -10.0):
        domain_elem = root.find('.//domain')
        if domain_elem is not None:
            x_min = domain_elem.find('x_min')
            y_min = domain_elem.find('y_min')
            z_min = domain_elem.find('z_min')
            x_max = domain_elem.find('x_max')
            y_max = domain_elem.find('y_max')
            z_max = domain_elem.find('z_max')
            
            if all(e is not None for e in [x_min, y_min, z_min, x_max, y_max, z_max]):
                domain_min = (
                    float(x_min.text), float(y_min.text), float(z_min.text)
                )
                domain_max = (
                    float(x_max.text), float(y_max.text), float(z_max.text)
                )
    
    # Extract substrate names
    substrate_names = []
    variables_elem = root.find('.//variables')
    if variables_elem is not None:
        for var in variables_elem.findall('variable'):
            name = var.get('name', 'unknown')
            substrate_names.append(name)
    
    # Extract cell type names and IDs
    cell_type_names = []
    cell_type_ids = []
    
    # First try: cell_types in simplified_data (PhysiCell 1.10+ output XML format)
    cell_types_elem = root.find('.//simplified_data/cell_types')
    if cell_types_elem is not None:
        for type_elem in cell_types_elem.findall('type'):
            cell_id = int(type_elem.get('ID', len(cell_type_ids)))
            name = type_elem.text or f'type_{cell_id}'
            cell_type_ids.append(cell_id)
            cell_type_names.append(name)
    
    # Second try: cell_definitions in settings XML format
    if not cell_type_names:
        cell_defs = root.find('.//cell_definitions')
        if cell_defs is not None:
            for cell_def in cell_defs.findall('cell_definition'):
                name = cell_def.get('name', 'unknown')
                cell_id = int(cell_def.get('ID', len(cell_type_names)))
                cell_type_names.append(name)
                cell_type_ids.append(cell_id)
    
    # Extract labels from simplified_data (column indices for MAT file)
    labels_elem = root.find('.//simplified_data/labels')
    custom_labels = {}
    if labels_elem is not None:
        for label in labels_elem.findall('label'):
            index = int(label.get('index', 0))
            size = int(label.get('size', 1))
            name = label.text or f'custom_{index}'
            custom_labels[index] = (name, size)
    
    extra = {
        'custom_labels': custom_labels,
        'xml_path': str(xml_path),
    }
    
    return PhysiCellMetadata(
        time=time,
        time_units=time_units,
        space_units=space_units,
        runtime=runtime,
        program_name=program_name,
        program_version=program_version,
        domain_min=domain_min,
        domain_max=domain_max,
        substrate_names=substrate_names,
        cell_type_names=cell_type_names,
        cell_type_ids=cell_type_ids,
        extra=extra,
    )


def get_cell_type_mapping(
    xml_path: Optional[Path] = None,
    settings_xml_path: Optional[Path] = None
) -> Dict[int, str]:
    """
    Get mapping from cell type IDs to names.
    
    Reads from the PhysiCell output XML (preferred) or settings XML.
    
    Parameters
    ----------
    xml_path : Path, optional
        Path to output*.xml file (preferred source for cell types).
    settings_xml_path : Path, optional
        Path to PhysiCell_settings.xml or config.xml file.
        
    Returns
    -------
    dict
        Mapping from cell type ID (int) to name (str).
    """
    mapping = {}
    
    # Try output XML first (has cell_types in simplified_data for PhysiCell 1.10+)
    if xml_path is not None:
        try:
            metadata = parse_physicell_xml(xml_path)
            for cell_id, name in zip(metadata.cell_type_ids, metadata.cell_type_names):
                mapping[cell_id] = name
        except Exception:
            pass
    
    # Fall back to settings XML if output XML didn't have cell types
    if not mapping and settings_xml_path is not None and settings_xml_path.exists():
        try:
            tree = ET.parse(settings_xml_path)
            root = tree.getroot()
            
            cell_defs = root.find('.//cell_definitions')
            if cell_defs is not None:
                for cell_def in cell_defs.findall('cell_definition'):
                    name = cell_def.get('name', 'unknown')
                    cell_id = int(cell_def.get('ID', len(mapping)))
                    mapping[cell_id] = name
        except Exception:
            pass
    
    # Default if nothing found
    if not mapping:
        mapping[0] = 'default'
    
    return mapping


# -----------------------------------------------------------------------------
# MAT File Parsing
# -----------------------------------------------------------------------------

# Standard PhysiCell cell data row indices for PhysiCell 1.10+ (MultiCellDS v2)
# These indices are derived from the <labels> section in output XML
CELL_DATA_INDICES_V2 = {
    'ID': 0,
    'position_x': 1,
    'position_y': 2,
    'position_z': 3,
    'total_volume': 4,
    'cell_type': 5,
    'cycle_model': 6,
    'current_phase': 7,
    'elapsed_time_in_phase': 8,
    'nuclear_volume': 9,
    'cytoplasmic_volume': 10,
    'fluid_fraction': 11,
    'calcified_fraction': 12,
    'dead': 26,  # Boolean flag for dead cells
    'radius': 37,
    'nuclear_radius': 38,
    'surface_area': 39,
}

# Legacy indices for older PhysiCell versions (pre-1.10)
CELL_DATA_INDICES_LEGACY = {
    'ID': 0,
    'position_x': 1,
    'position_y': 2,
    'position_z': 3,
    'total_volume': 4,
    'nuclear_volume': 5,
    'cytoplasmic_volume': 6,
    'fluid_volume': 7,
    'solid_volume': 8,
    'radius': 9,
    'nuclear_radius': 10,
    'surface_area': 11,
    'current_phase': 13,
    'elapsed_time_in_phase': 14,
    'cell_type': 5,  # Often stored here in older versions
}


def parse_cells_mat(
    mat_path: Path,
    cell_type_mapping: Optional[Dict[int, str]] = None,
    index_mapping: Optional[Dict[str, int]] = None
) -> Dict[str, np.ndarray]:
    """
    Parse a PhysiCell cells_physicell.mat file.
    
    Parameters
    ----------
    mat_path : Path
        Path to the *_cells_physicell.mat file.
    cell_type_mapping : dict, optional
        Mapping from cell type IDs to names.
    index_mapping : dict, optional
        Custom row index mapping for different PhysiCell versions.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'positions': (n_cells, 3) array of x, y, z coordinates
        - 'cell_types': (n_cells,) array of cell type names or IDs
        - 'cell_type_ids': (n_cells,) array of integer cell type IDs
        - 'volumes': (n_cells,) array of total volumes
        - 'radii': (n_cells,) array of cell radii
        - 'phases': (n_cells,) array of cell cycle phases
        - 'ids': (n_cells,) array of cell IDs
        - 'raw_data': Full matrix from file
    """
    from scipy.io import loadmat
    
    # Load the MAT file
    mat_data = loadmat(str(mat_path))
    
    # Find the cells data - usually named 'cells' or similar
    cell_matrix = None
    for key in ['cells', 'basic_agents', 'cell_data']:
        if key in mat_data:
            cell_matrix = mat_data[key]
            break
    
    if cell_matrix is None:
        # Try to find any 2D array that looks like cell data
        for key, value in mat_data.items():
            if not key.startswith('_') and isinstance(value, np.ndarray):
                if value.ndim == 2 and value.shape[0] >= 4:
                    cell_matrix = value
                    break
    
    if cell_matrix is None:
        raise ValueError(f"Could not find cell data in {mat_path}")
    
    # Ensure cells are columns
    if cell_matrix.shape[0] > cell_matrix.shape[1]:
        cell_matrix = cell_matrix.T
    
    n_cells = cell_matrix.shape[1]
    
    if n_cells == 0:
        return {
            'positions': np.empty((0, 3)),
            'cell_types': np.array([], dtype=str),
            'cell_type_ids': np.array([], dtype=int),
            'volumes': np.array([]),
            'radii': np.array([]),
            'phases': np.array([], dtype=int),
            'ids': np.array([], dtype=int),
            'raw_data': cell_matrix,
        }
    
    # Use provided or default index mapping
    if index_mapping is None:
        # Auto-detect based on matrix shape
        # PhysiCell 1.10+ typically has 150+ rows per cell
        if cell_matrix.shape[0] >= 30:
            index_mapping = CELL_DATA_INDICES_V2
        else:
            index_mapping = CELL_DATA_INDICES_LEGACY
    
    # Extract positions
    positions = np.column_stack([
        cell_matrix[index_mapping.get('position_x', 1), :],
        cell_matrix[index_mapping.get('position_y', 2), :],
        cell_matrix[index_mapping.get('position_z', 3), :],
    ])
    
    # Extract cell IDs
    ids = cell_matrix[index_mapping.get('ID', 0), :].astype(int)
    
    # Extract cell types
    cell_type_idx = index_mapping.get('cell_type', 5)
    if cell_type_idx < cell_matrix.shape[0]:
        cell_type_ids = cell_matrix[cell_type_idx, :].astype(int)
    else:
        cell_type_ids = np.zeros(n_cells, dtype=int)
    
    # Map to names if mapping provided
    if cell_type_mapping:
        cell_types = np.array([
            cell_type_mapping.get(int(ct), f'type_{int(ct)}')
            for ct in cell_type_ids
        ])
    else:
        cell_types = np.array([f'type_{int(ct)}' for ct in cell_type_ids])
    
    # Extract volumes
    vol_idx = index_mapping.get('total_volume', 4)
    if vol_idx < cell_matrix.shape[0]:
        volumes = cell_matrix[vol_idx, :]
    else:
        volumes = np.ones(n_cells) * 2494.0  # Default PhysiCell volume
    
    # Extract radii
    radius_idx = index_mapping.get('radius', 37)  # Default for V2 format
    if radius_idx < cell_matrix.shape[0]:
        radii = cell_matrix[radius_idx, :]
    else:
        # Compute from volume (assuming spherical cells)
        radii = np.cbrt(3 * volumes / (4 * np.pi))
    
    # Extract cell cycle phase
    phase_idx = index_mapping.get('current_phase', 7)  # Default for V2 format
    if phase_idx < cell_matrix.shape[0]:
        phases = cell_matrix[phase_idx, :].astype(int)
    else:
        phases = np.zeros(n_cells, dtype=int)
    
    # Extract dead flag (PhysiCell 1.10+ has explicit dead flag at index 26)
    dead_idx = index_mapping.get('dead', 26)
    if dead_idx < cell_matrix.shape[0]:
        dead_flags = cell_matrix[dead_idx, :].astype(int)
    else:
        # Fall back to inferring from phase code
        dead_flags = np.array([1 if p >= 100 else 0 for p in phases])
    
    return {
        'positions': positions,
        'cell_types': cell_types,
        'cell_type_ids': cell_type_ids,
        'volumes': volumes,
        'radii': radii,
        'phases': phases,
        'dead_flags': dead_flags,
        'ids': ids,
        'raw_data': cell_matrix,
    }


def parse_microenvironment_mat(
    mat_path: Path,
    substrate_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Parse a PhysiCell microenvironment MAT file.
    
    Parameters
    ----------
    mat_path : Path
        Path to the *_microenvironment0.mat file.
    substrate_names : list of str, optional
        Names of substrates (from XML metadata).
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'voxel_positions': (n_voxels, 3) array of voxel centers
        - 'concentrations': dict mapping substrate names to (n_voxels,) arrays
        - 'raw_data': Full matrix from file
    """
    from scipy.io import loadmat
    
    mat_data = loadmat(str(mat_path))
    
    # Find the microenvironment data
    me_matrix = None
    for key in ['multiscale_microenvironment', 'microenvironment', 'M']:
        if key in mat_data:
            me_matrix = mat_data[key]
            break
    
    if me_matrix is None:
        for key, value in mat_data.items():
            if not key.startswith('_') and isinstance(value, np.ndarray):
                if value.ndim == 2 and value.shape[0] >= 4:
                    me_matrix = value
                    break
    
    if me_matrix is None:
        raise ValueError(f"Could not find microenvironment data in {mat_path}")
    
    # Structure: rows 0-2 are x,y,z; row 3 is volume; rows 4+ are substrates
    voxel_positions = me_matrix[:3, :].T  # (n_voxels, 3)
    
    # Extract substrate concentrations
    concentrations = {}
    n_substrates = me_matrix.shape[0] - 4
    
    if substrate_names is None:
        substrate_names = [f'substrate_{i}' for i in range(n_substrates)]
    
    for i, name in enumerate(substrate_names[:n_substrates]):
        concentrations[name] = me_matrix[4 + i, :]
    
    return {
        'voxel_positions': voxel_positions,
        'concentrations': concentrations,
        'raw_data': me_matrix,
    }


# -----------------------------------------------------------------------------
# Cell Cycle Phase Mapping
# -----------------------------------------------------------------------------

# PhysiCell cell cycle phase codes
CELL_CYCLE_PHASES = {
    0: 'Ki67_positive_premitotic',
    1: 'Ki67_positive_postmitotic',
    2: 'Ki67_positive',
    3: 'Ki67_negative',
    4: 'G0G1_phase',
    5: 'G0_phase',
    6: 'G1_phase',
    7: 'G1a_phase',
    8: 'G1b_phase',
    9: 'G1c_phase',
    10: 'S_phase',
    11: 'G2M_phase',
    12: 'G2_phase',
    13: 'M_phase',
    14: 'live',
    100: 'apoptotic',
    101: 'necrotic_swelling',
    102: 'necrotic_lysed',
    103: 'necrotic',
    104: 'debris',
}


def get_phase_name(phase_code: int) -> str:
    """Get human-readable phase name from PhysiCell phase code."""
    return CELL_CYCLE_PHASES.get(phase_code, f'phase_{phase_code}')


def is_alive(phase_code: int) -> bool:
    """Check if a cell is alive based on its phase code."""
    return phase_code < 100


def is_dead(phase_code: int) -> bool:
    """Check if a cell is dead based on its phase code."""
    return phase_code >= 100
