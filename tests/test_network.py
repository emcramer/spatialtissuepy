"""
Tests for spatialtissuepy.network module.

Tests graph construction, centrality measures, clustering, communicability,
and assortativity analysis.
"""

import pytest
import numpy as np

# Check if networkx is available
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from spatialtissuepy import SpatialTissueData

if HAS_NETWORKX:
    from spatialtissuepy.network import (
        # Graph construction
        GraphMethod,
        build_graph,
        build_proximity_graph,
        build_knn_graph,
        build_delaunay_graph,
        build_gabriel_graph,
        # CellGraph
        CellGraph,
        # Centrality
        degree_centrality,
        betweenness_centrality,
        closeness_centrality,
        eigenvector_centrality,
        pagerank,
        centrality_by_type,
        mean_centrality_by_type,
        # Clustering
        clustering_coefficient,
        average_clustering,
        transitivity,
        triangles,
        clustering_by_type,
        connected_components,
        n_connected_components,
        largest_component_size,
        bridges,
        articulation_points,
        # Communicability
        communicability,
        communicability_between_types,
        shortest_path_length_between_types,
        average_shortest_path_length,
        diameter,
        global_efficiency,
        local_efficiency,
        # Assortativity
        degree_assortativity,
        type_assortativity,
        attribute_mixing_matrix,
        homophily_ratio,
        heterophily_ratio,
        average_neighbor_degree,
        neighbor_type_distribution,
    )


pytestmark = pytest.mark.skipif(
    not HAS_NETWORKX,
    reason="NetworkX not installed"
)


# =============================================================================
# Graph Construction Tests
# =============================================================================

class TestGraphConstruction:
    """Tests for graph construction methods."""
    
    def test_build_proximity_graph_basic(self, small_tissue):
        """Test basic proximity graph construction."""
        G = build_proximity_graph(
            small_tissue.coordinates,
            radius=50.0
        )
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == small_tissue.n_cells
        assert G.number_of_edges() > 0
    
    def test_build_proximity_graph_radius_effect(self, small_tissue):
        """Test that larger radius creates more edges."""
        G_small = build_proximity_graph(small_tissue.coordinates, radius=20)
        G_large = build_proximity_graph(small_tissue.coordinates, radius=50)
        
        # Larger radius should create more connections
        assert G_large.number_of_edges() >= G_small.number_of_edges()
    
    def test_build_proximity_graph_empty(self):
        """Test proximity graph with no points."""
        coords = np.array([]).reshape(0, 2)
        G = build_proximity_graph(coords, radius=50)
        
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
    
    def test_build_proximity_graph_isolated(self):
        """Test proximity graph with isolated points."""
        coords = np.array([[0, 0], [1000, 1000]])
        G = build_proximity_graph(coords, radius=10)
        
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 0
    
    def test_build_knn_graph_basic(self, small_tissue):
        """Test k-nearest neighbors graph."""
        G = build_knn_graph(small_tissue.coordinates, k=5)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == small_tissue.n_cells
        
        # Each node should have at most k neighbors (directed)
        # In undirected graph, may have more
        assert G.number_of_edges() > 0
    
    def test_build_knn_graph_mutual(self, small_tissue):
        """Test mutual k-NN graph."""
        G_mutual = build_knn_graph(
            small_tissue.coordinates,
            k=5,
            mutual_knn=True
        )
        G_regular = build_knn_graph(
            small_tissue.coordinates,
            k=5,
            mutual_knn=False
        )
        
        # Mutual should have fewer edges
        assert G_mutual.number_of_edges() <= G_regular.number_of_edges()
    
    def test_build_knn_graph_k_larger_than_n(self):
        """Test k-NN with k larger than number of points."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        G = build_knn_graph(coords, k=10)
        
        # Should connect to all available neighbors
        assert G.number_of_nodes() == 3
    
    def test_build_delaunay_graph_basic(self, small_tissue):
        """Test Delaunay triangulation graph."""
        G = build_delaunay_graph(small_tissue.coordinates[:, :2])
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == small_tissue.n_cells
        assert G.number_of_edges() > 0
    
    def test_build_delaunay_graph_pruning(self, small_tissue):
        """Test Delaunay graph with edge length pruning."""
        G_full = build_delaunay_graph(small_tissue.coordinates[:, :2])
        G_pruned = build_delaunay_graph(
            small_tissue.coordinates[:, :2],
            max_edge_length=30
        )
        
        # Pruned should have fewer or equal edges
        assert G_pruned.number_of_edges() <= G_full.number_of_edges()
    
    def test_build_delaunay_graph_insufficient_points(self):
        """Test Delaunay with too few points."""
        coords = np.array([[0, 0], [1, 1]])
        G = build_delaunay_graph(coords)
        
        # Delaunay needs at least 3 non-collinear points
        # Should return empty or minimal graph
        assert G.number_of_nodes() == 2
    
    def test_build_gabriel_graph_basic(self, small_tissue):
        """Test Gabriel graph construction."""
        G = build_gabriel_graph(small_tissue.coordinates[:, :2])
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == small_tissue.n_cells
    
    def test_build_gabriel_subset_of_delaunay(self, small_tissue):
        """Test Gabriel graph is subset of Delaunay."""
        coords = small_tissue.coordinates[:, :2]
        G_delaunay = build_delaunay_graph(coords)
        G_gabriel = build_gabriel_graph(coords)
        
        # Gabriel should have fewer or equal edges
        assert G_gabriel.number_of_edges() <= G_delaunay.number_of_edges()
    
    def test_build_graph_with_method_enum(self, small_tissue):
        """Test build_graph with GraphMethod enum."""
        G = build_graph(
            small_tissue.coordinates,
            method=GraphMethod.PROXIMITY,
            radius=50
        )
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == small_tissue.n_cells
    
    def test_build_graph_with_string_method(self, small_tissue):
        """Test build_graph with string method."""
        G = build_graph(
            small_tissue.coordinates,
            method='knn',
            k=5
        )
        
        assert isinstance(G, nx.Graph)
    
    def test_build_graph_invalid_method(self, small_tissue):
        """Test error with invalid method."""
        with pytest.raises((ValueError, AttributeError)):
            build_graph(
                small_tissue.coordinates,
                method='invalid_method'
            )


# =============================================================================
# CellGraph Class Tests
# =============================================================================

class TestCellGraph:
    """Tests for CellGraph class."""
    
    def test_cellgraph_from_spatial_data(self, small_tissue):
        """Test creating CellGraph from SpatialTissueData."""
        graph = CellGraph.from_spatial_data(
            small_tissue,
            method='proximity',
            radius=50
        )
        
        assert isinstance(graph, CellGraph)
        assert graph.n_nodes == small_tissue.n_cells
        assert graph.n_edges > 0
    
    def test_cellgraph_from_coordinates(self):
        """Test creating CellGraph from coordinates."""
        coords = np.random.rand(50, 2) * 100
        types = np.array(['A', 'B'] * 25)
        
        graph = CellGraph.from_coordinates(
            coords,
            types,
            method='knn',
            k=5
        )
        
        assert graph.n_nodes == 50
        assert len(graph.cell_types_unique) == 2
    
    def test_cellgraph_properties(self, small_tissue):
        """Test CellGraph properties."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        assert graph.n_nodes > 0
        assert graph.n_edges >= 0
        assert graph.method == 'proximity'
        assert 0 <= graph.density <= 1
        assert len(graph.cell_types_unique) > 0
    
    def test_cellgraph_get_nodes_by_type(self, small_tissue):
        """Test getting nodes by cell type."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        cell_type = graph.cell_types_unique[0]
        nodes = graph.get_nodes_by_type(cell_type)
        
        assert len(nodes) > 0
        assert all(graph.cell_types[i] == cell_type for i in nodes)
    
    def test_cellgraph_subgraph_by_type(self, small_tissue):
        """Test extracting subgraph by cell type."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        if len(graph.cell_types_unique) < 2:
            pytest.skip("Need multiple cell types")
        
        cell_type = graph.cell_types_unique[0]
        subgraph = graph.subgraph_by_type(cell_type)
        
        assert subgraph.n_nodes < graph.n_nodes
        assert len(subgraph.cell_types_unique) == 1
        assert subgraph.cell_types_unique[0] == cell_type
    
    def test_cellgraph_subgraph_multiple_types(self, small_tissue):
        """Test subgraph with multiple types."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        if len(graph.cell_types_unique) < 2:
            pytest.skip("Need multiple cell types")
        
        types_to_keep = graph.cell_types_unique[:2]
        subgraph = graph.subgraph_by_type(types_to_keep)
        
        assert set(subgraph.cell_types_unique).issubset(set(types_to_keep))
    
    def test_cellgraph_neighbors_of_type(self, small_tissue):
        """Test getting neighbors filtered by type."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        # Get node with neighbors
        node = 0
        while graph.G.degree(node) == 0 and node < graph.n_nodes - 1:
            node += 1
        
        if graph.G.degree(node) == 0:
            pytest.skip("No connected nodes")
        
        all_neighbors = graph.neighbors_of_type(node)
        assert len(all_neighbors) > 0
        
        # Filter by type
        if len(graph.cell_types_unique) > 1:
            cell_type = graph.cell_types_unique[0]
            filtered_neighbors = graph.neighbors_of_type(node, cell_type)
            assert len(filtered_neighbors) <= len(all_neighbors)
    
    def test_cellgraph_edge_type_counts(self, small_tissue):
        """Test counting edges by type pairs."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        counts = graph.edge_type_counts()
        
        assert isinstance(counts, dict)
        # Total edges should match
        total_edges = sum(counts.values())
        assert total_edges == graph.n_edges
    
    def test_cellgraph_to_networkx(self, small_tissue):
        """Test converting to NetworkX graph."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        G = graph.to_networkx()
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == graph.n_nodes
        assert G.number_of_edges() == graph.n_edges
    
    def test_cellgraph_repr_str(self, small_tissue):
        """Test string representations."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        repr_str = repr(graph)
        str_str = str(graph)
        
        assert 'CellGraph' in repr_str
        assert 'CellGraph' in str_str
        assert str(graph.n_nodes) in str_str


# =============================================================================
# Centrality Tests
# =============================================================================

class TestCentrality:
    """Tests for centrality measures."""
    
    def test_degree_centrality_basic(self, small_tissue):
        """Test degree centrality calculation."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        centrality = degree_centrality(graph.G)
        
        assert len(centrality) == graph.n_nodes
        assert all(0 <= v <= 1 for v in centrality.values())
    
    def test_betweenness_centrality_basic(self, small_tissue):
        """Test betweenness centrality."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        centrality = betweenness_centrality(graph.G)
        
        assert len(centrality) == graph.n_nodes
        assert all(0 <= v <= 1 for v in centrality.values())
    
    def test_closeness_centrality_basic(self, small_tissue):
        """Test closeness centrality."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        centrality = closeness_centrality(graph.G)
        
        assert len(centrality) == graph.n_nodes
        assert all(0 <= v <= 1 for v in centrality.values())
    
    def test_eigenvector_centrality_basic(self, small_tissue):
        """Test eigenvector centrality."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        try:
            centrality = eigenvector_centrality(graph.G)
            assert len(centrality) == graph.n_nodes
        except nx.PowerIterationFailedConvergence:
            pytest.skip("Eigenvector centrality did not converge")
    
    def test_pagerank_basic(self, small_tissue):
        """Test PageRank centrality."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        centrality = pagerank(graph.G)
        
        assert len(centrality) == graph.n_nodes
        # PageRank sums to 1
        assert abs(sum(centrality.values()) - 1.0) < 0.01
    
    def test_centrality_by_type(self, small_tissue):
        """Test centrality aggregated by cell type."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        result = centrality_by_type(graph, metric='degree')
        
        assert isinstance(result, dict)
        for cell_type in graph.cell_types_unique:
            assert cell_type in result
            assert isinstance(result[cell_type], dict)
            assert 'mean' in result[cell_type]
    
    def test_mean_centrality_by_type(self, small_tissue):
        """Test mean centrality by type."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        result = mean_centrality_by_type(graph, metric='betweenness')
        
        assert isinstance(result, dict)
        for cell_type in graph.cell_types_unique:
            assert cell_type in result
            assert isinstance(result[cell_type], float)
            assert result[cell_type] >= 0


# =============================================================================
# Clustering Tests
# =============================================================================

class TestClustering:
    """Tests for clustering metrics."""
    
    def test_clustering_coefficient_basic(self, small_tissue):
        """Test local clustering coefficient."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        clustering = clustering_coefficient(graph.G)
        
        assert len(clustering) == graph.n_nodes
        assert all(0 <= v <= 1 for v in clustering.values())
    
    def test_average_clustering_basic(self, small_tissue):
        """Test average clustering coefficient."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        avg_clustering = average_clustering(graph.G)
        
        assert isinstance(avg_clustering, float)
        assert 0 <= avg_clustering <= 1
    
    def test_transitivity_basic(self, small_tissue):
        """Test transitivity (global clustering)."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        trans = transitivity(graph.G)
        
        assert isinstance(trans, float)
        assert 0 <= trans <= 1
    
    def test_triangles_basic(self, small_tissue):
        """Test triangle counting."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        tri = triangles(graph.G)
        
        assert len(tri) == graph.n_nodes
        assert all(v >= 0 for v in tri.values())
    
    def test_clustering_by_type(self, small_tissue):
        """Test clustering by cell type."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        result = clustering_by_type(graph)
        
        assert isinstance(result, dict)
        for cell_type in graph.cell_types_unique:
            assert cell_type in result
    
    def test_connected_components(self, small_tissue):
        """Test connected component detection."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        components = connected_components(graph.G)
        
        assert isinstance(components, list)
        # Union of components should be all nodes
        all_nodes = set()
        for comp in components:
            all_nodes.update(comp)
        assert len(all_nodes) <= graph.n_nodes
    
    def test_n_connected_components(self, small_tissue):
        """Test counting connected components."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        n_comp = n_connected_components(graph.G)
        
        assert isinstance(n_comp, int)
        assert n_comp >= 1
        assert n_comp <= graph.n_nodes
    
    def test_largest_component_size(self, small_tissue):
        """Test largest component size."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        size = largest_component_size(graph.G)
        
        assert isinstance(size, int)
        assert 1 <= size <= graph.n_nodes
    
    def test_bridges(self, small_tissue):
        """Test bridge detection."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        bridge_edges = bridges(graph.G)
        
        assert isinstance(bridge_edges, list)
        # Bridges should be subset of edges
        assert len(bridge_edges) <= graph.n_edges
    
    def test_articulation_points(self, small_tissue):
        """Test articulation point detection."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        art_points = articulation_points(graph.G)
        
        assert isinstance(art_points, list)
        # Articulation points should be subset of nodes
        assert len(art_points) <= graph.n_nodes


# =============================================================================
# Communicability Tests
# =============================================================================

class TestCommunicability:
    """Tests for communicability metrics."""
    
    def test_communicability_basic(self, small_tissue):
        """Test communicability calculation."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        # Only test on small graphs (computationally expensive)
        if graph.n_nodes > 50:
            pytest.skip("Graph too large for communicability")
        
        comm = communicability(graph.G)
        
        assert isinstance(comm, dict)
        # Diagonal elements should be largest
        for node in graph.G.nodes():
            if node in comm:
                assert comm[node][node] >= max(
                    (comm[node][j] for j in comm[node] if j != node),
                    default=0
                )
    
    def test_communicability_between_types(self, small_tissue):
        """Test communicability between cell types."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        if len(graph.cell_types_unique) < 2:
            pytest.skip("Need multiple cell types")
        
        type_a = graph.cell_types_unique[0]
        type_b = graph.cell_types_unique[1]
        
        if graph.n_nodes > 50:
            pytest.skip("Graph too large")
        
        comm = communicability_between_types(graph, type_a, type_b)
        
        assert isinstance(comm, float)
        assert comm >= 0
    
    def test_shortest_path_length_basic(self, small_tissue):
        """Test shortest path lengths."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        if len(graph.cell_types_unique) < 2:
            pytest.skip("Need multiple cell types")
        
        type_a = graph.cell_types_unique[0]
        type_b = graph.cell_types_unique[1]
        
        result = shortest_path_length_between_types(graph, type_a, type_b)
        
        assert 'mean' in result
        assert 'median' in result
        assert 'min' in result
        assert 'max' in result
        
        if result['mean'] is not None:
            assert result['mean'] >= 0
    
    def test_average_shortest_path_length(self, small_tissue):
        """Test average shortest path length."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        try:
            avg_path = average_shortest_path_length(graph.G)
            assert isinstance(avg_path, float)
            assert avg_path >= 0
        except nx.NetworkXError:
            # Graph is not connected
            pytest.skip("Graph is not connected")
    
    def test_diameter(self, small_tissue):
        """Test graph diameter."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        try:
            diam = diameter(graph.G)
            assert isinstance(diam, int)
            assert diam >= 0
        except nx.NetworkXError:
            # Graph is not connected
            pytest.skip("Graph is not connected")
    
    def test_global_efficiency(self, small_tissue):
        """Test global efficiency."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        eff = global_efficiency(graph.G)
        
        assert isinstance(eff, float)
        assert 0 <= eff <= 1
    
    def test_local_efficiency(self, small_tissue):
        """Test local efficiency."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        eff = local_efficiency(graph.G)
        
        assert isinstance(eff, float)
        assert 0 <= eff <= 1


# =============================================================================
# Assortativity Tests
# =============================================================================

class TestAssortativity:
    """Tests for assortativity and mixing."""
    
    def test_degree_assortativity_basic(self, small_tissue):
        """Test degree assortativity."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        assort = degree_assortativity(graph.G)
        
        assert isinstance(assort, float)
        assert -1 <= assort <= 1
    
    def test_type_assortativity_basic(self, small_tissue):
        """Test cell type assortativity."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        assort = type_assortativity(graph)
        
        assert isinstance(assort, float)
        assert -1 <= assort <= 1
    
    def test_attribute_mixing_matrix(self, small_tissue):
        """Test mixing matrix calculation."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        matrix = attribute_mixing_matrix(graph)
        
        n_types = len(graph.cell_types_unique)
        assert matrix.shape == (n_types, n_types)
        # Matrix should sum to 1
        assert abs(matrix.values.sum() - 1.0) < 0.01
    
    def test_homophily_ratio(self, small_tissue):
        """Test homophily ratio."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        homophily = homophily_ratio(graph)
        
        assert isinstance(homophily, float)
        assert 0 <= homophily <= 1
    
    def test_heterophily_ratio(self, small_tissue):
        """Test heterophily ratio."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        heterophily = heterophily_ratio(graph)
        
        assert isinstance(heterophily, float)
        assert 0 <= heterophily <= 1
    
    def test_homophily_heterophily_sum(self, small_tissue):
        """Test homophily + heterophily = 1."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        homophily = homophily_ratio(graph)
        heterophily = heterophily_ratio(graph)
        
        assert abs(homophily + heterophily - 1.0) < 0.01
    
    def test_average_neighbor_degree(self, small_tissue):
        """Test average neighbor degree."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        avg_deg = average_neighbor_degree(graph.G)
        
        assert isinstance(avg_deg, dict)
        assert len(avg_deg) == graph.n_nodes
        assert all(v >= 0 for v in avg_deg.values())
    
    def test_neighbor_type_distribution(self, small_tissue):
        """Test neighbor type distribution."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        cell_type = graph.cell_types_unique[0]
        dist = neighbor_type_distribution(graph, cell_type=cell_type)
        
        assert isinstance(dist, dict)
        # Should sum to 1
        assert abs(sum(dist.values()) - 1.0) < 0.01


# =============================================================================
# Integration Tests
# =============================================================================

class TestNetworkIntegration:
    """Integration tests for network module."""
    
    def test_complete_network_workflow(self, medium_tissue):
        """Test complete network analysis workflow."""
        # 1. Build graph
        graph = CellGraph.from_spatial_data(
            medium_tissue,
            method='proximity',
            radius=50
        )
        assert graph.n_nodes == medium_tissue.n_cells
        
        # 2. Compute centrality
        degree_cent = centrality_by_type(graph, metric='degree')
        assert len(degree_cent) == len(graph.cell_types_unique)
        
        # 3. Compute clustering
        avg_clust = average_clustering(graph.G)
        assert 0 <= avg_clust <= 1
        
        # 4. Analyze assortativity
        type_assort = type_assortativity(graph)
        assert -1 <= type_assort <= 1
        
        # 5. Mixing matrix
        mixing = attribute_mixing_matrix(graph)
        assert mixing.shape[0] == len(graph.cell_types_unique)
    
    def test_graph_method_comparison(self, small_tissue):
        """Test different graph construction methods."""
        methods = ['proximity', 'knn', 'delaunay']
        graphs = {}
        
        for method in methods:
            if method == 'proximity':
                graphs[method] = CellGraph.from_spatial_data(
                    small_tissue, method=method, radius=50
                )
            elif method == 'knn':
                graphs[method] = CellGraph.from_spatial_data(
                    small_tissue, method=method, k=5
                )
            elif method == 'delaunay':
                graphs[method] = CellGraph.from_spatial_data(
                    small_tissue, method=method
                )
        
        # All should produce valid graphs
        for method, graph in graphs.items():
            assert graph.n_nodes == small_tissue.n_cells
            assert graph.n_edges > 0
    
    def test_subgraph_analysis(self, small_tissue):
        """Test analyzing subgraphs."""
        graph = CellGraph.from_spatial_data(small_tissue, radius=50)
        
        if len(graph.cell_types_unique) < 2:
            pytest.skip("Need multiple cell types")
        
        # Extract subgraph
        cell_type = graph.cell_types_unique[0]
        subgraph = graph.subgraph_by_type(cell_type)
        
        # Analyze subgraph
        degree_cent = degree_centrality(subgraph.G)
        assert len(degree_cent) == subgraph.n_nodes
        
        avg_clust = average_clustering(subgraph.G)
        assert 0 <= avg_clust <= 1


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestNetworkEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_graph(self):
        """Test with empty graph."""
        G = nx.Graph()
        
        centrality = degree_centrality(G)
        assert len(centrality) == 0
        
        avg_clust = average_clustering(G)
        assert avg_clust == 0.0
    
    def test_single_node_graph(self):
        """Test with single node."""
        coords = np.array([[0, 0]])
        types = np.array(['A'])
        
        graph = CellGraph.from_coordinates(coords, types, radius=50)
        
        assert graph.n_nodes == 1
        assert graph.n_edges == 0
    
    def test_disconnected_graph(self):
        """Test with disconnected components."""
        # Create two separate clusters
        coords = np.vstack([
            np.random.uniform([0, 0], [10, 10], (20, 2)),
            np.random.uniform([100, 100], [110, 110], (20, 2))
        ])
        types = np.array(['A'] * 40)
        
        graph = CellGraph.from_coordinates(coords, types, radius=5)
        
        n_comp = n_connected_components(graph.G)
        assert n_comp >= 2
    
    def test_fully_connected_graph(self):
        """Test with fully connected graph."""
        coords = np.random.uniform(0, 10, (10, 2))
        types = np.array(['A'] * 10)
        
        graph = CellGraph.from_coordinates(coords, types, radius=100)
        
        # Should be fully connected
        expected_edges = 10 * 9 / 2  # Complete graph
        assert graph.n_edges == expected_edges


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestNetworkPerformance:
    """Performance tests for network operations."""
    
    def test_graph_construction_performance(self, large_tissue):
        """Test graph construction with 10k cells."""
        import time
        
        start = time.time()
        graph = CellGraph.from_spatial_data(
            large_tissue,
            method='proximity',
            radius=50
        )
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 10.0
        assert graph.n_nodes == large_tissue.n_cells
    
    def test_centrality_performance(self, large_tissue):
        """Test centrality computation performance."""
        import time
        
        graph = CellGraph.from_spatial_data(large_tissue, radius=50)
        
        start = time.time()
        centrality = degree_centrality(graph.G)
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 1.0
        assert len(centrality) == graph.n_nodes
    
    def test_clustering_performance(self, large_tissue):
        """Test clustering computation performance."""
        import time
        
        graph = CellGraph.from_spatial_data(large_tissue, radius=50)
        
        start = time.time()
        avg_clust = average_clustering(graph.G)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0
        assert isinstance(avg_clust, float)
