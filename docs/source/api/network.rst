=======
Network
=======

.. module:: spatialtissuepy.network

The network module provides graph-based analysis of spatial tissue data.

.. note::

   This module requires NetworkX. Install with:

   .. code-block:: bash

       pip install networkx


CellGraph Class
---------------

.. autoclass:: spatialtissuepy.network.CellGraph
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__


Graph Construction
------------------

.. autofunction:: spatialtissuepy.network.build_graph

.. autofunction:: spatialtissuepy.network.proximity_graph

.. autofunction:: spatialtissuepy.network.knn_graph

.. autofunction:: spatialtissuepy.network.delaunay_graph


Centrality Measures
-------------------

.. autofunction:: spatialtissuepy.network.degree_centrality

.. autofunction:: spatialtissuepy.network.betweenness_centrality

.. autofunction:: spatialtissuepy.network.closeness_centrality

.. autofunction:: spatialtissuepy.network.eigenvector_centrality


Assortativity
-------------

.. autofunction:: spatialtissuepy.network.type_assortativity

.. autofunction:: spatialtissuepy.network.attribute_mixing_matrix


Module Contents
---------------

.. automodule:: spatialtissuepy.network
   :members:
   :undoc-members:
   :show-inheritance:
