# class structures
- dag_interface.py: abstract def for DAG
- dag_gen: PLP^T
- dag_manipulator.py: friend class to DAG class to enforce one node to be a confounder (i.e. if this
node only has one child, we connect another arrow to another node to make two children)
- dag_stack_indexer.py: block-diagnize stack different rectangles(macro-node)
- dag_viewer.py: only show the marginal dag
- data_gen.py: breadth-first-search data generation according to top order
- edge_models.py: f(x|pa(x))
- gen_dag_2level.py: bi-level DAG generation
- is_dag: use breath-first-search to check if there is loop
- noise_idiosyncratic.py: generate idosyncratic noise
- utils_topological_sort.py: do top sorting
- weight.py:  generating weight from different distributions: e.g. wishart
- wishart.py: p.d.f. that can generate a p.d. matrix
- main.py: defined user api
