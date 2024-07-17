class RefinedBigDag():
    def __init__(self, dict_dag):
        """
        """
        self.dict_num = {key: dag.shape[0] for key, dag in dict_dag.items()}
        list_accum_count = list(fun_accum_sum(self.dict_num))
        self.list_accum_count = [0] + list_accum_count[:-1]

    def get_global_ind(self, ind_macro_node, ind_local_node):
        """
        """
        return self.list_accum_count[ind_macro_node] + ind_local_node


def fun_accum_sum(dict_num):
    # dict_dag = {'0':3, '1':5, '2':4}
    count = 0
    for num in dict_num.values():
        count += num
        yield count
