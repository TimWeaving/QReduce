from utils import adjacency_matrix
from cs_vqe_tools_legacy import greedy_dfs
import numpy as np
from typing import Dict, List

class cs_vqe_model:
    """
    """
    def __init__(self, 
                ham: Dict[str,float],
                num_qubits: int,
                set_nc: List[str] = None
                ) -> None:
        self.ham = ham
        self.num_qubits = num_qubits
        if set_nc is not None:
            self.set_nc = set_nc
        else:
            self.set_nc = self.find_noncontextual_set()


    def find_noncontextual_set(self, search_time=3):
        """Method for extracting a noncontextual subset of the hamiltonian terms
        """
        # for now uses the legacy greedy DFS approach
        # to be updated once more efficient/effective methods are identified
        set_nc = greedy_dfs(self.ham, 
                            cutoff=search_time, 
                            criterion='weight')[1]
        return set_nc


    def decompose_set_nc(self):
        """Decompose a noncontextual set into its symmetry and
        remaining pairwise anticommuting cliques
        """
        commutation_matrix = adjacency_matrix(self.set_nc, self.num_qubits) == 0
        symmetry_indices=[]
        for index,commutes_with in enumerate(commutation_matrix):
            if np.all(commutes_with):
                symmetry_indices.append(index)

        cliques = []
        for i,commute_list in enumerate(commutation_matrix):
            if i not in symmetry_indices:
                C = frozenset([self.set_nc[j] for j,commutes in enumerate(np.array(commute_list)[0]) if commutes and j not in symmetry_indices])
                cliques.append(C)

        symmetry = [self.set_nc[i] for i in symmetry_indices]
        cliques = [list(C) for C in set(cliques)]
        
        return symmetry, cliques
        

if __name__ == "__main__":
    ham={"IIIII": -8.672007378626361, "ZIIII": -0.2968649524871013, "IZIII": -0.2968649524871013, "ZZIII": 0.3892355930337426, "IIZII": -0.2968649524871013, "ZIZII": 0.3892355930337426, "IZZII": 0.3892355930337426, "ZZZII": 0.26346920025692516, "IIIZI": -0.04669169228823615, "ZIIZI": 0.18309620710706126, "IZIZI": 0.18309620710706126, "ZZIZI": 0.28071464858438355, "IIZZI": 0.18309620710706126, "ZIZZI": 0.28071464858438355, "IZZZI": 0.28071464858438355, "ZZZZI": 2.243229244264402, "IIIIZ": -0.04669169228823542, "ZIIIZ": 0.18309620710706126, "IZIIZ": 0.18309620710706126, "ZZIIZ": 0.28071464858438355, "IIZIZ": 0.18309620710706126, "ZIZIZ": 0.28071464858438355, "IZZIZ": 0.28071464858438355, "ZZZIZ": 2.243229244264402, "IIIZZ": 0.669867864503426, "ZZZZZ": 0.2856181541250911, "XIIII": 0.0038997696437550566, "XZZZI": -0.0038997696437550566, "XZZIZ": -0.0038997696437550566, "XIIZZ": 0.0038997696437550566, "IXIII": 0.0038997696437550566, "ZXZZI": -0.0038997696437550566, "ZXZIZ": -0.0038997696437550566, "IXIZZ": 0.0038997696437550566, "XXIII": 0.012124689610585571, "YYIII": 0.012124689610585571, "IIXII": 0.0038997696437550566, "ZZXZI": -0.0038997696437550566, "ZZXIZ": -0.0038997696437550566, "IIXZZ": 0.0038997696437550566, "XIXII": 0.012124689610585571, "YIYII": 0.012124689610585571, "IXXII": 0.012124689610585571, "IYYII": 0.012124689610585571, "IIIXI": 0.03806162406796882, "ZIIXI": 0.0096106136829005, "IZIXI": 0.0096106136829005, "ZZIXI": -0.0096106136829005, "IIZXI": 0.0096106136829005, "ZIZXI": -0.0096106136829005, "IZZXI": -0.0096106136829005, "ZZZXI": -0.03806162406796882, "IIIXZ": -0.05977158959004002, "ZZZXZ": 0.05977158959004002, "XZZXI": -0.005140642807413051, "YZZYI": -0.005140642807413051, "XIIXZ": 0.005140642807413051, "YIIYZ": 0.005140642807413051, "ZXZXI": -0.005140642807413051, "ZYZYI": -0.005140642807413051, "IXIXZ": 0.005140642807413051, "IYIYZ": 0.005140642807413051, "ZZXXI": -0.005140642807413051, "ZZYYI": -0.005140642807413051, "IIXXZ": 0.005140642807413051, "IIYYZ": 0.005140642807413051, "IIIIX": 0.038061624067968855, "ZIIIX": 0.0096106136829005, "IZIIX": 0.0096106136829005, "ZZIIX": -0.0096106136829005, "IIZIX": 0.0096106136829005, "ZIZIX": -0.0096106136829005, "IZZIX": -0.0096106136829005, "ZZZIX": -0.038061624067968855, "IIIZX": -0.05977158959004002, "ZZZZX": 0.05977158959004002, "XZZIX": -0.005140642807413051, "XIIZX": 0.005140642807413051, "YZZIY": -0.005140642807413051, "YIIZY": 0.005140642807413051, "ZXZIX": -0.005140642807413051, "IXIZX": 0.005140642807413051, "ZYZIY": -0.005140642807413051, "IYIZY": 0.005140642807413051, "ZZXIX": -0.005140642807413051, "IIXZX": 0.005140642807413051, "ZZYIY": -0.005140642807413051, "IIYZY": 0.005140642807413051, "IIIXX": 0.022148953868166, "ZZZXX": -0.022148953868166, "XIIXX": 0.021702244865380426, "YIIYX": 0.021702244865380426, "YIIXY": 0.021702244865380426, "XIIYY": -0.021702244865380426, "IXIXX": 0.021702244865380426, "IYIYX": 0.021702244865380426, "IYIXY": 0.021702244865380426, "IXIYY": -0.021702244865380426, "IIXXX": 0.021702244865380426, "IIYYX": 0.021702244865380426, "IIYXY": 0.021702244865380426, "IIXYY": -0.021702244865380426}
    cs_vqe_mol=cs_vqe_model(ham, 5)
    print(cs_vqe_mol.decompose_set_nc())
    print(len(cs_vqe_mol.set_nc))