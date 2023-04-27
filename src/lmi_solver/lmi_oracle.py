import numpy as np
from .ldlt_mgr import LDLTMgr
from typing import Optional, Tuple

Cut = Tuple[np.ndarray, float]


class LMIOracle:
    """Oracle for Linear Matrix Inequality constraint.

    This oracle solves the following feasibility problem:

        find  x
        s.t.  (B − F * x) ⪰ 0

    """

    def __init__(self, F, B):
        """Construct a new lmi oracle object

        Arguments:
            F (List[np.ndarray]): [description]
            B (np.ndarray): [description]
        """
        self.F = F
        self.F0 = B
        self.Q = LDLTMgr(len(B))

    def assess_feas(self, x: np.ndarray) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (np.ndarray): [description]

        Returns:
            Optional[Cut]: [description]
        """

        def get_elem(i, j):
            return self.F0[i, j] - sum(
                Fk[i, j] * xk for Fk, xk in zip(self.F, x))

        if self.Q.factor(get_elem):
            return None
        ep = self.Q.witness()
        g = np.array([self.Q.sym_quad(Fk) for Fk in self.F])
        return g, ep
