
import itertools
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import value

def run_self_optimizing_model(build_model, disturbances, user_designs, metrics):
    """
    Run the self‐optimizing Control model over a set of disturbance scenarios and user designs.

    Args:
        build_model (callable): Function that returns a fresh Pyomo ConcreteModel.
        disturbances (dict): Mapping parameter names to lists of values to test.
        user_designs (dict): Mapping design labels to functions that apply constraints to a model.
        metrics (dict): Mapping metric names to callables (model, scenario) → float.

    Returns:
        results_df (pd.DataFrame):
            DataFrame of optimal metric values for each scenario (rows=metrics, cols=scenarios).
        loss_matrix (pd.DataFrame):
            DataFrame of losses for each design under each scenario,
            with an average and ranking (feasible designs only).
    """
    # 1. Generate all combos of disturbances, but only single‑factor changes
    params   = list(disturbances.keys())
    all_combos = [
        dict(zip(params, vals))
        for vals in itertools.product(*(disturbances[p] for p in params))
    ]
    nominal = all_combos[0]
    combos  = [c for c in all_combos if sum(c[p] != nominal[p] for p in params) <= 1]

    # 2. Name each scenario (e.g. "Nominal", "F=0.7", ...)
    scenario_names = []
    for c in combos:
        diffs = [(p,c[p]) for p in params if c[p] != nominal[p]]
        scenario_names.append("Nominal" if not diffs else f"{diffs[0][0]}={diffs[0][1]}")

    solver = pyo.SolverFactory('ipopt')
    # 3. Solve for base objective J under each scenario
    J_list = []
    for c in combos:
        m = build_model()
        for p,val in c.items(): getattr(m,p).set_value(val)
        solver.solve(m, tee=False)
        J_list.append(value(m.J))

    # 4. Collect controlled CV metrics under each scenario
    results = []
    for c in combos:
        m = build_model()
        for p,val in c.items(): getattr(m,p).set_value(val)
        solver.solve(m, tee=False)
        results.append([fn(m,c) for fn in metrics.values()])

    results_df = pd.DataFrame(results, index=scenario_names, columns=metrics.keys()).T.round(4)

    # 5. Build loss matrix for each user design
    col_labels = [f"Loss with {lbl}" for lbl in user_designs]
    loss_matrix = pd.DataFrame(index=scenario_names, columns=col_labels, dtype=object)

    for lbl, apply in user_designs.items():
        col = f"Loss with {lbl}"
        for i,(name,c) in enumerate(zip(scenario_names,combos)):
            J_nom = J_list[i]
            m = build_model()
            apply(m)                                  # impose design constraint
            for p,val in c.items(): getattr(m,p).set_value(val)
            try:
                solver.solve(m, tee=False)
                loss = round(J_nom - value(m.J), 2)
            except:
                loss = np.nan
            loss_matrix.at[name,col] = loss

    # 6. Mark negative‐loss designs as infeasible
    for col in loss_matrix:
        if any(pd.to_numeric(loss_matrix[col], errors='coerce') < 0):
            loss_matrix[col] = 'infeasible'

    # 7. Compute average loss & ranking (infeasible last)
    numeric = loss_matrix.replace('infeasible', np.nan).apply(pd.to_numeric)
    avg_loss = numeric.mean()
    infeas   = numeric.isnull().any()
    rank_vec = avg_loss.copy()
    max_ok   = rank_vec[~infeas].max()
    rank_vec[infeas] = max_ok + 1
    rankings = rank_vec.rank(method='dense', ascending=True).astype(int)

    loss_matrix.loc['Average loss'] = avg_loss.round(2)
    loss_matrix.loc['Ranking']      = rankings

    return results_df, loss_matrix
