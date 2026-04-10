import numpy as np
import pulp
import threading
import time
from tqdm import tqdm


def build_unmet_target_series(df):
    demand_left = None
    demand_without = None
    if "demand_left" in df.columns:
        demand_left = np.asarray(df["demand_left"], dtype=float)
    if "demand_without" in df.columns:
        demand_without = np.asarray(df["demand_without"], dtype=float)

    if demand_left is not None or demand_without is not None:
        if demand_left is None:
            demand_left = np.zeros(len(df), dtype=float)
        if demand_without is None:
            demand_without = np.zeros(len(df), dtype=float)
        target = demand_left + demand_without
        if float(np.nansum(target)) > 0.0:
            return target

    return None


def resolve_demand_series(df, demand_column=None):
    if demand_column is not None:
        return df[demand_column]

    unmet_target = build_unmet_target_series(df)
    if unmet_target is not None:
        return unmet_target

    for candidate in ("demand", "demand_without", "demand_left"):
        if candidate in df.columns:
            return df[candidate]

    raise KeyError(
        "Could not resolve demand column. Expected one of: "
        "'demand_without', 'demand_left', 'demand'."
    )


def resolve_existing_facility_mask(df, existing_column=None):
    if existing_column is not None:
        return np.asarray(df[existing_column], dtype=float) > 0

    for candidate in ("capacity", "capacity_left"):
        if candidate in df.columns:
            return np.asarray(df[candidate], dtype=float) > 0

    return None


def resolve_existing_capacity_series(df, existing_column=None):
    if existing_column is not None:
        return np.asarray(df[existing_column], dtype=float)

    for candidate in ("capacity", "capacity_left"):
        if candidate in df.columns:
            return np.asarray(df[candidate], dtype=float)

    return None

# 1. Создаем переменные для объектов (Y_j) и их вместимости (C_k)
def add_facility_variables(range_facility, var_name_y, var_name_c):
    y_vars = [pulp.LpVariable(var_name_y.format(i=i), lowBound=0, upBound=1, cat=pulp.LpInteger) for i in range_facility]
    c_vars = [pulp.LpVariable(var_name_c.format(i=i), lowBound=0, cat=pulp.LpInteger) for i in range_facility]
    return y_vars, c_vars

# 2. Создаем матрицу распределения спроса (Z_ij)
def add_assignment_variables(range_client, range_facility, var_name):
    return np.array([
        [pulp.LpVariable(var_name.format(i=i, j=j), lowBound=0, upBound=1, cat=pulp.LpContinuous) for j in range_facility]
        for i in range_client
    ])

# 3. Ограничение на вместимость объектов
def add_capacity_constraints(
    problem,
    y_vars,
    c_vars,
    z_vars,
    demand,
    range_client,
    range_facility,
    existing_capacity=None,
    min_new_capacity=50.0,
    fixed_new_capacity=None,
    progress=False,
):
    iterator = tqdm(
        range_facility,
        disable=(not progress),
        desc="[solver_flp] constraints: capacity",
        leave=False,
    )
    for j in iterator:
        problem += pulp.lpSum([demand[i] * z_vars[i, j] for i in range_client]) <= c_vars[j], f"capacity_constraint_{j}"
        problem += c_vars[j] <= y_vars[j] * 10000, f"open_capacity_constraint_{j}"
        # Minimum capacity is required only for newly opened facilities.
        # Existing facilities may have historical capacity below this threshold.
        if existing_capacity is None or float(existing_capacity[j]) <= 0.0:
            if fixed_new_capacity is not None:
                problem += c_vars[j] == y_vars[j] * float(fixed_new_capacity), f"fixed_capacity_constraint_{j}"
            else:
                problem += c_vars[j] >= y_vars[j] * float(min_new_capacity), f"min_capacity_constraint_{j}"
        if existing_capacity is not None and float(existing_capacity[j]) > 0.0:
            # In baseline-preserving mode we do not allow the optimizer to
            # silently remove already existing service capacity.
            problem += c_vars[j] >= float(existing_capacity[j]), f"existing_capacity_floor_{j}"

# 4. Ограничение на удовлетворение спроса
def add_demand_constraints(
    problem,
    z_vars,
    accessibility_matrix,
    range_client,
    range_facility,
    progress=False,
):
    iterator = tqdm(
        range_client,
        disable=(not progress),
        desc="[solver_flp] constraints: demand",
        leave=False,
    )
    for i in iterator:
        problem += pulp.lpSum([accessibility_matrix[i, j] * z_vars[i, j] for j in range_facility]) == 1, f"demand_constraint_{i}"

# Основная функция для решения объединенной задачи
def solve_combined_problem(
    cost_matrix,
    service_radius,
    demand_quantity,
    name="combined_problem",
    existing_facility_mask=None,
    existing_capacity=None,
    allow_existing_expansion=True,
    prefer_existing=False,
    existing_facility_discount=1.0,
    min_new_capacity=50.0,
    fixed_new_capacity=None,
    heartbeat_interval_sec=None,
    verbose=False,
):
    num_clients, num_facilities = cost_matrix.shape
    range_clients = range(num_clients)
    range_facilities = range(num_facilities)

    # Инициализация задачи минимизации
    problem = pulp.LpProblem(name, pulp.LpMinimize)

    # Матрица доступности (a_ij)
    accessibility_matrix = (cost_matrix <= service_radius).astype(int)

    # Переменные
    y_vars, c_vars = add_facility_variables(range_facilities, "y[{i}]", "c[{i}]")
    z_vars = add_assignment_variables(range_clients, range_facilities, "z[{i}_{j}]")

    # Целевая функция: минимизация количества объектов и общей вместимости.
    # При prefer_existing=True existing_facility_discount уменьшает штраф
    # открытия для уже существующих сервисных точек.
    w1, w2 = 1000, 1
    open_costs = np.ones(num_facilities, dtype=float)
    if prefer_existing and existing_facility_mask is not None:
        existing_facility_discount = float(existing_facility_discount)
        open_costs = np.where(
            np.asarray(existing_facility_mask, dtype=bool),
            np.maximum(0.0, 1.0 - existing_facility_discount),
            1.0,
        )
    problem += pulp.lpSum(
        [w1 * open_costs[j] * y_vars[j] + w2 * c_vars[j] for j in range_facilities]
    ), "objective_function"

    # Ограничения
    add_capacity_constraints(
        problem,
        y_vars,
        c_vars,
        z_vars,
        demand_quantity,
        range_clients,
        range_facilities,
        existing_capacity=existing_capacity,
        min_new_capacity=min_new_capacity,
        fixed_new_capacity=fixed_new_capacity,
        progress=verbose,
    )

    if existing_capacity is not None and not allow_existing_expansion:
        iterator = tqdm(
            range_facilities,
            disable=(not verbose),
            desc="[solver_flp] constraints: existing",
            leave=False,
        )
        for j in iterator:
            if float(existing_capacity[j]) > 0.0:
                # Baseline-preserving placement mode: existing services stay
                # exactly as they are, and the optimizer may only add new ones.
                problem += c_vars[j] == float(existing_capacity[j]), f"existing_capacity_fixed_{j}"
    add_demand_constraints(
        problem,
        z_vars,
        accessibility_matrix,
        range_clients,
        range_facilities,
        progress=verbose,
    )

    # Решение задачи
    solver = pulp.PULP_CBC_CMD(
        msg=False,
    )
    started = time.time()
    if heartbeat_interval_sec is None:
        heartbeat_interval_sec = 1.0

    result_box = {}
    error_box = {}
    done = threading.Event()

    def _solve():
        try:
            result_box["status"] = problem.solve(solver)
        except Exception as exc:  # noqa: BLE001
            error_box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_solve, daemon=True)
    thread.start()

    pulse = 0
    solve_progress = tqdm(
        total=0,
        disable=(not verbose),
        desc="[solver_flp] exact solve (running)",
        leave=False,
    )
    while not done.wait(float(heartbeat_interval_sec)):
        pulse += 1
        solve_progress.update(1)
    solve_progress.close()

    if "error" in error_box:
        raise error_box["error"]

    if verbose:
        print(
            "[solver_flp] exact solve setup "
            f"clients={num_clients} facilities={num_facilities} "
            f"variables={len(problem.variables())} constraints={len(problem.constraints)}",
            flush=True,
        )
        print(
            f"[solver_flp] exact solve finished in {time.time() - started:.1f}s "
            f"status={pulp.LpStatus[problem.status]}",
            flush=True,
        )

    if problem.status != 1:
        raise RuntimeError(f"Problem not solved: {pulp.LpStatus[problem.status]}.")

    fac2cli = []
    iterator = tqdm(
        range(len(y_vars)),
        disable=(not verbose),
        desc="[solver_flp] extract assignments",
        leave=False,
    )
    for j in iterator:
        if y_vars[j].value() > 0:
            fac_clients = [
                i
                for i in range(num_clients)
                if z_vars[i, j].value() is not None and z_vars[i, j].value() > 1e-9
            ]
            fac2cli.append(fac_clients)
        else:
            fac2cli.append([])

    # Формируем результаты
    facilities_open = [j for j in range_facilities if y_vars[j].value() > 0.5]
    assignment = np.array([[z_vars[i, j].value() for j in range_facilities] for i in range_clients])
    capacities = [c_vars[j].value() for j in range_facilities]

    return facilities_open, capacities, fac2cli

def block_coverage(
    matrix,
    SERVICE_RADIUS,
    df,
    id,
    demand_column=None,
    prefer_existing=False,
    existing_facility_discount=1.0,
    existing_column=None,
    keep_existing_capacity=False,
    allow_existing_expansion=True,
    min_new_capacity=50.0,
    fixed_new_capacity=None,
    heartbeat_interval_sec=None,
    verbose=False,
):
    demand_quantity = resolve_demand_series(df, demand_column=demand_column)
    existing_facility_mask = resolve_existing_facility_mask(df, existing_column=existing_column)
    existing_capacity = (
        resolve_existing_capacity_series(df, existing_column=existing_column)
        if keep_existing_capacity
        else None
    )
    facilities, capacities, fac2cli = solve_combined_problem(np.array(matrix),
                                                                SERVICE_RADIUS,
                                                                demand_quantity,
                                                                existing_facility_mask=existing_facility_mask,
                                                                existing_capacity=existing_capacity,
                                                                allow_existing_expansion=allow_existing_expansion,
                                                                prefer_existing=prefer_existing,
                                                                existing_facility_discount=existing_facility_discount,
                                                                min_new_capacity=min_new_capacity,
                                                                fixed_new_capacity=fixed_new_capacity,
                                                                heartbeat_interval_sec=heartbeat_interval_sec,
                                                                verbose=verbose)
    dict_info_hotels2 = dict([(k,l) for k,l in enumerate(fac2cli) if len(l)>0])
    res_id = {id[key]: [id[val] for val in value] for key, value in dict_info_hotels2.items()}

    return capacities, res_id
