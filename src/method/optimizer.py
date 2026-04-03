from .genetic_algorithm import choose_edges, genetic_algorithm_main
from .location_problem import block_coverage


def optimize_placement(
    matrix,
    df,
    service_radius,
    id_matrix,
    *,
    use_genetic=True,
    demand_column=None,
    population_size=50,
    num_generations=20,
    mutation_rate=0.7,
    num_parents=10,
    num_offspring=None,
    number_res="all",
    prefer_existing=False,
    existing_facility_discount=1.0,
    existing_column=None,
    keep_existing_capacity=False,
    allow_existing_expansion=True,
    min_new_capacity=50.0,
    heartbeat_interval_sec=None,
    verbose=False,
):
    best_candidate = matrix.copy()
    fitness_history = []
    edges = []

    if use_genetic:
        edges = choose_edges(sim_matrix=matrix, service_radius=service_radius)
        if num_offspring is None:
            num_offspring = population_size - num_parents
        best_candidate, fitness_history = genetic_algorithm_main(
            matrix=matrix,
            edges=edges,
            population_size=population_size,
            num_generations=num_generations,
            df=df,
            service_radius=service_radius,
            mutation_rate=mutation_rate,
            num_parents=num_parents,
            num_offspring=num_offspring,
            number_res=number_res,
            demand_column=demand_column,
            prefer_existing=prefer_existing,
            existing_facility_discount=existing_facility_discount,
            existing_column=existing_column,
        )

    capacities, res_id = block_coverage(
        best_candidate,
        service_radius,
        df,
        id_matrix,
        demand_column=demand_column,
        prefer_existing=prefer_existing,
        existing_facility_discount=existing_facility_discount,
        existing_column=existing_column,
        keep_existing_capacity=keep_existing_capacity,
        allow_existing_expansion=allow_existing_expansion,
        min_new_capacity=min_new_capacity,
        heartbeat_interval_sec=heartbeat_interval_sec,
        verbose=verbose,
    )

    return {
        "best_candidate": best_candidate,
        "fitness_history": fitness_history,
        "edges": edges,
        "capacities": capacities,
        "res_id": res_id,
        "use_genetic": use_genetic,
        "demand_column": demand_column,
        "prefer_existing": prefer_existing,
        "existing_facility_discount": existing_facility_discount,
        "keep_existing_capacity": keep_existing_capacity,
        "allow_existing_expansion": allow_existing_expansion,
        "min_new_capacity": min_new_capacity,
    }
