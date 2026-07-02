from __future__ import annotations

import argparse
from pathlib import Path


EXPERIMENTS = {
    "residual_atlas": "src3_explore.diagnostics.residual_atlas",
    "model_disagreement": "src3_explore.diagnostics.model_disagreement",
    "green_red_transfer": "src3_explore.diagnostics.green_red_transfer_analysis",
    "curve_dictionary": "src3_explore.representations.curve_dictionary",
    "day_embedding": "src3_explore.representations.day_embedding_clustering",
    "route_arrival_kernel": "src3_explore.mechanisms.route_arrival_kernel",
    "tollgate12_allocation": "src3_explore.mechanisms.tollgate12_allocation",
    "etc_component_model": "src3_explore.mechanisms.etc_component_model",
    "quantile_baselines": "src3_explore.probabilistic.quantile_baselines",
    "conformal_intervals": "src3_explore.probabilistic.conformal_intervals",
    "adversarial_validation": "src3_explore.diagnostics.adversarial_validation",
    "explain_graph_signal_audit": "src3_explore.explain.graph_signal_audit",
    "explain_gnn_message_passing_damage": "src3_explore.explain.gnn_message_passing_damage",
    "explain_graph_randomization_test": "src3_explore.explain.graph_randomization_test",
    "explain_edge_harm_matrix": "src3_explore.explain.edge_harm_matrix",
    "explain_signed_edge_smoothing": "src3_explore.explain.signed_edge_smoothing",
    "explain_edge_dropout_sanity_check": "src3_explore.explain.edge_dropout_sanity_check",
    "explain_scale_normalized_message_passing": "src3_explore.explain.scale_normalized_message_passing",
    "explain_heterophily_audit": "src3_explore.explain.heterophily_audit",
    "explain_route_graph_replacement": "src3_explore.explain.route_graph_replacement",
    "explain_route_legal_decomposition": "src3_explore.explain.route_legal_decomposition",
    "explain_sequence_permutation_test": "src3_explore.explain.sequence_permutation_test",
    "explain_nn_representation_swap": "src3_explore.explain.nn_representation_swap",
    "explain_nn_prediction_collapse": "src3_explore.explain.nn_prediction_collapse",
    "explain_noise_robustness_test": "src3_explore.explain.noise_robustness_test",
    "explain_nn_failure_modes": "src3_explore.explain.nn_failure_modes",
    "explain_information_decomposition": "src3_explore.explain.information_decomposition",
    "explain_oracle_ensemble_gap": "src3_explore.explain.oracle_ensemble_gap",
    "explain_oracle_soft_gate_diagnostics": "src3_explore.explain.oracle_soft_gate_diagnostics",
}


GROUPS = {
    "explain_gnn": [
        "explain_graph_signal_audit",
        "explain_gnn_message_passing_damage",
        "explain_graph_randomization_test",
        "explain_edge_harm_matrix",
        "explain_signed_edge_smoothing",
        "explain_edge_dropout_sanity_check",
        "explain_scale_normalized_message_passing",
        "explain_heterophily_audit",
        "explain_route_graph_replacement",
        "explain_route_legal_decomposition",
    ],
    "explain_nn": [
        "explain_sequence_permutation_test",
        "explain_nn_representation_swap",
        "explain_nn_prediction_collapse",
        "explain_noise_robustness_test",
        "explain_nn_failure_modes",
    ],
    "explain_oracle": [
        "explain_oracle_ensemble_gap",
        "explain_oracle_soft_gate_diagnostics",
        "explain_information_decomposition",
    ],
}
GROUPS["explain_all"] = list(
    dict.fromkeys(GROUPS["explain_gnn"] + GROUPS["explain_nn"] + GROUPS["explain_oracle"])
)


def load_run(module_name: str):
    import importlib

    module = importlib.import_module(module_name)
    return module.run


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="src3 Task 2 interpretability exploration framework")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS) + sorted(GROUPS) + ["all", "list"])
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/src3_explore"))
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)

    if args.experiment == "list":
        print("[groups]")
        for name in sorted(GROUPS):
            print(name)
        print("[experiments]")
        for name in sorted(EXPERIMENTS):
            print(name)
        return

    if args.experiment == "all":
        selected = sorted(EXPERIMENTS)
    elif args.experiment in GROUPS:
        selected = GROUPS[args.experiment]
    else:
        selected = [args.experiment]
    for name in selected:
        print(f"running={name}")
        card = load_run(EXPERIMENTS[name])(args.data_dir, args.output_dir, args.force_cache)
        print(f"card={args.output_dir / 'cards'}")
        print(card.to_markdown())
    if args.experiment in GROUPS or args.experiment.startswith("explain_"):
        from src3_explore.explain.summary import write_explain_summary

        summary = write_explain_summary(args.output_dir)
        print(f"summary={summary}")


if __name__ == "__main__":
    main()
