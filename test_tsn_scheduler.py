import unittest

from tsn_sim.candidate import build_candidates, instantiate_packets
from tsn_sim.config import HeuristicConfig, default_config
from tsn_sim.experiment import compare_heuristics, compare_with_milp
from tsn_sim.heuristics import schedule_with_heuristic
from tsn_sim.milp import solve_optimal_milp
from tsn_sim.models import Flow, Scenario
from tsn_sim.scenario import build_scenario


class TsnHeuristicSchedulerTest(unittest.TestCase):
    def test_default_scenario_schedules_packets_and_meets_deadlines(self):
        scenario = build_scenario(default_config(flow_count=10, seed=7))
        result = schedule_with_heuristic(scenario)

        self.assertIn(result.status, {"feasible", "partial"})
        self.assertGreater(result.metrics["scheduled_packet_count"], 0)
        # scheduling_success_rate = 1.0 when all packets are scheduled
        self.assertEqual(result.metrics["scheduling_success_rate"], 1.0)
        # true_deadline_satisfaction_rate checks scheduled packets only
        self.assertEqual(result.metrics["true_deadline_satisfaction_rate"], 1.0)
        # backward compat alias should match
        self.assertEqual(result.metrics["deadline_satisfaction_ratio"],
                         result.metrics["scheduling_success_rate"])
        # objective should equal effective_peak_load (peak + drop penalty)
        self.assertAlmostEqual(result.objective, result.metrics["effective_peak_load"], places=6)

        packet_keys = [(item.flow_id, item.packet_index) for item in result.schedule]
        self.assertEqual(len(packet_keys), len(set(packet_keys)))

    def test_candidate_generation_applies_c4_deadline_filter(self):
        flows = (
            Flow(
                flow_id=0,
                period_ms=4,
                packet_size_bits=100,
                deadline_ms=1.0,
                available_links=(1,),
                offset_ms=0.0,
                user_class="close",
            ),
        )
        scenario = Scenario(
            flows=flows,
            links=(1,),
            rb_capacity={(1, slot): 10 for slot in range(4)},
            rb_bits={(0, 1, slot): 100 for slot in range(4)},
            hyperperiod_ms=4,
            slot_ms=1.0,
        )
        packets = instantiate_packets(scenario)
        candidates, infeasible = build_candidates(scenario, packets)

        self.assertFalse(infeasible)
        self.assertEqual([candidate.slot for candidate in candidates[(0, 0)]], [0])

    def test_resource_tight_scenario_reports_unscheduled_packets(self):
        scenario = build_scenario(default_config(flow_count=10, seed=7))
        tight = Scenario(
            flows=scenario.flows,
            links=scenario.links,
            rb_capacity={cell: 1 for cell in scenario.rb_capacity},
            rb_bits=scenario.rb_bits,
            hyperperiod_ms=scenario.hyperperiod_ms,
            slot_ms=scenario.slot_ms,
        )

        result = schedule_with_heuristic(tight)

        self.assertEqual(result.status, "partial")
        self.assertGreater(result.metrics["infeasible_packet_count"], 0.0)

    def test_more_links_do_not_increase_drop_ratio_on_same_seed(self):
        # Use the deterministic greedy scheduler: more resources should
        # monotonically reduce the drop ratio. (The random_feasible baseline
        # is stochastic and does not satisfy this physical property.)
        one_link = schedule_with_heuristic(
            build_scenario(default_config(seed=11, link_count=1)), "greedy", seed=11
        )
        two_links = schedule_with_heuristic(
            build_scenario(default_config(seed=11, link_count=2)), "greedy", seed=11
        )
        three_links = schedule_with_heuristic(
            build_scenario(default_config(seed=11, link_count=3)), "greedy", seed=11
        )

        self.assertLessEqual(two_links.metrics["drop_ratio"], one_link.metrics["drop_ratio"] + 1e-9)
        self.assertLessEqual(three_links.metrics["drop_ratio"], two_links.metrics["drop_ratio"] + 1e-9)

    def test_scheduler_prefers_better_channel_when_capacity_is_equal(self):
        flows = (
            Flow(
                flow_id=0,
                period_ms=4,
                packet_size_bits=1000,
                deadline_ms=1.0,
                available_links=(1, 2),
                offset_ms=0.0,
                user_class="close",
            ),
        )
        links = (1, 2)
        scenario = Scenario(
            flows=flows,
            links=links,
            rb_capacity={(link, slot): 10 for link in links for slot in range(4)},
            rb_bits={
                (0, link, slot): (100 if link == 1 else 1000)
                for link in links
                for slot in range(4)
            },
            hyperperiod_ms=4,
            slot_ms=1.0,
        )

        # Use greedy: it prefers the channel with higher bits_per_rb
        # (fewer RBs needed), which is the property under test.
        result = schedule_with_heuristic(scenario, "greedy", seed=0)

        self.assertEqual(result.status, "feasible")
        self.assertEqual(result.schedule[0].link, 2)
        self.assertEqual(result.schedule[0].required_rb, 1)

    def test_all_heuristic_strategies_run(self):
        scenario = build_scenario(default_config(flow_count=10, seed=7))

        for strategy in ("random_feasible", "greedy", "proportional_fair", "genetic_algorithm"):
            with self.subTest(strategy=strategy):
                result = schedule_with_heuristic(scenario, strategy, seed=7)
                self.assertIn(result.status, {"feasible", "partial"})
                self.assertGreater(result.metrics["scheduled_packet_count"], 0)

    def test_milp_solves_small_scenario(self):
        scenario = build_scenario(default_config(flow_count=5, seed=7))
        result = solve_optimal_milp(scenario, time_limit_s=10.0)

        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.metrics["drop_ratio"], 0.0)
        # With 0% drop, effective_peak_load == peak_load
        self.assertAlmostEqual(result.objective, result.metrics["effective_peak_load"], places=6)
        self.assertAlmostEqual(result.objective, result.metrics["peak_load"], places=6)

    def test_experiment_tables_include_baseline_and_gap(self):
        config = default_config(flow_count=5, seed=7)

        heuristic_rows = compare_heuristics(config)
        milp_rows = compare_with_milp(config, time_limit_s=10.0)

        self.assertEqual(len(heuristic_rows), 4)
        self.assertEqual(milp_rows[0]["solver"], "milp_optimal")
        self.assertIn("gap_to_milp", milp_rows[-1])

    def test_heuristic_strategy_aliases_resolve(self):
        from tsn_sim.heuristics import resolve_strategy
        self.assertEqual(resolve_strategy("pf"), "proportional_fair")
        self.assertEqual(resolve_strategy("ga"), "genetic_algorithm")
        self.assertEqual(resolve_strategy("random_feasible"), "random_feasible")
        with self.assertRaises(ValueError):
            resolve_strategy("edf_min_load")

    def test_proportional_fair_and_ga_produce_valid_schedules(self):
        scenario = build_scenario(default_config(flow_count=10, seed=7))
        pf = schedule_with_heuristic(scenario, "proportional_fair", seed=7)
        ga = schedule_with_heuristic(
            scenario, "genetic_algorithm", seed=7,
            population_size=10, generations=5,
        )
        for result in (pf, ga):
            self.assertIn(result.status, {"feasible", "partial"})
            self.assertGreater(result.metrics["scheduled_packet_count"], 0)


if __name__ == "__main__":
    unittest.main()
