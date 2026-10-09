import sys
sys.path.insert(0, '.')

from tests.test_backtest import (
    test_engine_toy_example,
    test_engine_no_signals,
    test_engine_higher_costs_lower_final_equity,
    test_engine_no_lookahead,
    test_metrics_calculation
)


def run_all_tests():
    test_engine_toy_example()
    print("test_engine_toy_example passed")
    test_engine_no_signals()
    print("test_engine_no_signals passed")
    test_engine_higher_costs_lower_final_equity()
    print("test_engine_higher_costs_lower_final_equity passed")
    test_engine_no_lookahead()
    print("test_engine_no_lookahead passed")
    test_metrics_calculation()
    print("test_metrics_calculation passed")


if __name__ == "__main__":
    run_all_tests()
    print("All tests passed.")