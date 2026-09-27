"""Run: python scenario_check.py. Demonstrates allocation, never fits stop truth."""
from fractions import Fraction


def allocate(total, weights):
    """Integer largest remainders; stable input order breaks exact ties."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a nonnegative integer")
    if not weights:
        raise ValueError("eligible stop list is required")
    weights = [Fraction(str(w)) for w in weights]
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("nonnegative finite weights with positive sum required")
    quotas = [total * w / sum(weights) for w in weights]
    result = [q.numerator // q.denominator for q in quotas]
    remaining = total - sum(result)
    for i in sorted(range(len(quotas)), key=lambda i: (-(quotas[i]-result[i]), i))[:remaining]:
        result[i] += 1
    return result


def check():
    # One route total gives no way to distinguish two possible stop truths.
    first, second = [10, 20, 30], [30, 20, 10]
    assert first != second and sum(first) == sum(second) == 60
    assert allocate(2, [1, 1, 1]) == [1, 1, 0]  # Independent rounding gives 3.
    assert allocate(0, [1, 2]) == [0, 0]
    assert allocate(7, [0, 1, 2]) == [0, 2, 5]
    for total in range(101):
        for weights in ([1, 1, 1], [0, 1, 2], ["0.1", "0.2", "0.7"]):
            values = allocate(total, weights)
            assert sum(values) == total and min(values) >= 0
    for total, weights in ((-1, [1]), (1.5, [1]), (True, [1]), (1, []), (1, [0]), (1, [-1, 2])):
        try:
            allocate(total, weights)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid scenario accepted")
    print("PASS: distinct allocations share total; integer scenarios conserve every route hour")


if __name__ == "__main__":
    check()
