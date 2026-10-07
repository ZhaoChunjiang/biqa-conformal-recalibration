import math

ALPHA = 0.10

def interval_score(y, L, U, alpha=ALPHA):
    score = (U - L)
    if y < L:
        score += (2.0/alpha)*(L-y)
    elif y > U:
        score += (2.0/alpha)*(y-U)
    return score

def finite_sample_rank(n, alpha=ALPHA):
    return min(n, max(1, math.ceil((n+1)*(1-alpha))))

def test_full_range_score_is_one():
    assert abs(interval_score(0.5, 0.0, 1.0)-1.0) < 1e-12

def test_clipping_preserves_coverage_for_bounded_mos():
    y=0.10
    rawL, rawU = -0.20, 0.30
    L, U = max(0,rawL), min(1,rawU)
    assert (rawL <= y <= rawU) == (L <= y <= U)

def test_q40_rank():
    assert finite_sample_rank(40, 0.10) == 37

if __name__ == "__main__":
    test_full_range_score_is_one()
    test_clipping_preserves_coverage_for_bounded_mos()
    test_q40_rank()
    print("All reproducibility unit tests passed.")
