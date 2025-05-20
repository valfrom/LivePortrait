from profiling_graph import parse_profile_table, plot_profile

SAMPLE_TABLE = """
Name    Self CPU total %  Self CPU total  CPU total %  CPU total
aten::mm       50.00%      1.000ms        50.00%       1.000ms
aten::addmm    25.00%      0.500ms        25.00%       0.500ms
"""

def test_parse_profile_table():
    data = parse_profile_table(SAMPLE_TABLE)
    assert data == [("aten::mm", 1.0), ("aten::addmm", 0.5)]


def test_plot_profile(tmp_path):
    out_file = tmp_path / "out.png"
    plot_profile(parse_profile_table(SAMPLE_TABLE), out_file)
    assert out_file.exists()
