import torch
from src.utils.profiler import profile_operations


def test_profile_operations_returns_table():
    def ops():
        device = (
            'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        )
        x = torch.randn(16, 16, device=device)
        for _ in range(3):
            x = torch.relu(torch.mm(x, x))
        return x

    table = profile_operations(ops, row_limit=5)
    assert isinstance(table, str)
    assert "aten::" in table
