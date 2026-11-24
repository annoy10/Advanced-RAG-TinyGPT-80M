import torch
from transformer.train_model import atomic_torch_save

def test_atomic_checkpoint_round_trip(tmp_path):
    path = tmp_path / "recovery.pt"
    payload = {"step": 123, "weights": torch.tensor([1.0, 2.0])}
    atomic_torch_save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["step"] == 123
    assert torch.equal(loaded["weights"], payload["weights"])
    assert not path.with_suffix(".pt.tmp").exists()
