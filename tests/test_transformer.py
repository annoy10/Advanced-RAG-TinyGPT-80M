from pathlib import Path
def test_checkpoint_exists():
    assert (Path(__file__).parents[1]/"transformer/checkpoint/tinygpt.pt").exists()
