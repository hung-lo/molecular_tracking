from pathlib import Path
import pytest
from project_cli import resolve_selection

def test_explicit_legacy_path(tmp_path):
    dataset=tmp_path/"legacy"; dataset.mkdir()
    context=resolve_selection(dataset=dataset)
    assert context.dataset_dir==dataset.resolve()
def test_bare_and_mixed_selections_rejected(tmp_path):
    with pytest.raises(ValueError,match="requires"): resolve_selection()
    dataset=tmp_path/"legacy"; dataset.mkdir()
    with pytest.raises(ValueError,match="either"): resolve_selection(dataset=dataset,mouse_id="m")
