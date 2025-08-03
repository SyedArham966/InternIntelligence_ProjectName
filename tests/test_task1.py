import pathlib
import re
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import task1_hyperparameter_tuning as t


def test_task1_f1_exceeds_threshold(capsys):
    t.main()
    captured = capsys.readouterr().out
    f1_scores = [float(match) for match in re.findall(r"f1: ([0-9.]+)", captured)]
    assert f1_scores, "No F1 scores found in output"
    assert max(f1_scores) > 0.95
