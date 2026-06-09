import subprocess, sys


def test_run_backtest_help():
    out = subprocess.run([sys.executable, "scripts/run_backtest.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "--config" in out.stdout


def test_build_scores_help():
    out = subprocess.run([sys.executable, "scripts/build_scores.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
