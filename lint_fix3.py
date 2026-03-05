import subprocess

out = subprocess.run(["ruff", "check", ".", "--fix"], capture_output=True, text=True)
if out.returncode != 0:
    print(out.stdout)
