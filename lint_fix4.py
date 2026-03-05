import subprocess
out = subprocess.run(["grep", "-rn", "sys.modules\\[", "tests/"], capture_output=True, text=True)
print(out.stdout)
out = subprocess.run(["grep", "-rn", "sys.modules.setdefault", "tests/"], capture_output=True, text=True)
print(out.stdout)
