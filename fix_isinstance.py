import subprocess
out = subprocess.run(["grep", "-rn", "isinstance", "tests/"], capture_output=True, text=True)
print(out.stdout)
