import subprocess

print("Running black...")
out = subprocess.run(["black", "--check", "."], capture_output=True, text=True)
if out.returncode != 0:
    for line in out.stderr.split('\n'):
        if "error: cannot format" in line:
            print("Error found:", line)
