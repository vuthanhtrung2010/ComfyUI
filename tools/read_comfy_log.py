import os

if os.path.exists("/content/comfyui.log"):
    with open("/content/comfyui.log", "r") as f:
        lines = f.readlines()
    print("=== TAIL OF COMFYUI.LOG ===")
    for l in lines[-25:]:
        print(l.rstrip())
else:
    print("NO LOG FILE")
