import os, base64

img_path = "/content/ComfyUI/output/explainer_00001_.png"
if os.path.exists(img_path):
    print(f"IMAGE_FOUND: size={os.path.getsize(img_path)}")
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    print(f"IMAGE_B64_START:{data[:50]}")
    with open("/content/img_b64.txt", "w") as f:
        f.write(data)
    print("SAVED_B64")
else:
    print("IMAGE_NOT_FOUND")
