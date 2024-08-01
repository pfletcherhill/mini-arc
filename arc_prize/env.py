import modal

modal_image = modal.Image.debian_slim().pip_install("torch")
modal_app = modal.App(image=modal_image)
print(modal_app)

modal_volume = modal.Volume.from_name("arc-model-vol", create_if_missing=True)
