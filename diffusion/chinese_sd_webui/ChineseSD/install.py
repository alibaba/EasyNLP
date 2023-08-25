import os
import launch

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements_chsd.txt")

with open(req_file) as f:
    for lib in f:
        lib = lib.strip()
        if not launch.is_installed(lib):
            print("pip install {}".format(lib))
            launch.run_pip(f"install {lib}", f"sd-webui requirement: {lib}")
