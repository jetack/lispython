import tomllib

def define_env(env):
    try:
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        version = pyproject["project"]["version"]
    except Exception:
        version = "dev"

    env.variables["version"] = version