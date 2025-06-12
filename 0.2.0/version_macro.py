import toml

def define_env(env):
    try:
        pyproject = toml.load("pyproject.toml")
        version = pyproject["tool"]["poetry"]["version"]
    except Exception:
        version = "dev"

    env.variables["version"] = version