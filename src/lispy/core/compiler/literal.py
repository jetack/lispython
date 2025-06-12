import ast


def constant_compile(constant):
    return ast.Constant(value=eval(constant.value), **constant.position_info)


def string_compile(string):
    return ast.Constant(
        value=eval(string.value.replace("\r\n", "\\n").replace("\n", "\\n")),
        **string.position_info
    )


def name_compile(symbol, ctx):
    [name, *attrs] = symbol.name.replace("-", "_").split(".")
    position_info = {**symbol.position_info}
    position_info["end_col_offset"] = position_info["col_offset"] + len(name)
    rst = ast.Name(id=name, ctx=ast.Load(), **position_info)
    for attr in attrs:
        position_info["end_col_offset"] += 1 + len(attr)
        rst = ast.Attribute(value=rst, attr=attr, ctx=ast.Load(), **position_info)
    rst.ctx = ctx()
    return rst
