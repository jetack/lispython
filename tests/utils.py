import ast

from lispy.core.macro import compile_unit
from lispy.tools import parse


def stmt_to_dump(src):
    return ast.dump(ast.Module(compile_unit(parse(src), {}), type_ignores=[]))
