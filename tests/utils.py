import ast

from lispy.tools import macroexpand_then_compile, parse


def stmt_to_dump(src):
    return ast.dump(ast.Module(macroexpand_then_compile(parse(src)), type_ignores=[]))

