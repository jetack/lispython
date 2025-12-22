import ast

from .utils import stmt_to_dump


class TestMatchMethods:
    def test_match_value(self):
        assert stmt_to_dump("""
(match x
  (case "Relevant" ...))""") == ast.dump(ast.parse("""
match x:
    case "Relevant":
        ..."""))

    def test_match_singleton(self):
        assert stmt_to_dump("""
(match x
  (case None
     ...))""") == ast.dump(ast.parse("""
match x:
    case None:
        ..."""))

    def test_match_sequence(self):
        assert stmt_to_dump("""
(match x
  (case [1 2]
     ...))""") == ast.dump(ast.parse("""
match x:
    case [1, 2]:
        ..."""))

    def test_match_star(self):
        assert stmt_to_dump("""
(match x
  (case [1 2 *rest]
     ...)
  (case [*_]
    ...))""") == ast.dump(ast.parse("""
match x:
    case [1, 2, *rest]:
        ...
    case [*_]:
        ..."""))

    def test_match_mapping(self):
        assert stmt_to_dump("""
(match x
  (case {1 _ 2 _}
    ...)
  (case {**rest}
    ...))""") == ast.dump(ast.parse("""
match x:
    case {1: _, 2: _}:
        ...
    case {**rest}:
        ..."""))

    def test_match_class(self):
        assert stmt_to_dump("""
(match x
  (case (Point2D 0 0)
    ...)
  (case (Point3D :x 0 :y 0 :z 0)
    ...))""") == ast.dump(ast.parse("""
match x:
    case Point2D(0, 0):
        ...
    case Point3D(x=0, y=0, z=0):
        ..."""))

    def test_match_as(self):
        assert stmt_to_dump("""
(match x
  (case [x] as y
    ...)
  (case _
    ...))""") == ast.dump(ast.parse("""
match x:
    case [x] as y:
        ...
    case _:
        ..."""))

    def test_match_or(self):
        assert stmt_to_dump("""
(match x
  (case (| [x] y)
    ...))""") == ast.dump(ast.parse("""
match x:
    case [x] | (y):
        ..."""))
