from collections import deque


def multiline_p(s):
    return len(s.split("\n")) > 1


class Node:
    def __init__(self, *args, **kwargs):
        self.lineno = 0
        self.col_offset = 0
        self.end_lineno = 0
        self.end_col_offset = 0
        for [k, v] in kwargs.items():
            self.__dict__[k] = v

    def update_dict(self, key, value):
        self.__dict__[key] = value

    @property
    def position_info(self):
        return {
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "col_offset": self.col_offset,
            "end_col_offset": self.end_col_offset,
        }

    def _generator_expression(self, in_quasi):
        return Paren(
            Symbol(self.classname, **self.position_info),
            self.operands_generate(in_quasi),
            **self.position_info
        )

    def generator_expression(self, in_quasi=False):
        return self._generator_expression(in_quasi)

    def __str__(self):
        return self.get_source()

    def __eq__(self, other):
        return str(self) == other


def data_to_generator_expression(data):
    if isinstance(data, str):
        return String('"' + data + '"')
    elif isinstance(data, list):
        return Bracket(*data)
    elif isinstance(data, tuple):
        return Bracket(*data)
    else:
        return data


class Expression(Node):
    def __init__(self, *tokens, **kwargs):
        super().__init__(**kwargs)
        self.list = list(map(data_to_generator_expression, tokens))

    def append(self, t):
        return self.list.append(t)

    def operands_generate(self, in_quasi):
        return [sexp.generator_expression(in_quasi) for sexp in self.list]

    def generator_expression(self, in_quasi=False):
        return Paren(
            Symbol(self.classname, **self.position_info),
            *self.operands_generate(in_quasi),
            **self.position_info
        )

    def __repr__(self, depth=0):
        return "Expr(" + ", ".join([repr(e) for e in self.list]) + ")"

    def __iter__(self):
        return iter(self.list)

    def __getitem__(self, idx):
        return self.list[idx]

    def __len__(self):
        return len(self.list)

    @property
    def op(self):
        return self.list[0]

    @property
    def operands(self):
        return self.list[slice(1, None)]


class Paren(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classname = "Paren"

    def __repr__(self, depth=0):
        return "Paren" + "(" + ", ".join([repr(e) for e in self.list]) + ")"

    def get_source(self, indent=0):
        return "(" + " ".join([str(e) for e in self.list]) + ")"


def list_format(q, indent=0, col_max=10):
    rst = ""
    cur = indent - 1
    need_newline = False
    cnt = 0
    while q:
        e = q.popleft()
        if need_newline:
            need_newline = False
            rst += "\n" + " " * indent
            cur = indent
        else:
            rst += " "
            cur += 1
        e_src = e.get_source(cur)
        rst += e_src
        if multiline_p(e_src) or cnt == col_max - 1:
            need_newline = True
            cnt = 0
        else:
            cnt = (cnt + 1) % col_max
            cur += len(e_src)
    return rst[1:]


def comprehension_format(q, org_indent=0):
    rst = str(q.popleft())
    indent = org_indent
    cur = indent + len(rst)
    need_newline = False
    while q:
        e = q.popleft()
        e_src = e.get_source(cur)
        if need_newline:
            need_newline = False
            rst += "\n" + " " * indent
            cur = indent
        elif str(e) in ["for", "async-for"]:
            rst += "\n" + " " * org_indent
            cur = org_indent
            indent = org_indent + len(e_src)
        else:
            rst += " "
            cur += 1
        rst += e_src
        if multiline_p(e_src):
            need_newline = True
        else:
            cur += len(e_src)
    return rst


class Bracket(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classname = "Bracket"

    def __repr__(self, depth=0):
        return "Bracket" + "(" + ", ".join([repr(e) for e in self.list]) + ")"

    def get_source(self, indent=0):
        rst = "["
        indent += 1
        if len(self):
            q = deque(self.list)
            if str(self.op) == ":":
                rst += ": "
                indent += 2
                q.popleft()
                rst += list_format(q, indent)
            elif len(q) > 1 and str(q[1]) in ["for", "async-for"]:
                e_src = q.popleft().get_source(indent)
                rst += e_src
                if multiline_p(e_src):
                    rst += "\n" + " " * indent
                else:
                    rst += " "
                    indent += len(e_src) + 1
                rst += comprehension_format(q, indent)
            else:
                rst += list_format(q, indent)
        return rst + "]"


class Brace(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classname = "Brace"

    def __repr__(self, depth=0):
        return "Brace" + "(" + ", ".join([repr(e) for e in self.list]) + ")"

    def get_source(self, indent=0):
        rst = "{"
        indent += 1
        cur = indent
        if len(self):
            q = deque(self.list)
            if len(q) > 2 and str(q[2]) in ["for", "async-for"]:
                any_multiline = False
                for _ in range(2):
                    e_src = q.popleft().get_source(indent)
                    rst += e_src
                    if any_multiline or multiline_p(e_src):
                        any_multiline = True
                        rst += "\n" + " " * indent
                        cur = indent
                    else:
                        rst += " "
                        cur += len(e_src) + 1
                rst += comprehension_format(q, cur)
            elif str(self.op) == ",":
                rst += ", "
                indent += 2
                q.popleft()
                rst += list_format(q, indent)
            else:
                while q:
                    any_multiline = False
                    for _ in range(2):
                        e_src = q.popleft().get_source(indent)
                        rst += e_src
                        if not q:
                            break
                        elif any_multiline or multiline_p(e_src):
                            any_multiline = True
                            rst += "\n" + " " * indent
                            cur = indent
                        else:
                            rst += " "
                            cur += len(e_src) + 1
                    if q and any_multiline:
                        rst += "\n" + " " * indent
        return rst + "}"


class Wrapper(Node):
    def operands_generate(self, in_quasi):
        return self.value.generator_expression(in_quasi)


class FStrExpr(Wrapper):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "FStrExpr"

    def __repr__(self):
        return "FStrExpr(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return "(FStrExpr " + str(self.value) + ")"

    @property
    def name(self):
        return self.value.name

    def update_dict(self, key, value):
        self.__dict__[key] = value
        return self.value.update_dict(key, value)


class Annotation(Wrapper):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "Annotation"

    def __repr__(self):
        return "Ann(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return "^" + str(self.value)

    def append(self, e):
        return self.value.append(e)

    @property
    def name(self):
        return self.value.name

    def update_dict(self, key, value):
        self.__dict__[key] = value
        return self.value.update_dict(key, value)


class Keyword(Wrapper):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "Keyword"

    def __repr__(self):
        return "Kwd(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return ":" + str(self.value)

    @property
    def name(self):
        return self.value.name

    def update_dict(self, key, value):
        self.__dict__[key] = value
        return self.value.update_dict(value)


class Starred(Wrapper):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "Starred"

    def __repr__(self):
        return "Star(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return "*" + str(self.value)

    def append(self, e):
        return self.value.append(e)

    def update_dict(self, key, value):
        self.__dict__[key] = value
        return self.value.update_dict(key, value)


class DoubleStarred(Wrapper):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "DoubleStarred"

    def __repr__(self):
        return "DStar(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return "**" + str(self.value)

    def append(self, e):
        return self.value.append(e)

    def update_dict(self, key, value):
        self.__dict__[key] = value
        return self.value.update_dict(key, value)


class Literal(Node):
    def operands_generate(self, in_quasi):
        return String('"' + self.value + '"', **self.position_info)


class Symbol(Literal):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "Symbol"

    @property
    def name(self):
        return self.value.replace("-", "_")

    def __repr__(self):
        return "Sym(" + self.value + ")"

    def get_source(self, indent=0):
        return self.value


class String(Literal):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "String"

    def operands_generate(self, in_quasi):
        return String(
            '"' + self.value.replace("\\", "\\\\").replace('"', '\\"') + '"',
            **self.position_info
        )

    def __repr__(self):
        return "Str(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return self.value


class Constant(Literal):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "Constant"

    def __repr__(self):
        return "Const(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return str(self.value).replace("'", "")


class MetaIndicator(Node):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def operands_generate(self, in_quasi):
        return self.value.generator_expression(isinstance(self, QuasiQuote))


class Quote(MetaIndicator):
    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.classname = "Quote"

    def __repr__(self):
        return "Quote(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return "'" + str(self.value)


class QuasiQuote(Quote):
    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.classname = "QuasiQuote"

    def __repr__(self):
        return "QuasiQuote(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return "`" + str(self.value)


class Unquote(MetaIndicator):
    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.classname = "Unquote"

    def generator_expression(self, in_quasi=False):
        return self.value if in_quasi else self._generator_expression(False)

    def __repr__(self):
        return "Unquote(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return "~" + str(self.value)


class UnquoteSplice(Unquote):
    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.classname = "UnquoteSplice"

    def generator_expression(self, in_quasi=False):
        return (
            Starred(self.value, **self.position_info)
            if in_quasi
            else self._generator_expression(False)
        )

    def __repr__(self):
        return "UnquoteSplice(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return "~@" + str(self.value)
