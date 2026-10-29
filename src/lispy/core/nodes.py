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


def contains_unquote(node):
    """Whether an Unquote (or ~@) occurs anywhere in ``node``."""
    if isinstance(node, Unquote):
        return True
    if isinstance(node, Node):
        return any(
            contains_unquote(v) for v in vars(node).values() if v is not node
        )
    if isinstance(node, list):
        return any(contains_unquote(v) for v in node)
    return False


_POSITION_KEYS = ("lineno", "col_offset", "end_lineno", "end_col_offset")


def with_missing_positions(node, position_info):
    """A copy of ``node`` in which every node with no position (lineno 0)
    takes this one.

    A macro builds its expansion at run time from quasiquote templates, so
    those nodes carry the Node default (lineno 0) and the Python code compiled
    from them would report line 0. The expander passes the macro call's
    position, so tracebacks and compile errors point at the call. A node that
    has a position (an unquoted argument the user wrote) keeps its own.
    Descends every Node attribute, so f-string fields and their format specs,
    wrappers, quotes and glued heads are covered alike; ``seen`` guards nodes
    shared between places in the expansion.

    The tree is copied, not filled in place, because the macro may still own
    its nodes (a template held in a global and returned on every call):
    filling them in place would pin the first call's position on every later
    call, and mixing such stale nodes with fresh ones can make a start line
    exceed its end line, which compile() rejects. Nodes are plain attribute
    bags compared by name, never by identity (gensym returns a Symbol with a
    unique name); the id memo keeps sharing within the one expansion.

    ⚠️ Copy and fill are one iterative pass over an explicit stack, never
    copy.deepcopy or recursion: deepcopy spends several Python frames per
    nesting level, so a deeply nested macro call hit RecursionError far
    sooner than the same code without the macro. Nodes and plain lists,
    tuples and deques are copied; every other value (str, int, None) is
    shared.
    """
    memo = {}  # id(original) -> its copy
    root = [node]
    # (container, key): the slot still holding an original, to be replaced by
    # its copy. A tuple's slot also gets a finisher, popped once every slot
    # pushed after it (its items' whole subtrees) has been copied.
    stack = [(root, 0)]
    while stack:
        item = stack.pop()
        if len(item) == 3:
            container, key, items = item
            copied = tuple(items)
            memo[id(container[key])] = copied
            container[key] = copied
            continue
        container, key = item
        value = container[key]
        if id(value) in memo:
            container[key] = memo[id(value)]
            continue
        if isinstance(value, Node):
            copied = value.__class__.__new__(value.__class__)
            attrs = copied.__dict__
            attrs.update(value.__dict__)
            if not value.lineno:
                # Set the node's own attributes: Wrapper.update_dict delegates
                # to the wrapped value, which may already have a position.
                for position_key in _POSITION_KEYS:
                    attrs[position_key] = position_info[position_key]
            stack.extend((attrs, k) for k in attrs)
        elif type(value) is list:
            copied = list(value)
            stack.extend((copied, i) for i in range(len(copied)))
        elif type(value) is deque:
            copied = deque(value, value.maxlen)
            stack.extend((copied, i) for i in range(len(copied)))
        elif type(value) is tuple:
            items = list(value)
            stack.append((container, key, items))
            stack.extend((items, i) for i in range(len(items)))
            continue
        else:
            continue
        memo[id(value)] = copied
        container[key] = copied
    return root[0]


def data_to_generator_expression(data):
    if isinstance(data, bool) or data is None:
        return Constant(str(data))
    elif isinstance(data, (int, float, complex)):
        return Constant(str(data))
    elif isinstance(data, str):
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
    """A replacement field of an f/t-string.

    ``expr_text`` is the expression's source text (Interpolation.str, and
    the ``{expr=}`` echo, which is ``expr_text + debug``).  Built from a
    quasiquote with an unquote inside, the text is that of the expanded
    expression, padded as written: what the user would have typed.
    """

    def __init__(self, value, expr_text=None, expr_padding=("", ""), **kwargs):
        super().__init__(**kwargs)
        self.value = data_to_generator_expression(value)
        self.expr_text = (
            expr_padding[0] + str(self.value) + expr_padding[1]
            if expr_text is None
            else expr_text
        )
        self.classname = "FStrExpr"

    def __repr__(self):
        return "FStrExpr(" + repr(self.value) + ")"

    def get_source(self, indent=0):
        return "(FStrExpr " + str(self.value) + ")"

    def generator_expression(self, in_quasi=False):
        """Rebuild with conversion, format spec and expression text, so an
        f/t-string survives a quasiquote."""
        position_info = self.position_info
        format_spec = getattr(self, "format_spec", None)
        debug = getattr(self, "debug", None)
        text = self.expr_text
        if in_quasi and contains_unquote(self.value):
            # The text is only known once the unquote is expanded.
            lead = text[: len(text) - len(text.lstrip())]
            trail = text[len(text.rstrip()) :]
            text_args = [
                Keyword(Symbol("expr-padding")),
                Paren(Symbol(","), String(repr(lead)), String(repr(trail))),
            ]
        else:
            text_args = [Keyword(Symbol("expr-text")), String(repr(text))]
        return Paren(
            Symbol(self.classname, **position_info),
            self.value.generator_expression(in_quasi),
            Keyword(Symbol("conversion")),
            Constant(str(getattr(self, "conversion", -1))),
            Keyword(Symbol("format-spec")),
            Constant("None")
            if format_spec is None
            else format_spec.generator_expression(in_quasi),
            Keyword(Symbol("debug")),
            Constant("None") if debug is None else String(repr(debug)),
            *text_args,
            **position_info
        )

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
        return self.value.update_dict(key, value)


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


class GluedHead(Node):
    """``name[params]`` written with no space: a generic def/class/type name.

    The parser builds it only when ``[`` starts exactly where a bare symbol
    ends (``f [T]`` stays two forms).  It is meaningful only as the name of
    def, async-def, class and type; reading ``.name`` anywhere else is the
    misuse, so it raises the compile error instead of returning a string.
    """

    def __init__(self, head, params, **kwargs):
        super().__init__(**kwargs)
        self.head = head
        self.params = params
        self.classname = "GluedHead"

    def misuse_error(self):
        from lispy.core.errors import LispyCompileError

        text = str(self.head)
        return LispyCompileError(
            f"{text}[...] is only valid as a def/class/type name; "
            f"use (sub {text} ...)",
            self,
        )

    @property
    def name(self):
        raise self.misuse_error()

    def generator_expression(self, in_quasi=False):
        return Paren(
            Symbol(self.classname, **self.position_info),
            self.head.generator_expression(in_quasi),
            self.params.generator_expression(in_quasi),
            **self.position_info
        )

    def __repr__(self):
        return "Glued(" + repr(self.head) + ", " + repr(self.params) + ")"

    def get_source(self, indent=0):
        return self.head.get_source(indent) + self.params.get_source(
            indent + len(self.head.get_source())
        )


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
        return self.value.replace("->", "_to_").replace("-", "_").replace("?", "_p")

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
