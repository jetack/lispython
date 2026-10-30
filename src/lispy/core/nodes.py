from ast import literal_eval as _literal_eval
from collections import deque
from math import copysign as _copysign
from math import isfinite as _isfinite
from math import isinf as _isinf
from math import isnan as _isnan

from lispy.core.utils import trampoline


def multiline_p(s):
    return "\n" in s


def atom_text(node):
    """``str(node)`` when that text contains no bracketed form, else None.

    The one cheap way to read a form's name (a head such as ``def``, a word
    such as ``for``/``in``, a dict key such as ``+``): it never renders a
    subtree, so a lookup at every nesting level costs O(1) per level instead
    of the whole subtree's text (which made nested lists exponential to
    compile). Wrapper chains are followed iteratively.

    ⚠️ Invariant, relied on by ``Node.__eq__`` and the tests: ``atom_text(n)
    is None`` implies ``str(n)`` contains one of ``(``, ``[``, ``{``. So a
    None matches no name, and code comparing it with a bracket-free word gets
    exactly the answer ``str(n)`` would give.
    """
    parts = []
    pending = [node]
    while pending:
        node = pending.pop()
        if not isinstance(node, Node):
            parts.append(str(node))
        elif isinstance(node, Literal):
            parts.append(node._source(0))
        elif isinstance(node, (Expression, FStrExpr)):
            return None
        elif getattr(type(node), "source_prefix", None) is not None:
            parts.append(node.source_prefix)
            pending.append(node.value)
        elif isinstance(node, GluedHead):
            # Popped head first: the text is head then params.
            pending.append(node.params)
            pending.append(node.head)
        else:
            # A Node subclass this function does not know: its own text.
            parts.append(str(node))
    return "".join(parts)


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

    def generator_expression(self, in_quasi=False):
        """The form that builds this node at run time: a quote's expansion,
        ``(Paren (Symbol "f") ...)`` for ``(f ...)``. In a quasiquote
        (``in_quasi``) an unquote is its value and ``~@x`` is ``*x``.

        A trampoline (core/utils.py), never recursion: each class's
        ``_generator_steps`` returns a leaf's form, or is a generator that asks
        for a child's with ``form = yield child._generator_steps(in_quasi)``,
        so nesting depth costs no Python frames.
        """
        return trampoline(self._generator_steps(in_quasi))

    def _generator_steps(self, in_quasi):
        return self._classname_call((yield self.operands_generate(in_quasi)))

    def _classname_call(self, *operands):
        """``(ClassName operand ...)`` at this node's position."""
        return Paren(
            Symbol(self.classname, **self.position_info),
            *operands,
            **self.position_info,
        )

    def get_source(self, indent=0):
        """The form's source text, its first line starting at column ``indent``.

        Rendering is a trampoline (core/utils.py), never recursion: each
        class's ``_source`` returns a leaf's text, or is a generator that asks
        for a child's text with ``text = yield _source_of(child, child_indent)``
        and returns its own, so nesting depth costs no Python frames (a form
        nested past the recursion limit still renders, e.g. in a compile error
        message or macroexpand output).
        """
        return trampoline(self._source(indent))

    def __str__(self):
        return self.get_source()

    def __eq__(self, other):
        # Exactly ``str(self) == other``, without rendering in the common case
        # of comparing with a name (see atom_text's invariant).
        if isinstance(other, str):
            text = atom_text(self)
            if text is not None:
                return text == other
            if "(" not in other and "[" not in other and "{" not in other:
                return False
        return str(self) == other


def _source_of(child, indent):
    """A child's rendering as a trampoline step: its text if a leaf, else the
    generator that renders it; ``str(child)`` for a non-node."""
    if isinstance(child, Node):
        return child._source(indent)
    return str(child)


def contains_unquote(node):
    """Whether an Unquote (or ~@) occurs anywhere in ``node``: in a node's
    attributes or a list's items, at any depth. One iterative walk, each node
    and list visited once, so neither nesting nor sharing costs more."""
    pending = [node]
    seen = set()
    while pending:
        node = pending.pop()
        if isinstance(node, Unquote):
            return True
        if not isinstance(node, (Node, list)) or id(node) in seen:
            continue
        seen.add(id(node))
        pending.extend(vars(node).values() if isinstance(node, Node) else node)
    return False


_POSITION_KEYS = ("lineno", "col_offset", "end_lineno", "end_col_offset")


def macro_expansion_nodes(expansion, call):
    """A macro's return value as a node tree: a copy of ``expansion`` with
    Python data turned into nodes and every positionless node placed at the
    macro call ``call`` (the Paren that invoked the macro).

    The expansion contract: a macro may put Python data (None, bool, int,
    float, complex, Ellipsis, str, bytes, and lists and tuples of them)
    anywhere in its expansion's forms: an Expression's list, the single
    child form of a wrapper, quote or glued head (``_child_form_keys``), and
    the expansion itself. The expander gets literal arguments as Python
    values (``_unwrap_literal`` in ``macro.py``), and a macro that splices one
    in with ``(f.list.insert 1 x)`` or ``*~y`` bypasses the node constructors,
    so the conversion happens here, one level at a time through
    ``_data_node``, the one home of that conversion (shared with
    ``data_to_generator_expression``). Only those form slots are converted,
    never other attributes (lineno is an int, an f-string's conversion is an
    int). A value no literal spells (nan, a complex with a signed zero, inf
    or nan part) becomes a Constant carrying it exactly (``_exact_constant``).
    An Expression's list may be any list, tuple or deque; its copy is a plain
    list of nodes. Any other value (a dict, a set, an object, also as an
    Expression's list), or a form, list or tuple that contains itself (also
    one a quasiquote unquoted, which the constructor leaves in place), is a
    LispyCompileError at the call naming the macro.

    A macro builds its expansion at run time from quasiquote templates, so
    those nodes (and the converted data) carry the Node default (lineno 0)
    and the Python code compiled from them would report line 0. The expander
    passes the macro call's position, so tracebacks and compile errors point
    at the call. A node that has a position (an unquoted argument the user
    wrote) keeps its own. Descends every Node attribute, so f-string fields
    and their format specs, wrappers, quotes and glued heads are covered
    alike; the copy memos keep nodes shared between places in the expansion
    shared in the copy, and a list or tuple of data met in many places
    becomes one shared Bracket.

    The tree is copied, not filled in place, because the macro may still own
    its nodes (a template held in a global and returned on every call):
    filling them in place would pin the first call's position on every later
    call, and mixing such stale nodes with fresh ones can make a start line
    exceed its end line, which compile() rejects. Nodes are plain attribute
    bags compared by name, never by identity (gensym returns a Symbol with a
    unique name); the id memos keep sharing within the one expansion.

    ⚠️ Convert, copy and fill are one iterative pass over an explicit stack,
    never copy.deepcopy or recursion: deepcopy spends several Python frames
    per nesting level, so a deeply nested macro call hit RecursionError far
    sooner than the same code without the macro. Nodes and plain lists,
    tuples and deques are copied; every other value (str, int, None) is
    shared.
    """
    from lispy.core.errors import LispyCompileError

    position_info = call.position_info
    # Nodes made from data are replaced by their copies like any original;
    # holding them keeps their ids from being reused while the memo is live.
    made = []
    # id(Bracket made from a list or tuple) -> that list or tuple.
    sources = {}
    # id(data list or tuple) -> (data, its Bracket). One Bracket per distinct
    # list or tuple, else data shared as a DAG (D0 = [1]; Dk = [Dk-1, Dk-1])
    # is converted once per path, exponential in its depth. The value holds
    # the data so its id is not reused while the memo is live.
    brackets = {}
    # ids of the containers being copied (nodes, lists, deques), of the form
    # lists and of the data lists and tuples being converted: the path from
    # the root, whichever memo the copy goes to. Meeting
    # one again is a self-reference, not sharing, and wiring its copy in
    # would make the copy a cycle that later passes recurse through forever.
    # One set serves all of these kinds, as the ids of live objects are
    # distinct.
    active = set()

    def self_reference(value):
        return LispyCompileError(
            f"macro {call.op} returned a self-referencing "
            f"{type(value).__name__} in its expansion",
            call,
        )

    def enter(value):
        # Popped once every slot pushed after it (the subtree) is copied.
        active.add(id(value))
        stack.append((None, id(value)))

    def node_of(data):
        if isinstance(data, (list, tuple)):
            # Before the memo, so a memoized Bracket never hides a cycle.
            if id(data) in active:
                raise self_reference(data)
            if id(data) in brackets:
                return brackets[id(data)][1]
        node = _data_node(data)
        if node is data:
            if isinstance(node, Node):
                return node
            raise LispyCompileError(
                f"macro {call.op} returned a {type(data).__name__} in its "
                f"expansion, which has no lispython form",
                call,
            )
        made.append(node)
        if isinstance(data, (list, tuple)):
            sources[id(node)] = data
            brackets[id(data)] = (data, node)
        return node

    def form_list(elements):
        # An Expression's list as a plain list of nodes. Any list (a subclass
        # too, else a cycle through it escapes the checks), tuple or deque is
        # walked; fresh nodes from data have no position and are pushed below
        # to be filled like any template-built node. A Bracket made from a
        # list still holds raw items, converted when the pass reaches it,
        # never by recursion.
        if not isinstance(elements, (list, tuple, deque)):
            raise LispyCompileError(
                f"macro {call.op} returned a {type(elements).__name__} as a "
                f"form's list in its expansion, which has no lispython form",
                call,
            )
        if id(elements) in active:
            raise self_reference(elements)
        if id(elements) in forms:
            return forms[id(elements)]
        enter(elements)
        converted = [node_of(e) for e in elements]
        forms[id(elements)] = converted
        stack.extend((converted, i) for i in range(len(converted)))
        return converted

    # Two memos that never share an entry, because one container can be
    # reached both ways and each way needs a different copy: ``memo`` holds
    # plain copies (nodes, and lists, deques and tuples in any slot but a form
    # list), which keep raw data raw; ``forms`` holds an Expression's form
    # list (id of the original list, tuple or deque) converted through
    # ``node_of`` into a plain list. Each keeps sharing within its kind.
    memo = {}  # id(original) -> its plain copy
    forms = {}  # id(original form list) -> its converted list
    root = [node_of(expansion)]
    # (container, key): the slot still holding an original, to be replaced by
    # its copy. A tuple's slot also gets a finisher, popped once every slot
    # pushed after it (its items' whole subtrees) has been copied, and every
    # copied container (and the data a Bracket is made from) an exit marker
    # (None, its id), popped once its subtree is copied. A tuple needs none:
    # it memoizes only after its items, and any cycle through it also runs
    # through a node or list, which is active while its children are copied.
    # A value on the path is a self-reference whichever memo holds it, so the
    # active check comes before either memo.
    stack = [(root, 0)]
    while stack:
        item = stack.pop()
        if item[0] is None:
            active.discard(item[1])
            continue
        if len(item) == 3:
            container, key, items = item
            copied = tuple(items)
            memo[id(container[key])] = copied
            container[key] = copied
            continue
        container, key = item
        value = container[key]
        if id(value) in active:
            raise self_reference(value)
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
            enter(value)
            for k in _child_form_keys(value):
                attrs[k] = node_of(attrs[k])
            is_form = isinstance(value, Expression)
            # ⚠️ The other slots go below the list's exit marker, so they are
            # copied once the list has left the path: a sibling slot reaching
            # the node's own list (p.orig = p.list) is not a cycle; it gets
            # its own plain copy.
            stack.extend((attrs, k) for k in attrs if not (is_form and k == "list"))
            source = sources.get(id(value))
            if source is not None:
                if id(source) in active:
                    raise self_reference(source)
                enter(source)
            if is_form:
                attrs["list"] = form_list(value.list)
        elif type(value) is list:
            copied = list(value)
            enter(value)
            stack.extend((copied, i) for i in range(len(copied)))
        elif type(value) is deque:
            copied = deque(value, value.maxlen)
            enter(value)
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


_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escaped(ch):
    if ch in _ESCAPES:
        return _ESCAPES[ch]
    if ch.isprintable():
        return ch
    code = ord(ch)
    return (
        f"\\x{code:02x}"
        if code < 0x100
        else f"\\u{code:04x}"
        if code < 0x10000
        else f"\\U{code:08x}"
    )


def string_literal_source(value):
    """A double-quoted lispython literal whose value is exactly ``value``.

    The one home of quoting a str or bytes as source text: macro data,
    quoted symbols and strings all come here. Lispython has no single-quoted
    strings, so ``repr`` will not do. Every character the string scanner or
    evaluation would change is escaped: backslash and the quote, line endings
    (a raw \\r would be normalised to \\n, as CPython does to source), and
    anything not printable (control characters, lone surrogates); printable
    non-ASCII stays literal. Bytes get a ``b`` prefix and every byte outside
    printable ASCII as ``\\xNN``.
    """
    if isinstance(value, bytes):
        body = "".join(_escaped(chr(b)) if b < 0x80 else f"\\x{b:02x}" for b in value)
        return 'b"' + body + '"'
    return '"' + "".join(map(_escaped, value)) + '"'


def _same_float(a, b):
    # == misses the sign of zero and calls nan unequal to itself.
    return _copysign(1.0, a) == _copysign(1.0, b) and (
        a == b or (_isnan(a) and _isnan(b))
    )


def _float_source(value):
    if _isfinite(value):
        return repr(value)
    if _isinf(value):
        # The literal overflows to inf, as in CPython source.
        return "1e999" if value > 0 else "-1e999"
    # nan carries a sign bit that `same value` readers can tell.
    return '(- (float "nan"))' if _copysign(1.0, value) < 0 else '(float "nan")'


def _exact_constant(value):
    """A Constant compiling to ``value`` itself, for a value no literal spells
    (nan, a complex with a signed zero, inf or nan part).

    ⚠️ It compiles from ``exact``, never from a call such as ``(float "nan")``:
    a user binding of ``float``/``complex`` would shadow the call, and in a
    match pattern a call is a class pattern. The text is that call, for
    display only (macroexpand output); it reads back as the same value where
    the builtins are not rebound. ``constant_compile`` and
    ``number_pattern_expr`` read ``exact``.
    """
    if isinstance(value, complex):
        text = f"(complex {_float_source(value.real)} {_float_source(value.imag)})"
    else:
        text = _float_source(value)
    return Constant(text, exact=value)


def _float_node(value):
    if _isnan(value):
        return _exact_constant(value)
    return Constant(_float_source(value))


def _complex_node(value):
    # repr parenthesises a complex with a real part, "(1+2j)"; unparenthesised,
    # "1+2j" is one number token, where the parenthesised text reads as a call.
    text = repr(value).removeprefix("(").removesuffix(")")
    try:
        parsed = _literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, complex) and all(
        _same_float(a, b)
        for a, b in [(parsed.real, value.real), (parsed.imag, value.imag)]
    ):
        return Constant(text)
    # A signed zero or non-finite part has no literal (-0j is -(0j)).
    return _exact_constant(value)


def _data_node(data):
    """One level of Python data as a node, or ``data`` itself when it is a
    node or has no lispython form.

    A list or tuple becomes a Bracket whose list still holds the raw items:
    the caller's walk converts them, so nested data never costs a Python
    frame per level (``data_to_generator_expression``, ``macro_expansion_nodes``).
    """
    if isinstance(data, bool) or data is None or data is ...:
        return Constant("..." if data is ... else str(data))
    if isinstance(data, int):
        # hex has no digit limit; str() refuses ints over 4300 digits.
        return Constant(str(data) if data.bit_length() < 4000 else hex(data))
    if isinstance(data, float):
        return _float_node(data)
    if isinstance(data, complex):
        return _complex_node(data)
    if isinstance(data, (str, bytes)):
        return String(string_literal_source(data))
    if isinstance(data, (list, tuple)):
        bracket = Bracket()
        bracket.list = list(data)
        return bracket
    return data


def _child_form_keys(node):
    """Attributes of ``node`` that hold one child form, which a macro may set
    to Python data like any element of an Expression's list."""
    if isinstance(node, (Wrapper, MetaIndicator)):
        return ("value",)
    if isinstance(node, GluedHead):
        return ("head", "params")
    return ()


def data_to_generator_expression(data):
    """``data`` as a node tree compiling to the same value, nested lists and
    tuples included; a node is returned as is, and a value with no lispython
    form (a dict, a set, a list that contains itself) is returned unconverted
    and left in place, so ``macro_expansion_nodes`` reports it at the macro
    call (a quasiquote's Expression constructor comes here first).

    The one home of the data-to-node conversion, through ``_data_node``.
    Iterative, so a deeply nested list does not hit RecursionError. A list or
    tuple met in many places becomes one shared Bracket.
    """
    root = _data_node(data)
    brackets = {}
    pending = []
    if isinstance(data, (list, tuple)):
        # The memo value holds the data so its id is not reused meanwhile.
        brackets[id(data)] = (data, root)
        pending.append((root, data))
    # (Bracket, the list or tuple it is made from); (None, id) leaves that
    # list, so ``active`` is the path from the root and meeting a list on it
    # again is a self-reference (sharing a list in two places is fine).
    # ``done`` holds the lists already filled. A shared list is queued at
    # every place it appears but filled once, when first popped; queueing it
    # even on a memo hit keeps the cycle check sound: a cycle through a list
    # still waiting in the queue is found when that list is popped and its
    # items are checked against the path then, never skipped by the memo.
    active = set()
    done = set()
    while pending:
        [bracket, items] = pending.pop()
        if bracket is None:
            active.discard(items)
            continue
        if id(items) in active:
            return data
        if id(items) in done:
            continue
        done.add(id(items))
        active.add(id(items))
        pending.append((None, id(items)))
        nodes = []
        for item in items:
            if isinstance(item, (list, tuple)):
                if id(item) not in brackets:
                    brackets[id(item)] = (item, _data_node(item))
                node = brackets[id(item)][1]
                if id(item) in active:
                    return data
                if id(item) not in done:
                    pending.append((node, item))
            else:
                node = _data_node(item)
            nodes.append(node)
        bracket.list = nodes
    return root


class Expression(Node):
    def __init__(self, *tokens, **kwargs):
        super().__init__(**kwargs)
        self.list = list(map(data_to_generator_expression, tokens))

    def append(self, t):
        return self.list.append(t)

    def _generator_steps(self, in_quasi):
        operands = []
        for sexp in self.list:
            operands.append((yield sexp._generator_steps(in_quasi)))
        return self._classname_call(*operands)

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

    def _source(self, indent):
        # Elements render at indent 0, as str(e) always did.
        texts = []
        for e in self.list:
            texts.append((yield _source_of(e, 0)))
        return "(" + " ".join(texts) + ")"


# list_format and comprehension_format are _source helpers: generators run
# with ``yield from`` inside a node's _source (see Node.get_source).
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
        e_src = yield _source_of(e, cur)
        rst += e_src
        if multiline_p(e_src) or cnt == col_max - 1:
            need_newline = True
            cnt = 0
        else:
            cnt = (cnt + 1) % col_max
            cur += len(e_src)
    return rst[1:]


def comprehension_format(q, org_indent=0):
    rst = yield _source_of(q.popleft(), 0)
    indent = org_indent
    cur = indent + len(rst)
    need_newline = False
    while q:
        e = q.popleft()
        e_src = yield _source_of(e, cur)
        if need_newline:
            need_newline = False
            rst += "\n" + " " * indent
            cur = indent
        elif atom_text(e) in ("for", "async-for"):
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

    def _source(self, indent):
        rst = "["
        indent += 1
        if len(self):
            q = deque(self.list)
            if atom_text(self.op) == ":":
                rst += ": "
                indent += 2
                q.popleft()
                rst += yield from list_format(q, indent)
            elif len(q) > 1 and atom_text(q[1]) in ("for", "async-for"):
                e_src = yield _source_of(q.popleft(), indent)
                rst += e_src
                if multiline_p(e_src):
                    rst += "\n" + " " * indent
                else:
                    rst += " "
                    indent += len(e_src) + 1
                rst += yield from comprehension_format(q, indent)
            else:
                rst += yield from list_format(q, indent)
        return rst + "]"


class Brace(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classname = "Brace"

    def __repr__(self, depth=0):
        return "Brace" + "(" + ", ".join([repr(e) for e in self.list]) + ")"

    def _source(self, indent):
        rst = "{"
        indent += 1
        cur = indent
        if len(self):
            q = deque(self.list)
            if len(q) > 2 and atom_text(q[2]) in ("for", "async-for"):
                any_multiline = False
                for _ in range(2):
                    e_src = yield _source_of(q.popleft(), indent)
                    rst += e_src
                    if any_multiline or multiline_p(e_src):
                        any_multiline = True
                        rst += "\n" + " " * indent
                        cur = indent
                    else:
                        rst += " "
                        cur += len(e_src) + 1
                rst += yield from comprehension_format(q, cur)
            elif atom_text(self.op) == ",":
                rst += ", "
                indent += 2
                q.popleft()
                rst += yield from list_format(q, indent)
            else:
                while q:
                    any_multiline = False
                    for _ in range(2):
                        e_src = yield _source_of(q.popleft(), indent)
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


def _prefixed_source(self, indent):
    """``_source`` of a wrapper or quote: its ``source_prefix`` (the one home of
    that prefix, shared with atom_text), then its value rendered at indent 0."""
    return self.source_prefix + (yield _source_of(self.value, 0))


class Wrapper(Node):
    _source = _prefixed_source

    def operands_generate(self, in_quasi):
        return self.value._generator_steps(in_quasi)


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

    def _source(self, indent):
        return "(FStrExpr " + (yield _source_of(self.value, 0)) + ")"

    def _generator_steps(self, in_quasi):
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
        value = yield self.value._generator_steps(in_quasi)
        return Paren(
            Symbol(self.classname, **position_info),
            value,
            Keyword(Symbol("conversion")),
            Constant(str(getattr(self, "conversion", -1))),
            Keyword(Symbol("format-spec")),
            Constant("None")
            if format_spec is None
            else (yield format_spec._generator_steps(in_quasi)),
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
    source_prefix = "^"

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "Annotation"

    def __repr__(self):
        return "Ann(" + repr(self.value) + ")"

    def append(self, e):
        return self.value.append(e)

    @property
    def name(self):
        return self.value.name

    def update_dict(self, key, value):
        self.__dict__[key] = value
        return self.value.update_dict(key, value)


class Keyword(Wrapper):
    source_prefix = ":"

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "Keyword"

    def __repr__(self):
        return "Kwd(" + repr(self.value) + ")"

    @property
    def name(self):
        return self.value.name

    def update_dict(self, key, value):
        self.__dict__[key] = value
        return self.value.update_dict(key, value)


class Starred(Wrapper):
    source_prefix = "*"

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "Starred"

    def __repr__(self):
        return "Star(" + repr(self.value) + ")"

    def append(self, e):
        return self.value.append(e)

    def update_dict(self, key, value):
        self.__dict__[key] = value
        return self.value.update_dict(key, value)


class DoubleStarred(Wrapper):
    source_prefix = "**"

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "DoubleStarred"

    def __repr__(self):
        return "DStar(" + repr(self.value) + ")"

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

    def _generator_steps(self, in_quasi):
        head = yield self.head._generator_steps(in_quasi)
        return self._classname_call(
            head, (yield self.params._generator_steps(in_quasi))
        )

    def __repr__(self):
        return "Glued(" + repr(self.head) + ", " + repr(self.params) + ")"

    def _source(self, indent):
        # The params start where the head's text ends.
        head = yield _source_of(self.head, indent)
        return head + (yield _source_of(self.params, indent + len(head)))


class Literal(Node):
    def operands_generate(self, in_quasi):
        return String(string_literal_source(self.value), **self.position_info)

    def _generator_steps(self, in_quasi):
        # A leaf: its form, no generator.
        return self._classname_call(self.operands_generate(in_quasi))


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

    def _source(self, indent):
        return self.value


class String(Literal):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "String"

    def __repr__(self):
        return "Str(" + repr(self.value) + ")"

    def _source(self, indent):
        return self.value


class Constant(Literal):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.classname = "Constant"

    def _generator_steps(self, in_quasi):
        """Rebuild from the text, and keep an exact value (``_exact_constant``)
        exact: the Constant itself compiles to that value, so the rebuilt
        code names no ``float``/``complex`` a user may have rebound."""
        if "exact" not in self.__dict__:
            return super()._generator_steps(in_quasi)
        return self._classname_call(
            self.operands_generate(in_quasi), Keyword(Symbol("exact")), self
        )

    def __repr__(self):
        return "Const(" + repr(self.value) + ")"

    def _source(self, indent):
        return str(self.value).replace("'", "")


class MetaIndicator(Node):
    _source = _prefixed_source

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def operands_generate(self, in_quasi):
        return self.value._generator_steps(isinstance(self, QuasiQuote))


class Quote(MetaIndicator):
    source_prefix = "'"

    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.classname = "Quote"

    def __repr__(self):
        return "Quote(" + repr(self.value) + ")"


class QuasiQuote(Quote):
    source_prefix = "`"

    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.classname = "QuasiQuote"

    def __repr__(self):
        return "QuasiQuote(" + repr(self.value) + ")"


class Unquote(MetaIndicator):
    source_prefix = "~"

    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.classname = "Unquote"

    def _generator_steps(self, in_quasi):
        return self.value if in_quasi else Node._generator_steps(self, False)

    def __repr__(self):
        return "Unquote(" + repr(self.value) + ")"


class UnquoteSplice(Unquote):
    source_prefix = "~@"

    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.classname = "UnquoteSplice"

    def _generator_steps(self, in_quasi):
        return (
            Starred(self.value, **self.position_info)
            if in_quasi
            else Node._generator_steps(self, False)
        )

    def __repr__(self):
        return "UnquoteSplice(" + repr(self.value) + ")"
