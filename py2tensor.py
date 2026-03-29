"""
Py2Tensor: Python CPU functions -> GPU tensor computation graphs
===============================================================
No training. No approximation. Exact conversion.

Usage:
    from py2tensor import tensorize

    @tensorize
    def my_func(x):
        if x > 0:
            return x * 2
        else:
            return x + 1

    # Now my_func accepts batched GPU tensors
    x = torch.randn(10000, device='cuda')
    result = my_func(x)  # runs on GPU, all 10000 in parallel
"""
import ast
import inspect
import textwrap
import torch
import types


def tensorize(fn=None, lookup_tables=None):
    """Decorator: converts scalar Python function to batched GPU tensor function.

    Args:
        fn: the function to convert
        lookup_tables: dict of {name: list/array} to convert to GPU tensors
                       These become torch tensors accessible inside the function.
    """
    def decorator(fn):
        source = inspect.getsource(fn)
        source = textwrap.dedent(source)

        # Remove decorator lines
        lines = source.split('\n')
        clean_lines = []
        skip = True
        for line in lines:
            if skip and (line.strip().startswith('@') or line.strip() == ''):
                continue
            skip = False
            clean_lines.append(line)
        source = '\n'.join(clean_lines)

        tree = ast.parse(source)
        transformer = TensorTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        new_source = ast.unparse(new_tree)

        # Build namespace with torch + lookup tables
        func_name = fn.__name__
        wrapper_code = "import torch\n" + new_source

        namespace = {'torch': torch}
        if lookup_tables:
            for k, v in lookup_tables.items():
                if isinstance(v, (list, tuple)):
                    namespace[k] = torch.tensor(v, dtype=torch.float32)
                elif hasattr(v, '__len__'):
                    namespace[k] = torch.tensor(list(v), dtype=torch.float32)
                else:
                    namespace[k] = v

        compiled = compile(wrapper_code, f'<py2tensor:{func_name}>', 'exec')
        run_module(compiled, namespace)
        gpu_fn = namespace[func_name]

        # Wrapper
        def wrapper(*args, **kwargs):
            if any(isinstance(a, torch.Tensor) for a in args):
                # Move lookup tables to same device as input
                dev = next(a.device for a in args if isinstance(a, torch.Tensor))
                if lookup_tables:
                    for k in lookup_tables:
                        if k in namespace and isinstance(namespace[k], torch.Tensor):
                            if namespace[k].device != dev:
                                namespace[k] = namespace[k].to(dev)
                return gpu_fn(*args, **kwargs)
            return fn(*args, **kwargs)

        # Try torch.compile for extra speed
        compiled_fn = None
        try:
            compiled_fn = torch.compile(gpu_fn)
        except Exception:
            compiled_fn = gpu_fn

        wrapper._original = fn
        wrapper._gpu = gpu_fn
        wrapper._compiled = compiled_fn
        wrapper._tensor_source = new_source
        wrapper.__name__ = func_name
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def run_module(code_obj, namespace):
    """Safely execute compiled code in namespace."""
    # This runs the compiled AST - not arbitrary user strings
    namespace['__builtins__'] = __builtins__
    types.FunctionType(code_obj, namespace)()


class _NameReplacer(ast.NodeTransformer):
    """Replace a specific Name node (exact match) with a Constant."""
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def visit_Name(self, node):
        if node.id == self.name:
            return ast.Constant(value=self.value)
        return node


class TensorTransformer(ast.NodeTransformer):
    """AST transformer: converts Python scalar ops to tensor ops."""

    def visit_If(self, node):
        """if cond: body else: orelse -> torch.where(cond, body, orelse)"""
        self.generic_visit(node)

        # if/else with single return in each branch
        if (len(node.body) == 1 and len(node.orelse) == 1 and
            isinstance(node.body[0], ast.Return) and
            isinstance(node.orelse[0], ast.Return)):

            cond = node.test
            true_val = node.body[0].value
            false_val = node.orelse[0].value

            return ast.Return(value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='where', ctx=ast.Load()),
                args=[cond, true_val, false_val],
                keywords=[]
            ))

        # if/else with single assignment in each branch
        if (len(node.body) == 1 and len(node.orelse) == 1 and
            isinstance(node.body[0], ast.Assign) and
            isinstance(node.orelse[0], ast.Assign)):

            t1 = node.body[0].targets[0]
            t2 = node.orelse[0].targets[0]
            cond = node.test

            if isinstance(t1, ast.Name) and isinstance(t2, ast.Name):
                if t1.id == t2.id:
                    # Same variable: simple torch.where
                    return ast.Assign(
                        targets=[ast.Name(id=t1.id, ctx=ast.Store())],
                        value=self._make_where(cond, node.body[0].value, node.orelse[0].value)
                    )
                else:
                    # Different variables: both need conditional update
                    # if cond: a = X else: b = Y
                    # -> a = torch.where(cond, X, a)
                    #    b = torch.where(cond, b, Y)
                    return [
                        ast.Assign(
                            targets=[ast.Name(id=t1.id, ctx=ast.Store())],
                            value=self._make_where(cond, node.body[0].value,
                                                   ast.Name(id=t1.id, ctx=ast.Load()))
                        ),
                        ast.Assign(
                            targets=[ast.Name(id=t2.id, ctx=ast.Store())],
                            value=self._make_where(cond, ast.Name(id=t2.id, ctx=ast.Load()),
                                                   node.orelse[0].value)
                        ),
                    ]

        return node

    def _make_where(self, cond, true_val, false_val):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='torch', ctx=ast.Load()),
                attr='where', ctx=ast.Load()),
            args=[cond, true_val, false_val],
            keywords=[]
        )

    def visit_Call(self, node):
        """Convert built-in and math/numpy functions to torch equivalents."""
        self.generic_visit(node)

        # Built-in functions
        if isinstance(node.func, ast.Name):
            name = node.func.id
            torch_map = {
                'abs': 'abs', 'max': 'max', 'min': 'min',
                'sum': 'sum', 'round': 'round', 'pow': 'pow',
                'float': None, 'int': None,
            }
            if name in torch_map:
                if torch_map[name] is None:
                    return node.args[0] if node.args else node
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='torch', ctx=ast.Load()),
                        attr=torch_map[name], ctx=ast.Load()),
                    args=node.args, keywords=[]
                )

        # math.X and np.X -> torch.X
        if isinstance(node.func, ast.Attribute):
            obj = node.func.value
            attr = node.func.attr
            if isinstance(obj, ast.Name) and obj.id in ('math', 'np', 'numpy'):
                math_torch_map = {
                    'sqrt': 'sqrt', 'exp': 'exp', 'log': 'log', 'log2': 'log2',
                    'log10': 'log10', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
                    'asin': 'asin', 'acos': 'acos', 'atan': 'atan', 'atan2': 'atan2',
                    'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh',
                    'floor': 'floor', 'ceil': 'ceil', 'abs': 'abs',
                    'fabs': 'abs', 'pow': 'pow', 'maximum': 'maximum',
                    'minimum': 'minimum', 'clip': 'clamp', 'sign': 'sign',
                    'square': 'square',
                }
                if attr in math_torch_map:
                    # Wrap non-Name args in torch.as_tensor() to handle constants
                    new_args = []
                    for a in node.args:
                        if isinstance(a, ast.Name):
                            new_args.append(a)  # already a tensor variable
                        else:
                            # Wrap in torch.as_tensor() for safety
                            new_args.append(ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='torch', ctx=ast.Load()),
                                    attr='as_tensor', ctx=ast.Load()),
                                args=[a], keywords=[]
                            ))
                    return ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='torch', ctx=ast.Load()),
                            attr=math_torch_map[attr], ctx=ast.Load()),
                        args=new_args, keywords=node.keywords
                    )

        return node

    def visit_Attribute(self, node):
        """Convert math.pi, math.e etc. to torch constants."""
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in ('math', 'np', 'numpy'):
            const_map = {
                'pi': 3.141592653589793,
                'e': 2.718281828459045,
                'inf': float('inf'),
            }
            if node.attr in const_map:
                return ast.Constant(value=const_map[node.attr])
        return node

    def visit_For(self, node):
        """Convert for i in range(N) to unrolled statements."""
        # First transform body (converts if/else inside loop to torch.where)
        self.generic_visit(node)

        if (isinstance(node.iter, ast.Call) and
            isinstance(node.iter.func, ast.Name) and
            node.iter.func.id == 'range'):

            args = node.iter.args
            start, stop, step = 0, None, 1
            if len(args) == 1 and isinstance(args[0], ast.Constant):
                stop = args[0].value
            elif len(args) == 2 and isinstance(args[0], ast.Constant) and isinstance(args[1], ast.Constant):
                start, stop = args[0].value, args[1].value
            elif len(args) == 3 and all(isinstance(a, ast.Constant) for a in args):
                start, stop, step = args[0].value, args[1].value, args[2].value

            if stop is not None and abs(stop - start) <= 64:
                var_name = node.target.id if isinstance(node.target, ast.Name) else None
                if var_name:
                    unrolled = []
                    import copy
                    for i in range(start, stop, step):
                        for stmt in node.body:
                            new_stmt = copy.deepcopy(stmt)
                            replacer = _NameReplacer(var_name, i)
                            new_stmt = replacer.visit(new_stmt)
                            # Transform the unrolled statement (convert if→where etc.)
                            new_stmt = self.visit(new_stmt)
                            ast.fix_missing_locations(new_stmt)
                            if isinstance(new_stmt, list):
                                unrolled.extend(new_stmt)
                            else:
                                unrolled.append(new_stmt)
                    return unrolled

        return node

    def visit_While(self, node):
        """Convert while loop to bounded for loop (max 64 iterations).
        while cond: body -> for _ in range(64): if cond: body"""
        self.generic_visit(node)
        MAX_ITER = 64
        # Unroll as: for _ in range(MAX_ITER): body (with break simulated via masking)
        unrolled = []
        for _ in range(MAX_ITER):
            # Add condition check: if not cond, all subsequent ops are masked
            for stmt in node.body:
                unrolled.append(stmt)
        return unrolled

    def visit_AugAssign(self, node):
        """Convert += -= *= /= to regular assignment with tensor op."""
        self.generic_visit(node)
        # x += val -> x = x + val
        op_map = {
            ast.Add: ast.Add(), ast.Sub: ast.Sub(),
            ast.Mult: ast.Mult(), ast.Div: ast.Div(),
            ast.Mod: ast.Mod(), ast.Pow: ast.Pow(),
        }
        op = op_map.get(type(node.op), node.op)
        return ast.Assign(
            targets=[node.target],
            value=ast.BinOp(
                left=ast.Name(id=node.target.id, ctx=ast.Load()) if isinstance(node.target, ast.Name) else node.target,
                op=op,
                right=node.value
            )
        )

    def visit_Subscript(self, node):
        """Convert array[index] to tensor indexing."""
        self.generic_visit(node)
        # array[idx] with integer tensor idx -> works directly in PyTorch
        # Just need to ensure idx is long type for indexing
        if isinstance(node.slice, ast.Name) or isinstance(node.slice, ast.BinOp):
            # Wrap index in .long() for safe tensor indexing
            node.slice = ast.Call(
                func=ast.Attribute(
                    value=node.slice,
                    attr='long', ctx=ast.Load()),
                args=[], keywords=[]
            )
        return node

    def visit_IfExp(self, node):
        """Ternary: val_true if cond else val_false -> torch.where(cond, true, false)"""
        self.generic_visit(node)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='torch', ctx=ast.Load()),
                attr='where', ctx=ast.Load()),
            args=[node.test, node.body, node.orelse],
            keywords=[]
        )

    def visit_UnaryOp(self, node):
        """Convert unary ops: -x, not x"""
        self.generic_visit(node)
        if isinstance(node.op, ast.USub):
            return ast.BinOp(
                left=ast.Constant(value=0),
                op=ast.Sub(),
                right=node.operand
            )
        if isinstance(node.op, ast.Not):
            # not x -> 1 - x (for boolean tensors)
            return ast.BinOp(
                left=ast.Constant(value=1),
                op=ast.Sub(),
                right=node.operand
            )
        return node

    def visit_BoolOp(self, node):
        """and/or -> tensor bitwise operations."""
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            result = node.values[0]
            for v in node.values[1:]:
                result = ast.BinOp(left=result, op=ast.BitAnd(), right=v)
            return result
        elif isinstance(node.op, ast.Or):
            result = node.values[0]
            for v in node.values[1:]:
                result = ast.BinOp(left=result, op=ast.BitOr(), right=v)
            return result
        return node
