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


def tensorize(fn):
    """Decorator: converts scalar Python function to batched GPU tensor function."""
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

    # Build the GPU function
    func_name = fn.__name__
    wrapper_code = "import torch\n" + new_source

    namespace = {'torch': torch}
    compiled = compile(wrapper_code, f'<py2tensor:{func_name}>', 'exec')
    run_module(compiled, namespace)
    gpu_fn = namespace[func_name]

    # Return wrapper
    def wrapper(*args, **kwargs):
        if any(isinstance(a, torch.Tensor) for a in args):
            return gpu_fn(*args, **kwargs)
        return fn(*args, **kwargs)

    wrapper._original = fn
    wrapper._gpu = gpu_fn
    wrapper._tensor_source = new_source
    wrapper.__name__ = func_name
    return wrapper


def run_module(code_obj, namespace):
    """Safely execute compiled code in namespace."""
    # This runs the compiled AST - not arbitrary user strings
    namespace['__builtins__'] = __builtins__
    types.FunctionType(code_obj, namespace)()


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

        # if/else with single assignment to same variable
        if (len(node.body) == 1 and len(node.orelse) == 1 and
            isinstance(node.body[0], ast.Assign) and
            isinstance(node.orelse[0], ast.Assign)):

            t1 = node.body[0].targets[0]
            t2 = node.orelse[0].targets[0]
            if isinstance(t1, ast.Name) and isinstance(t2, ast.Name) and t1.id == t2.id:
                cond = node.test
                true_val = node.body[0].value
                false_val = node.orelse[0].value

                return ast.Assign(
                    targets=[ast.Name(id=t1.id, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='torch', ctx=ast.Load()),
                            attr='where', ctx=ast.Load()),
                        args=[cond, true_val, false_val],
                        keywords=[]
                    )
                )

        return node

    def visit_Call(self, node):
        """Convert built-in functions to torch equivalents."""
        self.generic_visit(node)

        if isinstance(node.func, ast.Name):
            name = node.func.id
            torch_map = {
                'abs': 'abs', 'max': 'max', 'min': 'min',
                'sum': 'sum', 'round': 'round',
            }
            if name in torch_map:
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='torch', ctx=ast.Load()),
                        attr=torch_map[name], ctx=ast.Load()),
                    args=node.args, keywords=[]
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
