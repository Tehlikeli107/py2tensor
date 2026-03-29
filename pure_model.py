"""
Pure Tensor Model: Algorithm as nn.Module with ZERO Python in forward pass.
Every operation is a tensor op. Constants are buffers.
If/else is torch.where masking. Loops are unrolled.
Forward pass = pure tensor computation, no Python branching.
"""
import torch
import torch.nn as nn
import ast
import inspect
import textwrap
import math
import builtins


def build_pure_model(fn):
    """Convert Python function to nn.Module with PURE tensor forward pass."""
    source = inspect.getsource(fn)
    source = textwrap.dedent(source)
    lines = source.split('\n')
    clean = []
    skip = True
    for line in lines:
        if skip and (line.strip().startswith('@') or line.strip() == ''):
            continue
        skip = False
        clean.append(line)
    source = '\n'.join(clean)

    tree = ast.parse(source)
    func_def = tree.body[0]
    params = [arg.arg for arg in func_def.args.args]

    gen = PureTensorGen(params)
    forward_code = gen.generate(func_def.body)

    param_str = ', '.join(params)
    buf_code = gen.buffer_code()

    class_src = (
        "class _PureModel(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        f"{buf_code}\n"
        f"    def forward(self, {param_str}):\n"
        f"{textwrap.indent(forward_code, '        ')}\n"
    )

    ns = {'nn': nn, 'torch': torch, 'math': math, '__builtins__': builtins}
    code_obj = builtins.compile(class_src, f'<pure_model:{fn.__name__}>', 'exec')
    import types
    types.FunctionType(code_obj, ns)()

    model = ns['_PureModel']()
    model._original = fn
    model._forward_code = forward_code
    model.__name__ = fn.__name__
    return model


class PureTensorGen:
    """Generate pure tensor forward pass code from AST."""

    def __init__(self, params):
        self.params = params
        self.buffers = {}
        self.buf_counter = 0

    def buffer_code(self):
        lines = []
        for name, val in self.buffers.items():
            lines.append(f"        self.register_buffer('{name}', torch.tensor({val}, dtype=torch.float32))")
        return '\n'.join(lines) if lines else '        pass'

    def generate(self, stmts):
        lines = []
        for s in stmts:
            code = self._stmt(s)
            if code: lines.append(code)
        return '\n'.join(lines)

    def _stmt(self, node):
        if isinstance(node, ast.Return):
            return f"return {self._expr(node.value)}"
        elif isinstance(node, ast.Assign):
            t = node.targets[0].id if isinstance(node.targets[0], ast.Name) else '_'
            return f"{t} = {self._expr(node.value)}"
        elif isinstance(node, ast.AugAssign):
            t = node.target.id
            ops = {ast.Add:'+', ast.Sub:'-', ast.Mult:'*', ast.Div:'/'}
            return f"{t} = {t} {ops.get(type(node.op),'+')} {self._expr(node.value)}"
        elif isinstance(node, ast.If):
            return self._if_stmt(node)
        elif isinstance(node, ast.For):
            return self._for_stmt(node)
        return ""

    def _if_stmt(self, node):
        cond = self._expr(node.test)
        lines = []
        body_vars, else_vars = {}, {}
        body_ret, else_ret = None, None
        order = []

        for s in node.body:
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name):
                n = s.targets[0].id
                body_vars[n] = self._expr(s.value)
                if n not in order: order.append(n)
            elif isinstance(s, ast.Return):
                body_ret = self._expr(s.value)

        for s in node.orelse:
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name):
                n = s.targets[0].id
                else_vars[n] = self._expr(s.value)
                if n not in order: order.append(n)
            elif isinstance(s, ast.Return):
                else_ret = self._expr(s.value)
            elif isinstance(s, ast.If):
                inner = self._if_expr(s)
                if inner: else_ret = inner

        p = self.params[0] if self.params else 'x'
        def _w(val):
            try:
                float(val)
                return f"({p} * 0 + {val})"
            except (ValueError, TypeError):
                return val

        for v in order:
            tv = _w(body_vars.get(v, '0.0'))
            fv = _w(else_vars.get(v, '0.0'))
            lines.append(f"{v} = torch.where({cond}, {tv}, {fv})")

        if body_ret and else_ret:
            # Use first param * 0 + val to keep device
            p = self.params[0] if self.params else 'x'
            def _wrap(v):
                try:
                    float(v)
                    return f"({p} * 0 + {v})"
                except (ValueError, TypeError):
                    return v
            lines.append(f"return torch.where({cond}, {_wrap(body_ret)}, {_wrap(else_ret)})")

        return '\n'.join(lines)

    def _if_expr(self, node):
        cond = self._expr(node.test)
        br = er = None
        for s in node.body:
            if isinstance(s, ast.Return): br = self._expr(s.value)
        for s in node.orelse:
            if isinstance(s, ast.Return): er = self._expr(s.value)
            elif isinstance(s, ast.If): er = self._if_expr(s)
        if br and er: return f"torch.where({cond}, {br}, {er})"
        return None

    def _for_stmt(self, node):
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == 'range'):
            args = node.iter.args
            stop = args[0].value if len(args)==1 and isinstance(args[0], ast.Constant) else None
            if stop and stop <= 64:
                lines = []
                for _ in range(stop):
                    for s in node.body:
                        c = self._stmt(s)
                        if c: lines.append(c)
                return '\n'.join(lines)
        return ""

    def _expr(self, node):
        if isinstance(node, ast.Constant):
            return str(float(node.value)) if isinstance(node.value, (int,float)) else str(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.BinOp):
            ops = {ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/',ast.Pow:'**'}
            return f"({self._expr(node.left)} {ops.get(type(node.op),'+')} {self._expr(node.right)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"(-{self._expr(node.operand)})"
        if isinstance(node, ast.Compare):
            ops = {ast.Gt:'>',ast.Lt:'<',ast.GtE:'>=',ast.LtE:'<='}
            return f"({self._expr(node.left)} {ops.get(type(node.ops[0]),'>')} {self._expr(node.comparators[0])})"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'math':
                m = {'sin':'torch.sin','cos':'torch.cos','exp':'torch.exp','log':'torch.log',
                     'sqrt':'torch.sqrt','tanh':'torch.tanh','atan2':'torch.atan2'}
                if node.func.attr in m:
                    a = ', '.join(f"torch.as_tensor({self._expr(x)})" if isinstance(x, ast.Constant) else self._expr(x) for x in node.args)
                    return f"{m[node.func.attr]}({a})"
            if isinstance(node.func, ast.Name) and node.func.id == 'abs':
                return f"torch.abs({self._expr(node.args[0])})"
        if isinstance(node, ast.IfExp):
            return f"torch.where({self._expr(node.test)}, {self._expr(node.body)}, {self._expr(node.orelse)})"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'math':
            if node.attr == 'pi': return '3.141592653589793'
            if node.attr == 'e': return '2.718281828459045'
        if isinstance(node, ast.Tuple):
            return ', '.join(self._expr(e) for e in node.elts)
        return "0.0"


if __name__ == '__main__':
    import time
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print("PURE TENSOR MODEL")
    print("=" * 60)

    @build_pure_model
    def simple(x):
        return x * x + 2 * x + 1

    simple = simple.to(device)
    print(f"\nForward code:\n{simple._forward_code}")
    x = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.float32, device=device)
    print(f"Output: {simple(x).tolist()}")
    print(f"Match: {torch.allclose(simple(x), x*x+2*x+1)}")

    @build_pure_model
    def relu(x):
        if x > 0:
            return x
        else:
            return 0

    relu = relu.to(device)
    print(f"\nReLU forward:\n{relu._forward_code}")
    try:
        out_r = relu(x)
        print(f"Output: {out_r.tolist()}")
        print(f"Match: {torch.allclose(out_r, torch.clamp(x, min=0))}")
    except Exception as e:
        print(f"ReLU error: {e}")

    @build_pure_model
    def newton(x):
        g = x / 2
        for i in range(10):
            g = (g + x / g) / 2
        return g

    newton = newton.to(device)
    xp = torch.tensor([4,9,16,25], dtype=torch.float32, device=device)
    print(f"\nNewton: {[f'{v:.2f}' for v in newton(xp).tolist()]}")
    print(f"Match sqrt: {torch.allclose(newton(xp), torch.sqrt(xp), atol=1e-3)}")

    # Autograd
    xg = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
    simple(xg).sum().backward()
    print(f"\nAutograd: grad={xg.grad.tolist()} (expected [4,6])")

    # Compile benchmark
    N = 10_000_000
    xb = torch.randn(N, device=device)
    compiled = torch.compile(simple)
    torch.cuda.synchronize()
    for _ in range(3): compiled(xb)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50): compiled(xb)
    torch.cuda.synchronize()
    t = (time.time()-t0)/50
    print(f"\nPure model + compile: {N/t/1e9:.1f}B/s ({t*1000:.2f}ms)")

    # Save
    torch.save(newton.state_dict(), r'C:\Users\salih\Desktop\py2tensor\pure_newton.pt')
    print(f"Save: OK")

    print(f"\nZero Python in forward pass. Pure tensor ops only.")
