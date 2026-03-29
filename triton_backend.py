"""
Py2Tensor Triton Backend: ALL ops fused into single GPU kernel
=============================================================
Instead of generating PyTorch ops (each = separate CUDA kernel),
generate ONE Triton kernel that does everything in one pass.
"""
import torch
import ast
import inspect
import textwrap
import math
import time

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class TritonCodeGen:
    """Convert Python AST to Triton kernel body code."""

    def __init__(self, params):
        self.params = params

    def visit_body(self, stmts):
        lines = []
        for stmt in stmts:
            code = self.visit_stmt(stmt)
            if code:
                lines.append(code)
        return '\n'.join(lines)

    def visit_stmt(self, node):
        if isinstance(node, ast.Return):
            return f"result = {self.visit_expr(node.value)}"
        elif isinstance(node, ast.Assign):
            target = node.targets[0].id if isinstance(node.targets[0], ast.Name) else '_'
            return f"{target} = {self.visit_expr(node.value)}"
        elif isinstance(node, ast.If):
            return self.visit_if(node)
        elif isinstance(node, ast.AugAssign):
            target = node.target.id
            op = self.visit_op(node.op)
            return f"{target} = {target} {op} {self.visit_expr(node.value)}"
        elif isinstance(node, ast.For):
            return self.visit_for(node)
        return ""

    def visit_for(self, node):
        """Unroll for i in range(N) into repeated statements."""
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == 'range'):
            args = node.iter.args
            start, stop, step = 0, None, 1
            if len(args) == 1 and isinstance(args[0], ast.Constant):
                stop = args[0].value
            elif len(args) >= 2 and isinstance(args[0], ast.Constant) and isinstance(args[1], ast.Constant):
                start, stop = args[0].value, args[1].value
                if len(args) == 3 and isinstance(args[2], ast.Constant):
                    step = args[2].value
            if stop is not None and abs(stop - start) <= 64:
                var = node.target.id if isinstance(node.target, ast.Name) else None
                lines = []
                import re, copy
                for i in range(start, stop, step):
                    for stmt in node.body:
                        # Deep copy to avoid modifying original AST
                        s = copy.deepcopy(stmt)
                        # Replace loop var at AST level for correctness
                        if var:
                            class _R(ast.NodeTransformer):
                                def visit_Name(self, n):
                                    if n.id == var: return ast.Constant(value=i)
                                    return n
                            s = _R().visit(s)
                        code = self.visit_stmt(s)
                        lines.append(code)
                return '\n'.join(lines)
        return "# unsupported for loop"

    def visit_if(self, node):
        cond = self.visit_expr(node.test)
        lines = []
        body_map = {}
        else_map = {}
        body_order = []
        body_return = None
        else_return = None

        for s in node.body:
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name):
                name = s.targets[0].id
                body_map[name] = self.visit_expr(s.value)
                if name not in body_order: body_order.append(name)
            elif isinstance(s, ast.Return):
                body_return = self.visit_expr(s.value)

        for s in node.orelse:
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name):
                name = s.targets[0].id
                else_map[name] = self.visit_expr(s.value)
                if name not in body_order: body_order.append(name)
            elif isinstance(s, ast.Return):
                else_return = self.visit_expr(s.value)
            elif isinstance(s, ast.If):
                lines.append(self.visit_if(s))

        for var in body_order:
            tv = body_map.get(var, '0.0')
            fv = else_map.get(var, '0.0')
            lines.append(f"{var} = tl.where({cond}, {tv}, {fv})")

        if body_return is not None and else_return is not None:
            lines.append(f"result = tl.where({cond}, {body_return}, {else_return})")
        elif body_return is not None:
            lines.append(f"result = tl.where({cond}, {body_return}, result)")

        return '\n'.join(lines)

    def visit_expr(self, node):
        if isinstance(node, ast.Constant):
            return f"{float(node.value)}" if isinstance(node.value, (int, float)) else str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            return f"({self.visit_expr(node.left)} {self.visit_op(node.op)} {self.visit_expr(node.right)})"
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"(-{self.visit_expr(node.operand)})"
        elif isinstance(node, ast.Compare):
            return f"({self.visit_expr(node.left)} {self.visit_cmpop(node.ops[0])} {self.visit_expr(node.comparators[0])})"
        elif isinstance(node, ast.Call):
            return self.visit_call(node)
        elif isinstance(node, ast.IfExp):
            return f"tl.where({self.visit_expr(node.test)}, {self.visit_expr(node.body)}, {self.visit_expr(node.orelse)})"
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'math':
                if node.attr == 'pi': return '3.141592653589793'
                if node.attr == 'e': return '2.718281828459045'
        return "0.0"

    def visit_call(self, node):
        args = [self.visit_expr(a) for a in node.args]
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'math':
                m = {'sin':'tl.sin','cos':'tl.cos','exp':'tl.exp','log':'tl.log',
                     'sqrt':'tl.sqrt','abs':'tl.abs','tanh':'tl.tanh'}
                if node.func.attr in m:
                    return f"{m[node.func.attr]}({args[0]})"
                if node.func.attr == 'atan2':
                    return f"tl.atan2({args[0]}, {args[1]})"
        if isinstance(node.func, ast.Name):
            if node.func.id == 'abs': return f"tl.abs({args[0]})"
        return f"({', '.join(args)})"

    def visit_op(self, op):
        return {ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/',ast.Pow:'**',ast.Mod:'%'}.get(type(op),'+')

    def visit_cmpop(self, op):
        return {ast.Gt:'>',ast.Lt:'<',ast.GtE:'>=',ast.LtE:'<=',ast.Eq:'==',ast.NotEq:'!='}.get(type(op),'>')


def tensorize_triton(fn):
    """Decorator: generates a SINGLE fused Triton kernel from Python function."""
    if not HAS_TRITON:
        # Fallback to regular tensorize
        from py2tensor import tensorize
        return tensorize(fn)

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

    gen = TritonCodeGen(params)
    body_code = gen.visit_body(func_def.body)

    param_ptrs = ', '.join(f'{p}_ptr' for p in params)
    loads = '\n    '.join(f'{p} = tl.load({p}_ptr + offsets, mask=mask, other=0.0)' for p in params)

    # Indent body code to 4 spaces (inside function)
    body_lines = body_code.strip().split('\n')
    indented_body = '\n'.join('    ' + line.strip() for line in body_lines if line.strip())

    kernel_src = f"""@triton.jit
def _fused_kernel({param_ptrs}, output_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    {loads}
{indented_body}
    tl.store(output_ptr + offsets, result, mask=mask)
"""
    # Triton needs source code on disk (inspect.getsource requirement)
    import tempfile, importlib, os
    tmpdir = tempfile.mkdtemp()
    modname = f"_triton_kernel_{fn.__name__}"
    filepath = os.path.join(tmpdir, f"{modname}.py")
    with open(filepath, 'w') as f:
        f.write("import triton\nimport triton.language as tl\nimport math\n\n")
        f.write(kernel_src)

    import sys
    sys.path.insert(0, tmpdir)
    mod = importlib.import_module(modname)
    kernel = mod._fused_kernel
    sys.path.pop(0)

    def wrapper(*args):
        tensors = []
        for a in args:
            if isinstance(a, torch.Tensor):
                tensors.append(a.float().contiguous())
            else:
                tensors.append(torch.tensor(float(a), device='cuda', dtype=torch.float32))

        n = max(t.numel() for t in tensors)
        tensors = [t.expand(n).contiguous() if t.numel() < n else t for t in tensors]
        output = torch.empty(n, device='cuda', dtype=torch.float32)

        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        kernel[grid](*tensors, output, n, BLOCK=BLOCK)
        return output

    wrapper._original = fn
    wrapper._triton_source = kernel_src
    wrapper.__name__ = fn.__name__
    return wrapper


# ================================================================
if __name__ == '__main__':
    from py2tensor import tensorize

    print("=" * 60)
    print("TRITON vs PYTORCH BENCHMARK")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    N = 10_000_000
    WARMUP, ROUNDS = 5, 50

    def bench(name, fn, *args):
        torch.cuda.synchronize()
        for _ in range(WARMUP): fn(*args)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(ROUNDS): out = fn(*args)
        torch.cuda.synchronize()
        t = (time.time() - t0) / ROUNDS
        print(f"  {name:<35} {t*1000:>7.2f}ms {N/t/1e9:>6.1f}B/s")
        return t, out

    x = torch.randn(N, device='cuda')

    # Test 1
    print(f"\n--- sin(x) * exp(-x*0.1) ---")

    @tensorize
    def f_pt(x): return math.sin(x) * math.exp(-x * 0.1)

    @tensorize(compile=True)
    def f_co(x): return math.sin(x) * math.exp(-x * 0.1)

    @tensorize_triton
    def f_tr(x): return math.sin(x) * math.exp(-x * 0.1)

    t1, o1 = bench("@tensorize (PyTorch ops)", f_pt, x)
    t2, o2 = bench("@tensorize(compile=True)", f_co, x)
    t3, o3 = bench("@tensorize_triton (FUSED)", f_tr, x)
    print(f"  Match: {torch.allclose(o1, o3, atol=1e-4)}")
    print(f"  Triton vs PyTorch: {t1/t3:.2f}x | Triton vs compile: {t2/t3:.2f}x")

    # Test 2
    print(f"\n--- if x>0: x*x+1 else: exp(x) ---")

    @tensorize
    def g_pt(x):
        if x > 0: return x * x + 1
        else: return math.exp(x)

    @tensorize(compile=True)
    def g_co(x):
        if x > 0: return x * x + 1
        else: return math.exp(x)

    @tensorize_triton
    def g_tr(x):
        if x > 0: return x * x + 1
        else: return math.exp(x)

    t1, o1 = bench("@tensorize (PyTorch ops)", g_pt, x)
    t2, o2 = bench("@tensorize(compile=True)", g_co, x)
    t3, o3 = bench("@tensorize_triton (FUSED)", g_tr, x)
    print(f"  Match: {torch.allclose(o1, o3, atol=1e-4)}")
    print(f"  Triton vs PyTorch: {t1/t3:.2f}x | Triton vs compile: {t2/t3:.2f}x")

    # Test 3: Gaussian PDF
    print(f"\n--- Gaussian PDF ---")

    @tensorize
    def h_pt(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @tensorize(compile=True)
    def h_co(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @tensorize_triton
    def h_tr(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    t1, o1 = bench("@tensorize (PyTorch ops)", h_pt, x)
    t2, o2 = bench("@tensorize(compile=True)", h_co, x)
    t3, o3 = bench("@tensorize_triton (FUSED)", h_tr, x)
    print(f"  Match: {torch.allclose(o1, o3, atol=1e-4)}")
    print(f"  Triton vs PyTorch: {t1/t3:.2f}x | Triton vs compile: {t2/t3:.2f}x")
