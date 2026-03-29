"""
Py2Tensor Model Backend: Python function -> nn.Module
=====================================================
Every operation becomes a layer. Weights are fixed constants.
Forward pass = the algorithm. No training needed.

Benefits over function backend:
- Composable: embed inside other models
- Autograd: native gradient support
- Serializable: torch.save/load
- Optimizable: torch.compile on module
- Inspectable: print(model) shows architecture
"""
import torch
import torch.nn as nn
import ast
import inspect
import textwrap
import math


def tensorize_model(fn):
    """Convert Python function to nn.Module with fixed weights."""
    source = inspect.getsource(fn)
    source = textwrap.dedent(source)

    # Remove decorator
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

    # Build the module
    builder = ModuleBuilder(params, func_def.body)
    module = builder.build()
    module._original = fn
    module._source = source
    return module


class AlgorithmModule(nn.Module):
    """A nn.Module that executes a Python algorithm as tensor ops."""

    def __init__(self, operations, constants, param_names):
        super().__init__()
        self.operations = operations  # list of (op_type, args)
        self.param_names = param_names

        # Register constants as buffers (saved with model, moved with .to())
        for name, value in constants.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def forward(self, *args):
        # Build variable namespace
        ns = {}
        for i, name in enumerate(self.param_names):
            ns[name] = args[i].float() if isinstance(args[i], torch.Tensor) else torch.tensor(float(args[i]))

        # Add constants
        for name, buf in self.named_buffers():
            ns[name] = buf

        # Execute operations
        for op in self.operations:
            result = self._exec_op(op, ns)
            if result is not None:
                return result

        return ns.get('result', torch.tensor(0.0))

    def _exec_op(self, op, ns):
        op_type = op[0]

        if op_type == 'assign':
            var_name, expr = op[1], op[2]
            ns[var_name] = self._eval_expr(expr, ns)

        elif op_type == 'return':
            return self._eval_expr(op[1], ns)

        elif op_type == 'where':
            var_name, cond_expr, true_expr, false_expr = op[1], op[2], op[3], op[4]
            cond = self._eval_expr(cond_expr, ns)
            true_val = self._eval_expr(true_expr, ns)
            false_val = self._eval_expr(false_expr, ns)
            ns[var_name] = torch.where(cond, true_val, false_val)

        elif op_type == 'where_return':
            cond_expr, true_expr, false_expr = op[1], op[2], op[3]
            cond = self._eval_expr(cond_expr, ns)
            true_val = self._eval_expr(true_expr, ns)
            false_val = self._eval_expr(false_expr, ns)
            return torch.where(cond, true_val, false_val)

        return None

    def _eval_expr(self, expr, ns):
        if isinstance(expr, (int, float)):
            # Get device from any input tensor in namespace
            dev = 'cpu'
            for v in ns.values():
                if isinstance(v, torch.Tensor):
                    dev = v.device
                    break
            return torch.tensor(float(expr), device=dev)
        if isinstance(expr, str):
            return ns.get(expr, torch.tensor(0.0))
        if isinstance(expr, tuple):
            op = expr[0]
            if op == 'add': return self._eval_expr(expr[1], ns) + self._eval_expr(expr[2], ns)
            if op == 'sub': return self._eval_expr(expr[1], ns) - self._eval_expr(expr[2], ns)
            if op == 'mul': return self._eval_expr(expr[1], ns) * self._eval_expr(expr[2], ns)
            if op == 'div': return self._eval_expr(expr[1], ns) / self._eval_expr(expr[2], ns)
            if op == 'pow': return self._eval_expr(expr[1], ns) ** self._eval_expr(expr[2], ns)
            if op == 'neg': return -self._eval_expr(expr[1], ns)
            if op == 'gt': return self._eval_expr(expr[1], ns) > self._eval_expr(expr[2], ns)
            if op == 'lt': return self._eval_expr(expr[1], ns) < self._eval_expr(expr[2], ns)
            if op == 'gte': return self._eval_expr(expr[1], ns) >= self._eval_expr(expr[2], ns)
            if op == 'lte': return self._eval_expr(expr[1], ns) <= self._eval_expr(expr[2], ns)
            if op == 'sin': return torch.sin(self._eval_expr(expr[1], ns))
            if op == 'cos': return torch.cos(self._eval_expr(expr[1], ns))
            if op == 'exp': return torch.exp(self._eval_expr(expr[1], ns))
            if op == 'log': return torch.log(self._eval_expr(expr[1], ns))
            if op == 'sqrt': return torch.sqrt(self._eval_expr(expr[1], ns))
            if op == 'tanh': return torch.tanh(self._eval_expr(expr[1], ns))
            if op == 'abs': return torch.abs(self._eval_expr(expr[1], ns))
            if op == 'where':
                c = self._eval_expr(expr[1], ns)
                t = self._eval_expr(expr[2], ns)
                f = self._eval_expr(expr[3], ns)
                return torch.where(c, t, f)
        return torch.tensor(0.0)


class ModuleBuilder:
    """Build nn.Module from Python AST."""

    def __init__(self, params, body):
        self.params = params
        self.body = body
        self.operations = []
        self.constants = {}
        self.const_counter = 0

    def build(self):
        for stmt in self.body:
            self._process_stmt(stmt)
        return AlgorithmModule(self.operations, self.constants, self.params)

    def _process_stmt(self, node):
        if isinstance(node, ast.Return):
            expr = self._process_expr(node.value)
            self.operations.append(('return', expr))

        elif isinstance(node, ast.Assign):
            target = node.targets[0].id if isinstance(node.targets[0], ast.Name) else 'result'
            expr = self._process_expr(node.value)
            self.operations.append(('assign', target, expr))

        elif isinstance(node, ast.AugAssign):
            target = node.target.id
            op_map = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div'}
            op = op_map.get(type(node.op), 'add')
            expr = self._process_expr(node.value)
            self.operations.append(('assign', target, (op, target, expr)))

        elif isinstance(node, ast.If):
            self._process_if(node)

        elif isinstance(node, ast.For):
            self._process_for(node)

    def _process_if(self, node):
        cond = self._process_expr(node.test)

        # Return in both branches
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Return) and
            len(node.orelse) == 1 and isinstance(node.orelse[0], ast.Return)):
            true_val = self._process_expr(node.body[0].value)
            false_val = self._process_expr(node.orelse[0].value)
            self.operations.append(('where_return', cond, true_val, false_val))
            return

        # Assignments
        body_assigns = [s for s in node.body if isinstance(s, ast.Assign)]
        else_assigns = [s for s in node.orelse if isinstance(s, ast.Assign)]

        body_map = {s.targets[0].id: self._process_expr(s.value) for s in body_assigns if isinstance(s.targets[0], ast.Name)}
        else_map = {s.targets[0].id: self._process_expr(s.value) for s in else_assigns if isinstance(s.targets[0], ast.Name)}

        # Nested if in else
        for s in node.orelse:
            if isinstance(s, ast.If):
                # Process nested if separately
                self._process_if_nested(s, cond, body_map)
                return

        for var in set(body_map) | set(else_map):
            true_val = body_map.get(var, var)
            false_val = else_map.get(var, var)
            self.operations.append(('where', var, cond, true_val, false_val))

    def _process_if_nested(self, node, outer_cond, outer_body):
        """Handle nested if/else by chaining where operations."""
        inner_cond = self._process_expr(node.test)

        # For returns
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Return)):
            inner_true = self._process_expr(node.body[0].value)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.Return):
                inner_false = self._process_expr(node.orelse[0].value)
                # where(outer, outer_true, where(inner, inner_true, inner_false))
                inner_where = ('where', inner_cond, inner_true, inner_false)
                for var, val in outer_body.items():
                    self.operations.append(('where_return', outer_cond, val, inner_where))
                    return
            elif len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # Triple nested — handle recursively
                pass

        # Fallback: process as separate operations
        inner_body = {s.targets[0].id: self._process_expr(s.value) for s in node.body if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)}
        inner_else = {s.targets[0].id: self._process_expr(s.value) for s in node.orelse if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)}

        for var in set(outer_body) | set(inner_body) | set(inner_else):
            outer_val = outer_body.get(var, var)
            inner_true = inner_body.get(var, var)
            inner_false = inner_else.get(var, var)
            inner_where = ('where', inner_cond, inner_true, inner_false)
            self.operations.append(('where', var, outer_cond, outer_val, inner_where))

    def _process_for(self, node):
        """Unroll for-loop."""
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == 'range'):
            args = node.iter.args
            start, stop, step = 0, None, 1
            if len(args) == 1 and isinstance(args[0], ast.Constant): stop = args[0].value
            elif len(args) >= 2: start, stop = args[0].value, args[1].value

            if stop and abs(stop - start) <= 64:
                for i in range(start, stop, step):
                    for stmt in node.body:
                        self._process_stmt(stmt)

    def _process_expr(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return ('neg', self._process_expr(node.operand))
        if isinstance(node, ast.BinOp):
            op_map = {ast.Add:'add', ast.Sub:'sub', ast.Mult:'mul', ast.Div:'div', ast.Pow:'pow'}
            op = op_map.get(type(node.op), 'add')
            return (op, self._process_expr(node.left), self._process_expr(node.right))
        if isinstance(node, ast.Compare):
            op_map = {ast.Gt:'gt', ast.Lt:'lt', ast.GtE:'gte', ast.LtE:'lte'}
            op = op_map.get(type(node.ops[0]), 'gt')
            return (op, self._process_expr(node.left), self._process_expr(node.comparators[0]))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'math':
                    fn_map = {'sin':'sin','cos':'cos','exp':'exp','log':'log','sqrt':'sqrt','tanh':'tanh','abs':'abs'}
                    if node.func.attr in fn_map:
                        return (fn_map[node.func.attr], self._process_expr(node.args[0]))
            if isinstance(node.func, ast.Name):
                if node.func.id == 'abs':
                    return ('abs', self._process_expr(node.args[0]))
        if isinstance(node, ast.IfExp):
            return ('where', self._process_expr(node.test), self._process_expr(node.body), self._process_expr(node.orelse))
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'math':
                if node.attr == 'pi': return 3.141592653589793
                if node.attr == 'e': return 2.718281828459045
        return 0


# ================================================================
if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print("MODEL BACKEND: Python function -> nn.Module")
    print("=" * 60)

    # Test 1: Simple
    @tensorize_model
    def simple(x):
        return x * x + 2 * x + 1

    print(f"\n--- simple(x) = x^2 + 2x + 1 ---")
    print(f"Type: {type(simple)}")
    print(f"Module: {simple}")
    simple = simple.to(device)

    x = torch.tensor([0, 1, 2, 3, -1], dtype=torch.float32, device=device)
    out = simple(x)
    expected = x*x + 2*x + 1
    print(f"Output: {out.tolist()}")
    print(f"Expected: {expected.tolist()}")
    print(f"Match: {torch.allclose(out, expected)}")

    # Test 2: If/else
    @tensorize_model
    def relu(x):
        if x > 0:
            return x
        else:
            return 0

    print(f"\n--- relu(x) ---")
    relu = relu.to(device)
    x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32, device=device)
    out = relu(x)
    print(f"Output: {out.tolist()}")
    print(f"Match: {torch.allclose(out, torch.clamp(x, min=0))}")

    # Test 3: Math functions
    @tensorize_model
    def gaussian(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    print(f"\n--- gaussian(x) ---")
    gaussian = gaussian.to(device)
    x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32, device=device)
    out = gaussian(x)
    expected_g = torch.exp(-0.5 * x * x) / math.sqrt(2 * 3.14159265)
    print(f"Output: {[f'{v:.4f}' for v in out.tolist()]}")
    print(f"Match: {torch.allclose(out, expected_g, atol=1e-3)}")

    # Test 4: Newton sqrt as MODEL
    @tensorize_model
    def newton_model(x):
        g = x / 2
        for i in range(10):
            g = (g + x / g) / 2
        return g

    print(f"\n--- newton_model(x) = sqrt(x) via 10 iterations ---")
    newton_model = newton_model.to(device)
    x = torch.tensor([4, 9, 16, 25, 100], dtype=torch.float32, device=device)
    out = newton_model(x)
    expected_n = torch.sqrt(x)
    print(f"Output: {[f'{v:.4f}' for v in out.tolist()]}")
    print(f"Expected: {[f'{v:.4f}' for v in expected_n.tolist()]}")
    print(f"Match: {torch.allclose(out, expected_n, atol=1e-3)}")

    # Test 5: Autograd
    print(f"\n--- AUTOGRAD on nn.Module ---")
    x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    y = simple(x)
    y.sum().backward()
    print(f"d/dx(x^2+2x+1) at x={x.data.tolist()}: grad={x.grad.tolist()}")
    print(f"Expected: {(2*x.data+2).tolist()}")

    # Test 6: Save/Load
    print(f"\n--- SAVE/LOAD ---")
    torch.save(newton_model.state_dict(), r"C:\Users\salih\Desktop\py2tensor\newton_model.pt")
    print(f"Saved newton_model.pt")

    # Test 7: Compose with other modules
    print(f"\n--- COMPOSE: nn.Sequential ---")
    pipeline = nn.Sequential(
        simple,  # x -> x^2+2x+1
        relu,    # -> clamp(0)
    ).to(device)
    x = torch.tensor([-3, -1, 0, 1, 3], dtype=torch.float32, device=device)
    out = pipeline(x)
    print(f"Sequential(simple, relu) at x={x.tolist()}: {out.tolist()}")

    # Benchmark
    print(f"\n--- BENCHMARK: 10M elements ---")
    N = 10_000_000
    x = torch.randn(N, device=device)
    newton_model = newton_model.to(device)

    # Warmup
    for _ in range(3): simple(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(30): simple(x)
    torch.cuda.synchronize()
    t = (time.time() - t0) / 30
    print(f"  simple: {N/t/1e9:.1f}B/s")

    x_pos = torch.rand(N, device=device) * 100 + 0.1
    for _ in range(3): newton_model(x_pos)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10): newton_model(x_pos)
    torch.cuda.synchronize()
    t = (time.time() - t0) / 10
    print(f"  newton:  {N/t/1e6:.0f}M/s (10 iter, model overhead)")

    print(f"\n  Note: Model backend has interpretation overhead.")
    print(f"  For max speed, use @tensorize or @tensorize_triton.")
    print(f"  Model backend is for: composability, save/load, embedding in larger models.")
