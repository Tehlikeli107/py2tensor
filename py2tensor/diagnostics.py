"""
Py2Tensor Diagnostics: analyze a function and report what can/can't be GPU'd.
Suggests fixes for unsupported patterns.
"""
import ast
import inspect
import textwrap


def diagnose(fn):
    """Analyze a Python function and report GPU compatibility.

    Returns:
        dict with 'compatible', 'issues', 'suggestions', 'score'
    """
    source = inspect.getsource(fn)
    source = textwrap.dedent(source)

    # Remove decorators
    lines = source.split('\n')
    clean = []
    skip = True
    for line in lines:
        if skip and (line.strip().startswith('@') or line.strip() == ''):
            continue
        skip = False
        clean.append(line)
    source = '\n'.join(clean)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {'compatible': False, 'issues': [f'Syntax error: {e}'],
                'suggestions': ['Fix syntax first'], 'score': 0}

    analyzer = FunctionAnalyzer()
    analyzer.visit(tree)

    issues = []
    suggestions = []
    score = 100

    # Check each finding
    if analyzer.has_recursion:
        issues.append("Recursion detected (function calls itself)")
        suggestions.append("Convert to iterative: use for-loop instead of recursion")
        score -= 30

    if analyzer.has_while:
        issues.append(f"While loop detected ({analyzer.while_count}x)")
        suggestions.append("Will be auto-bounded to 64 iterations with @gpu.all")
        score -= 10

    if analyzer.has_try:
        issues.append("Try/except detected")
        suggestions.append("Will be auto-stripped with @gpu.all (safe execution)")
        score -= 5

    if analyzer.has_dict_literal:
        issues.append(f"Dict literal detected ({analyzer.dict_count}x)")
        suggestions.append("Will be auto-converted to tensor lookup with @gpu.all")
        score -= 5

    if analyzer.has_list_literal:
        issues.append(f"List literal detected ({analyzer.list_count}x)")
        suggestions.append("Will be auto-converted to tensor with @gpu.all")
        score -= 5

    if analyzer.has_string_ops:
        issues.append("String operations detected")
        suggestions.append("Strings -> int tensors for comparison. len() -> compile-time constant")
        score -= 15

    if analyzer.has_print:
        issues.append("Print/logging detected (side effect)")
        suggestions.append("Remove print statements or move to CPU wrapper")
        score -= 20

    if analyzer.has_file_io:
        issues.append("File I/O detected")
        suggestions.append("Use I/O pipeline: CPU reads, GPU processes, CPU writes")
        score -= 40

    if analyzer.has_class:
        issues.append("Class/method detected")
        suggestions.append("Extract method body into standalone function")
        score -= 25

    if analyzer.has_import:
        issues.append("Import statement inside function")
        suggestions.append("Move imports to top level")
        score -= 10

    if analyzer.has_global:
        issues.append("Global/nonlocal variable access")
        suggestions.append("Pass as function parameter instead")
        score -= 15

    # Positive findings
    positives = []
    if analyzer.has_math: positives.append("math.* functions (auto-converted)")
    if analyzer.has_if: positives.append(f"if/else ({analyzer.if_count}x -> torch.where)")
    if analyzer.has_for: positives.append(f"for loops ({analyzer.for_count}x -> unrolled)")
    if analyzer.has_augassign: positives.append("Augmented assign (+=, *=)")
    if analyzer.n_params > 1: positives.append(f"Multi-input ({analyzer.n_params} params -> batched)")

    # Best backend recommendation
    if score >= 90:
        if analyzer.has_for and analyzer.for_count > 0:
            best = "@gpu.triton (iterative -> fused kernel)"
        else:
            best = "@gpu.fast (simple math -> torch.compile)"
    elif score >= 60:
        best = "@gpu.all (auto dict/list/string conversion)"
    elif score >= 30:
        best = "@gpu (basic, with manual adjustments)"
    else:
        best = "Needs significant rewriting for GPU"

    score = max(0, min(100, score))

    result = {
        'compatible': score >= 30,
        'score': score,
        'issues': issues,
        'suggestions': suggestions,
        'positives': positives,
        'best_backend': best,
        'stats': {
            'params': analyzer.n_params,
            'if_count': analyzer.if_count,
            'for_count': analyzer.for_count,
            'lines': len(clean),
        }
    }

    # Print report
    print(f"\n{'='*50}")
    print(f"DIAGNOSIS: {fn.__name__}")
    print(f"{'='*50}")
    print(f"  Score: {score}/100 {'[COMPATIBLE]' if score >= 30 else '[NEEDS WORK]'}")
    print(f"  Best backend: {best}")

    if positives:
        print(f"\n  Supported:")
        for p in positives:
            print(f"    [+] {p}")

    if issues:
        print(f"\n  Issues:")
        for i, (issue, sugg) in enumerate(zip(issues, suggestions)):
            print(f"    [{i+1}] {issue}")
            print(f"        Fix: {sugg}")

    print(f"\n  Stats: {analyzer.n_params} params, {analyzer.if_count} if/else, "
          f"{analyzer.for_count} loops, {len(clean)} lines")

    return result


class FunctionAnalyzer(ast.NodeVisitor):
    """Analyze function for GPU compatibility."""

    def __init__(self):
        self.n_params = 0
        self.has_recursion = False
        self.has_while = False
        self.while_count = 0
        self.has_try = False
        self.has_dict_literal = False
        self.dict_count = 0
        self.has_list_literal = False
        self.list_count = 0
        self.has_string_ops = False
        self.has_print = False
        self.has_file_io = False
        self.has_class = False
        self.has_import = False
        self.has_global = False
        self.has_math = False
        self.has_if = False
        self.if_count = 0
        self.has_for = False
        self.for_count = 0
        self.has_augassign = False
        self.func_name = None

    def visit_FunctionDef(self, node):
        self.func_name = node.name
        self.n_params = len(node.args.args)
        self.generic_visit(node)

    def visit_While(self, node):
        self.has_while = True
        self.while_count += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.has_if = True
        self.if_count += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.has_for = True
        self.for_count += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.has_augassign = True
        self.generic_visit(node)

    def visit_Try(self, node):
        self.has_try = True
        self.generic_visit(node)

    def visit_Dict(self, node):
        self.has_dict_literal = True
        self.dict_count += 1
        self.generic_visit(node)

    def visit_List(self, node):
        self.has_list_literal = True
        self.list_count += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id == self.func_name:
                self.has_recursion = True
            if node.func.id == 'print':
                self.has_print = True
            if node.func.id == 'open':
                self.has_file_io = True

        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'math':
                    self.has_math = True
                if node.func.value.id in ('os', 'pathlib'):
                    self.has_file_io = True

        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.has_class = True
        self.generic_visit(node)

    def visit_Import(self, node):
        self.has_import = True
    def visit_ImportFrom(self, node):
        self.has_import = True

    def visit_Global(self, node):
        self.has_global = True
    def visit_Nonlocal(self, node):
        self.has_global = True


# ================================================================
if __name__ == '__main__':
    import math

    print("Testing diagnose() on various functions:\n")

    # Easy
    def simple(x):
        return x * x + 1
    diagnose(simple)

    # Medium
    def medium(x, y):
        if x > 0:
            return math.sin(x) + y
        else:
            return math.exp(y)
    diagnose(medium)

    # Complex
    def complex_fn(x):
        rates = {0: 0.05, 1: 0.10, 2: 0.15}
        for i in range(10):
            x = x * rates[0]
        if x > 100:
            return 100
        else:
            return x
    diagnose(complex_fn)

    # Hard
    def hard(x):
        if x <= 1: return 1
        return x * hard(x - 1)
    diagnose(hard)

    # Impossible
    def impossible(x):
        print(f"Processing {x}")
        with open("log.txt", "w") as f:
            f.write(str(x))
        return x
    diagnose(impossible)
