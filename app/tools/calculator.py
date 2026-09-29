"""
安全计算器工具模块。

使用 AST 解析实现安全表达式求值，
不使用 eval()，只支持基本算术运算。

LLM 只能看到:
    calculate(expression: str) -> str

Python 内部实现细节不暴露给 LLM。
"""

import ast
import logging
import operator
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 支持的二元运算符映射
# ──────────────────────────────────────────────
BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

# 支持的一元运算符
UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate(expression: str) -> str:
    """
    安全地计算数学表达式。

    只支持: + - * / % 和括号 ()，以及数字和一元正负号。
    不支持函数调用、属性访问、赋值等。

    参数:
        expression: 数学表达式字符串，如 "12345 * 6789"

    返回:
        计算结果字符串，或错误信息
    """
    if not expression or not expression.strip():
        return "错误：表达式不能为空。"

    expression = expression.strip()
    logger.info("Calculating expression: %s", expression)

    try:
        # 解析为 AST
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        logger.error("Division by zero in: %s", expression)
        return "错误：除零错误，表达式中存在除以零的运算。"
    except SyntaxError:
        logger.error("Syntax error in expression: %s", expression)
        return f"错误：表达式语法不正确 '{expression}'。"
    except ValueError as e:
        logger.error("Invalid expression: %s - %s", expression, e)
        return f"错误：{e}"
    except Exception as e:
        logger.error("Calculation failed: %s - %s", expression, e)
        return f"错误：无法计算表达式 '{expression}'。"

    logger.info("Calculation result: %s = %s", expression, result)
    return f"{expression} = {result}"


def _eval_node(node: ast.AST) -> float:
    """
    递归求值 AST 节点。

    只处理以下节点类型:
    - ast.Constant: 数字常量
    - ast.Num: 旧版 Python 数字节点 (兼容)
    - ast.BinOp: 二元运算
    - ast.UnaryOp: 一元运算
    - ast.Expression: 表达式根节点

    其他类型一律拒绝，保证安全。
    """
    # 数字常量
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")

    # 二元运算: a + b, a - b, a * b, a / b, a % b
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in BINARY_OPS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")

        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return BINARY_OPS[op_type](left, right)

    # 一元运算: +a, -a
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in UNARY_OPS:
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")

        operand = _eval_node(node.operand)
        return UNARY_OPS[op_type](operand)

    # 拒绝所有其他节点类型
    raise ValueError(f"不支持的表达式类型: {type(node).__name__}")
