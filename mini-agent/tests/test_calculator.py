"""
Calculator 工具单元测试。
"""

import pytest
from app.tools.calculator import calculate


class TestCalculate:
    """安全计算器测试。"""

    def test_addition(self) -> None:
        result = calculate("2 + 3")
        assert "5" in result

    def test_subtraction(self) -> None:
        result = calculate("10 - 4")
        assert "6" in result

    def test_multiplication(self) -> None:
        result = calculate("12345 * 6789")
        assert "83810205" in result

    def test_division(self) -> None:
        result = calculate("100 / 4")
        assert "25" in result

    def test_modulo(self) -> None:
        result = calculate("10 % 3")
        assert "1" in result

    def test_complex_expression(self) -> None:
        result = calculate("100 / 4 + 25")
        assert "50" in result

    def test_parentheses(self) -> None:
        result = calculate("(2 + 3) * 4")
        assert "20" in result

    def test_negative_number(self) -> None:
        result = calculate("-5 + 10")
        assert "5" in result

    def test_decimal_result(self) -> None:
        result = calculate("7 / 2")
        assert "3.5" in result

    def test_division_by_zero(self) -> None:
        result = calculate("10 / 0")
        assert "错误" in result

    def test_empty_expression(self) -> None:
        result = calculate("")
        assert "错误" in result

    def test_unsafe_expression(self) -> None:
        """测试不安全的表达式被拒绝。"""
        result = calculate("__import__('os').system('ls')")
        assert "错误" in result

    def test_unsupported_function(self) -> None:
        """测试不支持的表达式类型被拒绝。"""
        result = calculate("abs(-5)")
        assert "错误" in result
