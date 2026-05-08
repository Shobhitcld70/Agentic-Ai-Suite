from __future__ import annotations
from fastmcp import FastMCP
from dotenv import load_dotenv
import requests
import os

load_dotenv()

mcp = FastMCP("arith")

def _as_number(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError("Expected a number (int/float/numeric string)")

@mcp.tool()
async def add(a: float, b: float) -> float:
    return _as_number(a) + _as_number(b)

@mcp.tool()
async def subtract(a: float, b: float) -> float:
    return _as_number(a) - _as_number(b)

@mcp.tool()
async def multiply(a: float, b: float) -> float:
    return _as_number(a) * _as_number(b)

@mcp.tool()
async def divide(a: float, b: float) -> float:
    a = _as_number(a)
    b = _as_number(b)
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b

@mcp.tool()
async def get_stock_price(symbol: str) -> dict:
    url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=" + symbol + "&apikey=" + os.getenv("ALPHA_VANTAGE_API", "0EUE9XC9UXA05RF5")
    r = requests.get(url)
    return r.json()

if __name__ == "__main__":
    mcp.run()