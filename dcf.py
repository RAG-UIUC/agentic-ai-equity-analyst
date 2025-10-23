import requests
import pandas as pd
import uuid
from urllib.request import urlopen
import certifi, json
import os
from langchain_core.tools import tool

"""
Discounted Cash Flow (DCF) valuation tool for the Agentic AI Equity Analyst project.
This script can be run standalone, imported by the Valuation Agent, or wrapped as
a LangChain Tool. It computes intrinsic value, terminal value, and undervaluation %
based on projected Free Cash Flows (FCFs).
"""

from typing import List, Dict


def calculate_dcf(
    free_cash_flows: List[float],
    discount_rate: float,
    terminal_growth_rate: float,
    current_price: float
) -> Dict[str, float]:
    """
    Calculate the intrinsic value of a company using the Discounted Cash Flow model.

    Parameters
    ----------
    free_cash_flows : list of float
        Projected free cash flows for the next N years (usually 5).
    discount_rate : float
        Discount rate (e.g., WACC) as a decimal. Example: 0.10 for 10%.
    terminal_growth_rate : float
        Long-term perpetual growth rate (e.g., 0.02 for 2%).
    current_price : float
        Current market price per share or per company basis.

    Returns
    -------
    dict
        Dictionary containing intrinsic value, current price, undervaluation %, and terminal value.
    """

    n = len(free_cash_flows)

    # Step 1: Discount each year's FCF
    discounted_fcf = [
        fcf / ((1 + discount_rate) ** (i + 1))
        for i, fcf in enumerate(free_cash_flows)
    ]

    # Step 2: Compute terminal value using Gordon Growth Model
    terminal_value = (
        free_cash_flows[-1] * (1 + terminal_growth_rate)
    ) / (discount_rate - terminal_growth_rate)

    # Step 3: Discount terminal value to present
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** n)

    # Step 4: Sum discounted FCFs + discounted terminal value
    intrinsic_value = sum(discounted_fcf) + discounted_terminal_value

    # Step 5: Compare intrinsic value vs current price
    undervaluation_percent = ((intrinsic_value - current_price) / current_price) * 100

    return {
        "intrinsic_value": round(intrinsic_value, 2),
        "current_price": round(current_price, 2),
        "undervaluation_percent": round(undervaluation_percent, 2),
        "terminal_value": round(terminal_value, 2),
    }


try:
    from langchain.tools import tool 

    @tool("DCF Valuation Tool")
    def dcf_tool(
        free_cash_flows: List[float],
        discount_rate: float,
        terminal_growth_rate: float,
        current_price: float
    ) -> Dict[str, float]:
        """
        LangChain-compatible wrapper for the DCF valuation model.
        Enables use within Valuation Agents or LangGraph flows.
        """
        return calculate_dcf(free_cash_flows, discount_rate, terminal_growth_rate, current_price)

except ImportError:
    # LangChain not installed; skip wrapper definition.
    pass



if __name__ == "__main__":
    print("\n=== DCF Valuation Test ===")

    # Example input values (PER-SHARE free cash flow estimates, not millions)
    fcf_estimates = [9.0, 9.5, 10, 10.5, 11]   # Projected FCF per share
    discount_rate = 0.09                         # 9%
    terminal_growth_rate = 0.025                 # 2.5%
    current_price = 174.50                       # Current share price

    result = calculate_dcf(
        free_cash_flows=fcf_estimates,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        current_price=current_price
    )

    # Print clean formatted output
    print(f"Intrinsic value (DCF): ${result['intrinsic_value']:.2f}")
    print(f"Current price: ${result['current_price']:.2f}")

    # If undervalued_percent is negative, show “Overvalued by”
    if result["undervaluation_percent"] >= 0:
        print(f"Undervalued by: {result['undervaluation_percent']:.1f}%")
    else:
        print(f"Overvalued by: {abs(result['undervaluation_percent']):.1f}%")

