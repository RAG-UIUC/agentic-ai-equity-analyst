import os, re, datetime, pytz
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict
import yfinance as yf

# ---------------------- ENV + EMBEDDINGS ---------------------- #
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

collection = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="parser_data",  # your parsed filings
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)


# ---------------------- DCF FUNCTION ---------------------- #
def calculate_dcf(
    free_cash_flows: List[float],
    discount_rate: float,
    terminal_growth_rate: float,
    current_price: float,
    shares_outstanding: float = None
) -> Dict[str, float]:
    """
    Calculate intrinsic value using the Discounted Cash Flow model.

    If shares_outstanding is provided, returns per-share valuation.
    """
    n = len(free_cash_flows)

    # Step 1: Discount each year's FCF
    discounted_fcf = [
        fcf / ((1 + discount_rate) ** (i + 1))
        for i, fcf in enumerate(free_cash_flows)
    ]

    # Step 2: Compute terminal value
    terminal_value = (
        free_cash_flows[-1] * (1 + terminal_growth_rate)
    ) / (discount_rate - terminal_growth_rate)

    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** n)
    intrinsic_value_total = sum(discounted_fcf) + discounted_terminal_value

    # Step 3: Adjust for per-share valuation
    if shares_outstanding and shares_outstanding > 0:
        intrinsic_value = intrinsic_value_total / shares_outstanding
    else:
        intrinsic_value = intrinsic_value_total

    undervaluation_percent = ((intrinsic_value - current_price) / current_price) * 100

    return {
        "intrinsic_value": round(intrinsic_value, 2),
        "current_price": round(current_price, 2),
        "undervaluation_percent": round(undervaluation_percent, 2),
        "terminal_value": round(terminal_value, 2),
        "shares_outstanding": shares_outstanding,
    }


# ---------------------- HELPERS ---------------------- #
def extract_number_with_unit(text: str, context: str = "") -> List[float]:
    """
    Extracts numeric values with optional units (billion, %, etc.)
    Handles FCF, prices, growth rates, discount rates, etc.
    Accepts decimal rates (0.09) while rejecting dates/months/years.
    """
    pattern = r"(\$?\d+(?:\.\d+)?)\s*(billion|million|thousand|%|percent)?"
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    values = []

    MONTHS = {
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "january", "february", "march", "april", "june", "july", "august", "september",
        "october", "november", "december"
    }

    for match in matches:
        num_str, unit = match.groups()
        unit = (unit or "").lower()

        # Skip 4-digit years
        if len(num_str) == 4 and num_str.startswith(("19", "20")):
            continue

        # Skip day-of-month numbers near month names (e.g., "June 29, 2024")
        span_start, span_end = match.span()
        window = text[max(0, span_start - 12):min(len(text), span_end + 12)].lower()
        if num_str.isdigit() and 1 <= int(num_str) <= 31:
            if any(m in window for m in MONTHS):
                continue

        # Price must have a $
        if "price" in context.lower() and not num_str.strip().startswith("$"):
            continue

        # Convert numeric part
        num = float(num_str.replace("$", ""))

        # Handle monetary scale
        if unit in {"billion", "million", "thousand"}:
            if unit == "billion":
                num *= 1e9
            elif unit == "million":
                num *= 1e6
            elif unit == "thousand":
                num *= 1e3

        # Handle percentages
        elif unit in {"%", "percent"}:
            if "growth" in context.lower() or "discount" in context.lower():
                num = num / 100
            else:
                continue

        # Handle bare numbers intelligently
        else:
            if "cash flow" in context.lower():
                # Skip unrealistically small numbers (e.g., 50 or 1000)
                if num < 1e8:
                    continue

            elif "price" in context.lower():
                # Prices must be reasonable
                if num < 1:
                    continue

            elif "growth" in context.lower() or "discount" in context.lower():
                # Allow decimals like 0.05–0.20 (5–20%)
                if 0 < num < 1:
                    pass  # keep as-is
                elif 1 <= num <= 100:
                    num = num / 100  # convert from percent
                else:
                    # Out-of-range (like 2024 or 500) → skip
                    continue

        values.append(num)

    return values


def query_chunks(company: str, year: str, query: str, k: int = 5) -> str:
    """Query relevant text from parser_data."""
    results = collection.similarity_search(f"{company} {year} {query}", k=k)
    return " ".join([r.page_content for r in results])


# ---------------------- NEW: YAHOO-BASED DCF HELPER ---------------------- #
def get_dcf_inputs_from_yahoo(ticker: str, years: int = 5) -> Dict[str, float]:
    """
    Build DCF input parameters using only Yahoo Finance data:
      - Free cash flows (last `years` from cashflow statement)
      - Current price
      - Terminal growth rate (FCF CAGR)
      - Discount rate (CAPM-style from beta)
      - Shares outstanding
    """
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info

    # Current price
    current_price = info.get("currentPrice")
    if current_price is None:
        hist = yf_ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"Could not determine current price for {ticker}")
        current_price = float(hist["Close"].iloc[-1])

    # Shares outstanding
    shares_outstanding = info.get("sharesOutstanding")

    # Free cash flows from cashflow statement
    cashflow = yf_ticker.cashflow
    if cashflow is None or cashflow.empty:
        raise ValueError(f"No cash flow statement available for {ticker}")

    fcf_row = None
    for name in ["Free Cash Flow", "FreeCashFlow", "FreeCashFlowToFirm"]:
        if name in cashflow.index:
            fcf_row = cashflow.loc[name]
            break

    if fcf_row is None:
        raise ValueError(f"Could not find Free Cash Flow row for {ticker}")

    fcf_row = fcf_row.dropna().iloc[:years]
    if fcf_row.empty:
        raise ValueError(f"Not enough FCF data for {ticker}")

    free_cash_flows = [float(v) for v in reversed(fcf_row.values)]

    # Growth: FCF CAGR
    if len(free_cash_flows) >= 2 and free_cash_flows[0] > 0:
        first, last = free_cash_flows[0], free_cash_flows[-1]
        n = len(free_cash_flows)
        cagr = (last / first) ** (1 / (n - 1)) - 1
        terminal_growth_rate = max(min(cagr, 0.15), -0.10)
    else:
        terminal_growth_rate = 0.025

    # Discount rate: CAPM-ish from beta
    beta = info.get("beta", None)
    risk_free = 0.04
    market_return = 0.09
    equity_risk_premium = market_return - risk_free

    if beta is not None:
        discount_rate = risk_free + beta * equity_risk_premium
    else:
        discount_rate = 0.09

    # Ensure terminal growth < discount rate
    if terminal_growth_rate >= discount_rate:
        terminal_growth_rate = discount_rate - 0.01

    return {
        "free_cash_flows": free_cash_flows,
        "current_price": current_price,
        "terminal_growth_rate": terminal_growth_rate,
        "discount_rate": discount_rate,
        "shares_outstanding": shares_outstanding,
    }


# ---------------------- MAIN: RUN BOTH METHODS + DIFFERENCE ---------------------- #
if __name__ == "__main__":
    company = "Apple"
    year = "2024"

    

    try:
        # ---------- 1) VECTOR / FILINGS-BASED PIPELINE (YOUR ORIGINAL LOGIC) ----------
        fcf_text = query_chunks(company, year, "free cash flow operating cash flow")
        price_text = query_chunks(company, year, "stock price market price share price trading at")
        growth_text = query_chunks(company, year, "growth rate terminal growth long-term growth")
        discount_text = query_chunks(company, year, "WACC cost of capital discount rate")
        symbol_text = query_chunks(company, year, "stock symbol ticker symbol company symbol")
        ticker_match = re.search(r"\b[A-Z]{1,5}\b", symbol_text)
        ticker = ticker_match.group(0) if ticker_match else None
        #print(f"Ticker symbol (from filings): {ticker if ticker else '⚠️ Not found'}")

        #print("=== Retrieved Chunks (Preview) ===")
        #print("FCF:", fcf_text[:250], "\n")
        #print("Price:", price_text[:250], "\n")
        #print("Growth:", growth_text[:250], "\n")
        #print("Discount:", discount_text[:250], "\n")

        # Extract numeric values
        fcf_values = extract_number_with_unit(fcf_text, "cash flow")
        current_price_list = extract_number_with_unit(price_text, "price")
        growth_values = extract_number_with_unit(growth_text, "growth")
        discount_values = extract_number_with_unit(discount_text, "discount rate")

        # Handle missing / noisy
        fcf_values = fcf_values[:5] if fcf_values else [9e9, 9.5e9, 10e9, 10.5e9, 11e9]
        if fcf_values and max(fcf_values) < 5e10:
            print(" Detected low FCF — scaling by 4× to annualize quarterly figure.")
            fcf_values = [fcf * 4 for fcf in fcf_values]

        current_price = current_price_list[0] if current_price_list else 174.5
        terminal_growth_rate = growth_values[0] if growth_values else 0.025
        discount_rate = discount_values[0] if discount_values else 0.09

        # Sanity checks
        if not (-0.05 <= terminal_growth_rate <= 0.06):
            terminal_growth_rate = 0.025
        if not (0.04 <= discount_rate <= 0.20):
            discount_rate = 0.09
        if discount_rate <= terminal_growth_rate:
            discount_rate = terminal_growth_rate + 0.01

        # Shares outstanding (year-specific if possible)
        shares_outstanding = None
        if ticker:
            try:
                yf_ticker = yf.Ticker(ticker)
                year_int = int(year)
                start_date = f"{year_int}-01-01"
                end_date = f"{year_int}-12-31"
                shares_hist = yf_ticker.get_shares_full(start=start_date, end=end_date)
                if shares_hist is not None and not shares_hist.empty:
                    shares_outstanding = int(shares_hist.iloc[-1])
                    print(f"Shares Outstanding ({year}): {shares_outstanding:,}")
                else:
                    shares_outstanding = yf_ticker.info.get("sharesOutstanding")
                    print(f" Using latest shares outstanding ({shares_outstanding:,}) — no data for {year}.")
            except Exception as e:
                print(f" Could not retrieve shares outstanding for {ticker}: {e}")

        print("\n Parsed Inputs (ChomaDB Method):")
        print(f"Free Cash Flows: {fcf_values}")
        print(f"Current Price: {current_price}")
        print(f"Terminal Growth Rate: {terminal_growth_rate}")
        print(f"Discount Rate: {discount_rate}\n")

        vector_result = calculate_dcf(
            free_cash_flows=fcf_values,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            current_price=current_price,
            shares_outstanding=shares_outstanding
        )

        print("=== ChromaDB Valuation ===")
        for k, v in vector_result.items():
            print(f"{k}: {v}")

        # ---------- 2) YAHOO-BASED DCF PIPELINE ----------
        yahoo_result = None
        if ticker:
            try:
                print("\n Running Yahoo-based DCF...")
                yahoo_inputs = get_dcf_inputs_from_yahoo(ticker, years=5)
                yahoo_result = calculate_dcf(
                    free_cash_flows=yahoo_inputs["free_cash_flows"],
                    discount_rate=yahoo_inputs["discount_rate"],
                    terminal_growth_rate=yahoo_inputs["terminal_growth_rate"],
                    current_price=yahoo_inputs["current_price"],
                    shares_outstanding=yahoo_inputs["shares_outstanding"],
                )

                print("\n Parsed Inputs (Yahoo Method):")
                print(f"Free Cash Flows: {yahoo_inputs['free_cash_flows']}")
                print(f"Current Price: {yahoo_inputs['current_price']}")
                print(f"Terminal Growth Rate: {yahoo_inputs['terminal_growth_rate']}")
                print(f"Discount Rate: {yahoo_inputs['discount_rate']}")
                print(f"Shares Outstanding: {yahoo_inputs['shares_outstanding']}")

                print("\n=== Yahoo-Based Valuation ===")
                for k, v in yahoo_result.items():
                    print(f"{k}: {v}")

            except Exception as e:
                print(f" Error during Yahoo-based DCF: {e}")
        else:
            print("\n Skipping Yahoo-based DCF — no ticker symbol found from filings.")

        # ---------- 3) DIFFERENCE BETWEEN METHODS ----------
        if yahoo_result is not None:
            iv_vector = vector_result["intrinsic_value"]
            iv_yahoo = yahoo_result["intrinsic_value"]
            abs_diff = round(iv_yahoo - iv_vector, 2)
            avg_iv = (iv_vector + iv_yahoo) / 2 if (iv_vector + iv_yahoo) != 0 else None
            pct_diff = round(abs_diff / avg_iv * 100, 2) if avg_iv else None

            print("\n===  Method Comparison ===")
            print(f"Chroma DB IV: {iv_vector}")
            print(f"Yahoo IV:         {iv_yahoo}")
            print(f"Absolute Diff:    {abs_diff}")
            print(f"% Diff vs Avg:    {pct_diff}%")

    except Exception as e:
        print(f" Error during valuation: {e}")
