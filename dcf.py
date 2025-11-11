import os, re, datetime, pytz
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
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
#@tool 
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
        "jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec",
        "january","february","march","april","june","july","august","september",
        "october","november","december"
    }

    for match in matches:
        num_str, unit = match.groups()
        unit = (unit or "").lower()

        # Skip 4-digit years
        if len(num_str) == 4 and num_str.startswith(("19", "20")):
            continue

        # Skip day-of-month numbers near month names (e.g., "June 29, 2024")
        span_start, span_end = match.span()
        window = text[max(0, span_start-12):min(len(text), span_end+12)].lower()
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

# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    company = "Apple"
    year = "2024"

    print(f"\n🔍 Querying parser_data for {company} ({year})...\n")

    try:
        # --- Retrieve Relevant Chunks ---
        fcf_text = query_chunks(company, year, "free cash flow operating cash flow")
        price_text = query_chunks(company, year, "stock price market price share price trading at")
        growth_text = query_chunks(company, year, "growth rate terminal growth long-term growth")
        discount_text = query_chunks(company, year, "WACC cost of capital discount rate")
        # --- Retrieve Ticker Symbol ---
        symbol_text = query_chunks(company, year, "stock symbol ticker symbol company symbol")
        ticker_match = re.search(r"\b[A-Z]{1,5}\b", symbol_text)

        ticker = ticker_match.group(0) if ticker_match else None
        print(f"Ticker symbol found: {ticker if ticker else '⚠️ Not found'}")


        print("=== Retrieved Chunks (Preview) ===")
        print("FCF:", fcf_text[:250], "\n")
        print("Price:", price_text[:250], "\n")
        print("Growth:", growth_text[:250], "\n")
        print("Discount:", discount_text[:250], "\n")

        # --- Extract Numeric Values ---
        fcf_values = extract_number_with_unit(fcf_text, "cash flow")
        current_price_list = extract_number_with_unit(price_text, "price")
        growth_values = extract_number_with_unit(growth_text, "growth")
        discount_values = extract_number_with_unit(discount_text, "discount rate")

        # --- Handle Missing or Noisy Data ---
        fcf_values = fcf_values[:5] if fcf_values else [9e9, 9.5e9, 10e9, 10.5e9, 11e9]
        # Auto-correct likely quarterly FCFs (e.g., 20–30B for Apple
        if fcf_values and max(fcf_values) < 5e10:  # less than $50B
            print("⚠️ Detected low FCF — scaling by 4× to annualize quarterly figure.")
            fcf_values = [fcf * 4 for fcf in fcf_values]

        # ✅ Require that price has a dollar sign
        current_price = current_price_list[0] if current_price_list else 174.5
        terminal_growth_rate = growth_values[0] if growth_values else 0.025
        discount_rate = discount_values[0] if discount_values else 0.09

        # --- Sanity checks and defaults ---
        if not (-0.05 <= terminal_growth_rate <= 0.06):  # -5% to +6%
            terminal_growth_rate = 0.025  # reset to 2.5%
        if not (0.04 <= discount_rate <= 0.20):          # 4% to 20%
            discount_rate = 0.09  # reset to 9%
            # Ensure r > g for terminal value
        if discount_rate <= terminal_growth_rate:
            discount_rate = terminal_growth_rate + 0.01


        # --- Get Shares Outstanding using yfinance ---
        shares_outstanding = None
        if ticker:
            try:
                yf_ticker = yf.Ticker(ticker)
                year_int = int(year)
                start_date = f"{year_int}-01-01"
                end_date = f"{year_int}-12-31"
                # Try to pull year-specific shares
                shares_hist = yf_ticker.get_shares_full(start=start_date, end=end_date)
                if shares_hist is not None and not shares_hist.empty:
                    # Get the last reported share count during that year
                     shares_outstanding = int(shares_hist.iloc[-1])
                     print(f"Shares Outstanding ({year}): {shares_outstanding:,}")
                else:
                    # Fallback to latest
                    shares_outstanding = yf_ticker.info.get("sharesOutstanding")
                    print(f"⚠️ Using latest shares outstanding ({shares_outstanding:,}) — no data for {year}.")
            except Exception as e:
                print(f"⚠️ Could not retrieve shares outstanding for {ticker}: {e}")
        else:
            shares_outstanding = None


        

        print("\n✅ Parsed Inputs:")
        print(f"Free Cash Flows: {fcf_values}")
        print(f"Current Price: {current_price}")
        print(f"Terminal Growth Rate: {terminal_growth_rate}")
        print(f"Discount Rate: {discount_rate}\n")

        # --- Run DCF ---
        result = calculate_dcf(
        free_cash_flows=fcf_values,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        current_price=current_price,
        shares_outstanding=shares_outstanding
        )


        print("=== 📊 Valuation Results ===")
        for k, v in result.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"⚠️ Error during valuation: {e}")

"""
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

"""