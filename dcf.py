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
    Extract numbers with unit detection and contextual filtering.
    Example: '$27.9 billion' -> 27900000000.0
    """
    # Capture the original text span so we can check for $ directly
    pattern = r"(\$?\d+(?:\.\d+)?)\s*(billion|million|thousand|%)?"
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    values = []

    for match in matches:
        num_str, unit = match.groups()

        # Skip years (e.g., 2020–2030)
        if len(num_str) == 4 and num_str.startswith(("19", "20")):
            continue

        # --- 1️⃣ Handle price: must have a $ in front of this specific match
        if "price" in context.lower():
            if not num_str.strip().startswith("$"):
                continue  # reject 51 million, 14, etc.

        # --- 2️⃣ Convert to numeric
        num = float(num_str.replace("$", ""))

        # --- 3️⃣ Apply units
        if unit:
            unit = unit.lower()
            if "billion" in unit:
                num *= 1e9
            elif "million" in unit:
                num *= 1e6
            elif "thousand" in unit:
                num *= 1e3
            elif "%" in unit:
                if "growth" in context.lower() or "discount" in context.lower():
                    num = num / 100
                else:
                    continue
        else:
            # context-specific small filters
            if "cash flow" in context.lower() and num < 1e8:
                continue
            if "growth" in context.lower() and num > 1:
                num = num / 100
            if "discount" in context.lower() and num > 1:
                num = num / 100

        values.append(num)

    return values

def query_chunks(company: str, year: str, query: str, k: int = 5) -> str:
    """Query relevant text from parser_data."""
    results = collection.similarity_search(f"{company} {year} {query}", k=k)
    return " ".join([r.page_content for r in results])

# ---------------------- MAIN ---------------------- #

@tool 
def find_dcf(company: str, year: str):
    """Perform a Discounted Cash Flow analysis of a given company in a specific year.  
   
   """
    
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

        # ✅ Require that price has a dollar sign
        current_price = current_price_list[0] if current_price_list else 174.5
        terminal_growth_rate = growth_values[0] if growth_values else 0.025
        discount_rate = discount_values[0] if discount_values else 0.09

        # --- Get Shares Outstanding using yfinance ---
        if ticker:
            try:
                yf_ticker = yf.Ticker(ticker)
                shares_outstanding = yf_ticker.info.get("sharesOutstanding")
                print(f"Shares Outstanding: {shares_outstanding:,}")
            except Exception as e:
                print(f"⚠️ Could not retrieve shares outstanding for {ticker}: {e}")
                shares_outstanding = None
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


        #print("=== 📊 Valuation Results ===")
        #for k, v in result.items():
        #    print(f"{k}: {v}")

        return result 
    
    except Exception as e:
        print(f"⚠️ Error during valuation: {e}")
