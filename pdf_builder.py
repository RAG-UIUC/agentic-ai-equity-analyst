from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter

doctype = letter
doc = canvas.Canvas(filename="financial_report.pdf", pagesize=doctype)

# mildly annoying convention where (0,0) is at the bottom left
w, h = doctype 
font = "Helvetica"
font_sz = 9

# testing string 
content = """
The provided data consists of key financial information from Apple Inc.'s 2022 10-K filing, specifically focusing on revenue from various product and service categories over the fiscal years 2020, 2021, and 2022. Here's a summary and analysis of the information:

### Revenue Breakdown (in millions USD)
1. **iPhone:**
   - 2020: $137,781
   - 2021: $191,973
   - 2022: $205,489

2. **Mac:**
   - 2020: $28,622
   - 2021: $35,190
   - 2022: $40,177

3. **iPad:**
   - 2020: $23,724
   - 2021: $31,862
   - 2022: $29,292

4. **Wearables, Home, and Accessories:**
   - 2020: $30,620
   - 2021: $38,367
   - 2022: $41,241

5. **Services:**
   - 2020: $53,768
   - 2021: $68,425
   - 2022: $78,129

6. **Total Net Sales:**
   - 2020: $274,515
   - 2021: $365,817
   - 2022: $394,328

### Key Observations
- **Growth Across Segments:** There was a uniformly strong growth in net sales across all product categories from 2020 to 2022. The iPhone category, being the most significant revenue driver, saw an increase of approximately 11.3% in 2022 from 2021.
- **Notable Performance of Services:** The Services category showed significant growth, rising by approximately 14.2% from 2021 to 2022. This indicates Apple's strategic focus on leveraging its ecosystem beyond hardware sales.
- **Wearables, Home, and Accessories:** This category continued to grow, reflecting successful product offerings like the Apple Watch and AirPods. The category grew by around 7.5% year-over-year from 2021 to 2022.
- **Mac and iPad Trends:** While Mac sales increased steadily, iPad sales saw a minor decrease in 2022 compared to 2021, suggesting potential market saturation or competition effects.

### Geographic Insights
- Revenue proportions were generally consistent across segments, except for Greater China, where iPhone revenue constituted a higher proportion in recent years.

### Deferred Revenue
- The deferred revenue as of September 2022 was $12.4 billion, up from $11.9 billion in the previous year, reflecting growth in future business commitments and ongoing subscription services.

### Analysis
Apple's fiscal year 2022 demonstrates robust performance amid global economic challenges. The company continues to expand its services sector, capturing more consumer engagement and boosting recurring revenue streams. The sustained demand for iPhones and related technological products underscores Apple's resilience and capacity to innovate. Enhanced performance in the Greater China region also portrays Apple's strategic penetration into key international markets. Overall, Apple's diverse product portfolio and services business are essential for sustaining its growth momentum."""

txt = doc.beginText(50, h-50)
txt.setFont(font, font_sz)


prev = 0

for i in range(len(content)):
    if content[i] == '\n' or doc.stringWidth(content[prev:i], fontName=font, fontSize=font_sz) > w-100:
        txt.textLine(content[prev:i])
        prev = i 


txt.textLine(content[prev:])

doc.drawText(txt)
doc.save()