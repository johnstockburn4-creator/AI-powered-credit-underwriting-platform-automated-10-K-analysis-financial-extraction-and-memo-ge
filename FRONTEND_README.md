# Credit AI Frontend

A beautiful web interface for the Credit AI financial analysis platform.

## Quick Start

### 1. Start the Backend API

```bash
# Make sure you're in the project directory
cd /Users/liamcleary/Optimus2-backend-for-credit-AI

# Activate virtual environment if needed
source venv/bin/activate

# Make sure your .env file has OPENAI_API_KEY set
# Run the FastAPI server
uvicorn main:app --reload --port 8000
```

The API will start at `http://localhost:8000`

### 2. Open the Frontend

Simply open the `frontend.html` file in your web browser:

```bash
open frontend.html
```

Or navigate to the file in your browser:
- File path: `/Users/liamcleary/Optimus2-backend-for-credit-AI/frontend.html`

## Features

### 📊 Comprehensive Analysis Display
- **Credit Memo Tab**: Full underwriting memo with segment-by-segment analysis
- **MD&A Insights Tab**: Business driver commentary with offsetting factors clearly marked
- **Key Metrics Tab**: Visual cards showing revenue, EBITDA, leverage, FCC with YoY changes
- **Detailed Financials Tab**: Complete income statement, balance sheet, and cash flow

### 🎯 Segment Coverage
- Automatically identifies all business segments from MD&A
- Displays segment-specific drivers with dollar impacts
- Highlights offsetting factors in green for easy visibility

### ✅ Data Quality Indicators
- Completeness score displayed prominently
- Validation flags and data gaps clearly shown
- Extraction warnings for missing segments or offsetting factors

### 💼 Borrower Context
- Optional borrower information form
- Covenant tracking (Max Leverage, Min FCC)
- Facility type and use of proceeds

## Using the Frontend

1. **Upload File**: Click or drag-and-drop a PDF or HTML file (10-K, 10-Q)
2. **Optional**: Fill in borrower information and covenant details
3. **Click Analyze**: Processing takes 30-60 seconds
4. **Review Results**:
   - Start with the Credit Memo for executive summary
   - Check MD&A Insights to see all business segments and drivers
   - Review Key Metrics for financial highlights
   - Examine Detailed Financials for complete data

## MD&A Insights Display

The MD&A tab shows:
- **Segments Identified**: Lists all business units found in the document
- **Segment Cards**: One card per business unit with:
  - Revenue drivers with $ impacts
  - Gross margin drivers with $ impacts
  - Volume/price metrics
  - **OFFSETTING factors in GREEN** - easy to spot!

## API Endpoint

The frontend calls: `POST http://localhost:8000/v1/analyze`

Make sure your FastAPI server is running on port 8000.

## Troubleshooting

### CORS Errors
If you see CORS errors, the backend already has CORS middleware configured to allow all origins. Make sure the backend is running.

### Connection Refused
Ensure the FastAPI server is running:
```bash
uvicorn main:app --reload --port 8000
```

### File Upload Fails
- Check file size (max 10MB)
- Ensure file is PDF or HTML format
- Check backend logs for detailed error messages

## Example Analysis Flow

1. Upload Mosaic 10-K
2. System extracts financials for multiple periods
3. MD&A analysis identifies segments:
   - Phosphates
   - Potash
   - Mosaic Fertilizantes
4. For each segment, extracts:
   - Revenue drivers (volume, price, mix)
   - Gross margin drivers (costs, efficiencies)
   - **Offsetting factors** (marked clearly)
5. Generates comprehensive credit memo with all insights

## Next Steps

To enhance the frontend:
- Add Excel export button (backend already has `/v1/export/excel` endpoint)
- Add historical run tracking
- Add covenant compliance visual indicators
- Add downloadable PDF reports
