# Real-Time Market Sentiment Analyzer

This project implements a pipeline to analyze market sentiment for a given company using Azure OpenAI and LangChain.

## Features

- Accepts a company name as input.
- Extracts or generates its stock code using Yahoo Finance tools (via LangChain).
- Fetches the latest news for the company using Yahoo Finance.
- Sends the news to Azure OpenAI (GPT-4o-mini or similar) for sentiment analysis and entity extraction.
- Outputs a structured JSON object as required by assignment guidelines.
- (Optional) Uses Langfuse for prompt tracing, debugging, and monitoring.
- Includes clear code structure for extension, testing, and productionization.


## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install yfinance requests beautifulsoup4 langchain-openai langfuse pydantic
   ```

2. **Set Environment Variables**:
   ```bash
   export AZURE_OPENAI_ENDPOINT="your_api_endpoint"
   export AZURE_OPENAI_API_KEY="your_api_key"
   export LANGFUSE_PUBLIC_KEY="your_langfuse_public_key"
   export LANGFUSE_SECRET_KEY="your_langfuse_secret_key"
   ```

3. **Run the Script**:
   ```bash
   python assignment1/rimanshu_assignment1.py "Microsoft"
   ```

## Azure and Langfuse API Setup Steps

- **Azure OpenAI API**: Set up an Azure OpenAI resource in the Azure portal.Deploy a supported model (e.g., gpt-4o-mini).Obtain your API key and endpoint URL. Ensure you have a valid API key and endpoint URL. These should be set in your environment variables.
- **Langfuse API**: Register on langfuse.com or your organization's instance.Get your public and secret keys.Set up the integration in the script or via environment variables.Set up your Langfuse account and obtain the public and secret keys. These should also be set in your environment variables.

## Sample Output JSON

The script outputs a structured JSON for sentiment analysis. Here's an example for "Microsoft":

```json
{
  "company_name": "Microsoft",
  "stock_code": "MSFT",
  "newsdesc": "Microsoft has launched a new AI-powered toolkit for businesses, enhancing productivity functionalities across its suite of applications.",
  "sentiment": "Positive",
  "people_names": ["Satya Nadella"],
  "places_names": ["Redmond"],
  "other_companies_referred": ["Google", "Amazon"],
  "related_industries": ["Technology", "Software", "AI"],
  "market_implications": "The introduction of AI solutions is likely to increase Microsoft's competitive edge in the productivity software market and could attract more enterprise customers.",
  "confidence_score": 0.87
}
```