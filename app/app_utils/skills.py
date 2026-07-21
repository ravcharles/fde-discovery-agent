# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google import genai
from google.genai import types

# --- LLM Client ---
client = genai.Client()
MODEL_NAME = "gemini-2.5-flash"

# --- Reusable FDE Skill Tools (AFC-ready) ---

def data_insight_question(target_customer_or_industry: str, business_strategy: str) -> str:
    """Analyze external factors and industry trends to craft a provocative question highlighting unconsidered needs.
    
    Args:
        target_customer_or_industry: The target customer segment or industry (e.g. Retail, Healthcare, Finance).
        business_strategy: The customer's strategy (e.g. digital transformation, cost optimization).
    """
    prompt = f"""You are a Google Cloud AI Forward Deployed Engineer (FDE) preparing a conversation for {target_customer_or_industry}. 
Analyze external factors outside their control, such as specific industry trends (e.g. rising data volumes, regulatory shifts, or increasing competition), that could impact their business strategy: {business_strategy}.

Identify a problem, opportunity, or risk they may not yet realize exists, supported by compelling data points or insights (e.g., '80% of enterprises struggle with data silos, leading to a 40% reduction in decision-making speed'). 
Craft a provocative question, such as, 'How prepared is your organization to address [specific challenge or risk] without a unified cloud strategy?' 
Tie this question directly to a Google Cloud solution by demonstrating how it addresses the identified gap and creates measurable value. Ensure the insight is provocative yet collaborative, designed to spark curiosity and encourage the customer to envision success with Google Cloud.

Return only the final provocative question and data-backed insight as a string."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                tools=[{"google_search": {}}]
            )
        )
        return response.text
    except Exception as e:
        return f"Error executing tool data_insight_question: {str(e)}. Instructions to recovery: The LLM should bypass live search grounding and proceed with the conversation by leveraging existing local discovery context state to formulate a provocative question."

def why_change_pitch(company_or_persona: str, target_industry_or_market: str) -> str:
    """Start analyzing recent trends and disruptions to craft a compelling Why Change opening statement.
    
    Args:
        company_or_persona: The target company name and user persona.
        target_industry_or_market: The industry or market segment (e.g. SaaS workflow automation, retail logistics).
    """
    prompt = f"""You are a Google Cloud FDE preparing a conversation for {company_or_persona}. 
Start by analyzing recent trends and disruptions in {target_industry_or_market}, such as specific examples (e.g. cloud adoption rates, regulatory changes, or advancements in AI/ML), and identify the primary challenges faced by the target role/persona in achieving their business goals. 

Clearly articulate why current approaches or solutions in the market are falling short (e.g., inefficiencies, risks, or missed opportunities). 
Explain the risks of maintaining the status quo, including quantifiable consequences of inaction. 
Finally, craft a compelling case for adopting a Google Cloud solution as the transformative new way to solve these challenges, detailing measurable outcomes (e.g., cost savings, productivity gains, scalability improvements) and an engaging opening statement that captures their attention and sets the tone for the discussion.

Return only the final Why Change pitch opening statement and disruption details as a string."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                tools=[{"google_search": {}}]
            )
        )
        return response.text
    except Exception as e:
        return f"Error executing tool why_change_pitch: {str(e)}. Instructions to recovery: The LLM should bypass live search grounding and proceed with the conversation by leveraging existing local discovery context state to formulate the why-change pitch."

def triple_metric_scaling(google_cloud_solution: str, customer_name_or_industry: str) -> str:
    """Identify key metrics that illustrate Google Cloud value cascading across Project, Business Unit, and Corporate levels.
    
    Args:
        google_cloud_solution: The proposed Google Cloud solution or service.
        customer_name_or_industry: The name of the customer or their target industry.
    """
    prompt = f"""You are a Google Cloud FDE demonstrating the value of {google_cloud_solution} to {customer_name_or_industry}. 
Identify key metrics that illustrate its impact across the Project, Business Unit, and Corporate levels. For example:
* At the Project level, demonstrate how the solution improves operational efficiency, reduces costs, or accelerates specific workflows (e.g., 'Deploying BigQuery reduced report generation time by 60%').
* At the Business Unit level, link these improvements to broader outcomes, such as enhancing team productivity, achieving faster decision-making, or reducing resource allocation costs (e.g., 'Optimizing workloads with Anthos improved cross-department collaboration by 30%').
* At the Corporate level, quantify the overarching business impact, such as increasing revenue, improving customer satisfaction, or enhancing competitive positioning (e.g., 'Leveraging Google Cloud increased revenue by $X million through faster product launches').

Ensure the metrics are connected across levels to show how success cascades from tactical to strategic goals. Finally, provide actionable recommendations on tracking these metrics (e.g., using Looker dashboards) to ensure continuous alignment with the customer’s long-term objectives.

Return only the final Triple Metric breakdown as a string."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                tools=[{"google_search": {}}]
            )
        )
        return response.text
    except Exception as e:
        return f"Error executing tool triple_metric_scaling: {str(e)}. Instructions to recovery: The LLM should bypass live search grounding and proceed with the conversation by leveraging existing local discovery context state to formulate the triple metric breakdown."
