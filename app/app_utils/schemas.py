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

from pydantic import BaseModel, Field

# --- Schemas ---

class TriageResponse(BaseModel):
    is_related: bool = Field(description="True if the request is related to business, IT, cloud, AI, data, software, or workflow automation.")
    target_mode: str = Field(description="Set to 'why_change' if the customer's initial description explicitly seeks to build a pitch or justify why they should migrate/change/adopt a new technology. Set to 'discovery' if it's general discovery of their current challenges.")
    polite_decline: str = Field(default="", description="If not related, a polite declining message. Otherwise empty.")
    general_draft: str = Field(default="", description="If related, a draft of a general problem statement summarizing their initial input. Otherwise empty.")
    first_question: str = Field(default="", description="If related and target_mode is 'discovery', a welcoming partner introduction followed by the first discovery question targeting goals or business objectives. Otherwise empty.")

class ProblemStatementDraft(BaseModel):
    context: str = Field(description="Business objectives, current environment, stakeholders. Must be a concise, high-impact 2-3 sentence summary.")
    quantify: str = Field(description="Risks/blockers, dependencies/constraints, success criteria. Must be a concise, high-impact 2-3 sentence summary.")
    scope: str = Field(description="Timelines, technical requirements. Must be a concise, high-impact 2-3 sentence summary.")

class DiscoveryResponse(BaseModel):
    refined_draft: ProblemStatementDraft = Field(description="The updated and refined problem statement structured draft.")
    customer_confidence: str = Field(description="Assessment of customer's confidence in their answer (e.g., High, Medium, Low).")
    customer_accuracy: str = Field(description="Assessment of whether they are describing root causes or just symptoms, cause and impact, or justification (e.g., Symptom-focused, Cause-and-Impact, Justification-focused).")
    finalize: bool = Field(description="True if we have collected sufficient details for Context, Quantify, and Scope to seek customer agreement.")
    next_question: str = Field(default="", description="If finalize is False, the next discovery question. Otherwise empty.")
    next_question_type: str = Field(default="", description="If finalize is False, the Question Type classification from page 5 of presentation (e.g. Background, Decisions, Surface-level Problem, Problem Clarity, Problem Scoping, Current Thinking, Solution Implications). Otherwise empty.")
    next_question_category: str = Field(default="", description="If finalize is False, the Environment Category classification from page 6 of presentation (e.g. Goals, What, Who, When, Where, How Bad, Benefits, Solutions). Otherwise empty.")

class AgreementResponse(BaseModel):
    agreed: bool = Field(description="True if the customer explicitly agreed (e.g., 'yes', 'looks good', 'correct').")
    edits: str = Field(default="", description="If they disagreed and provided edits/comments, summarize them. Otherwise empty.")

class WhyChangePlannerDraft(BaseModel):
    target: str = Field(description="Target organization, role, and industry persona.")
    business_goals: str = Field(description="2-3 specific business goals from the customer's perspective.")
    known_needs: str = Field(description="The recognized needs originally identified by the customer.")
    unconsidered_needs: str = Field(description="Unseen or underappreciated problems, opportunities, or risks introduced by the FDE to disrupt Status Quo Bias.")
    trends_changes: str = Field(description="Real but not obvious market/industry trends creating pressure.")
    previous_approach: str = Field(description="What they attempted previously and why it fails in light of trends.")
    consequences: str = Field(description="Negative impact, risks, and quantified costs of keeping the status quo.")
    new_way: str = Field(description="The Google Cloud solution described as actions (what they can do differently), not just product names.")
    likely_outcomes_project: str = Field(description="Project level impact (e.g., operational efficiency, reducing costs, report generation speed).")
    likely_outcomes_business_unit: str = Field(description="Business Unit level impact (e.g., team productivity, decision speed, resource costs).")
    likely_outcomes_corporate: str = Field(description="Corporate level impact (e.g., revenue growth, CSAT, competitive positioning).")

class WhyChangeLoopResponse(BaseModel):
    refined_planner: WhyChangePlannerDraft = Field(description="The updated Why Change Planner draft.")
    primary_bias: str = Field(description="Identify the primary Status Quo Bias cause from Slide 15: Preference Stability, Cost of Change, Selection Difficulty, Anticipated Regret / Blame.")
    status_quo_analysis: str = Field(description="Analysis of how to disrupt the identified Status Quo Bias.")
    rubric_evaluation: str = Field(description="Self-evaluation against the Why Change Rubric (Business Goals, Trends, Previous Approach, Consequences, New Way, Likely Outcomes). Use values like Poor/Minimal/Sufficient/Above Average/Excellent.")
    finalize: bool = Field(description="True if all sections in the planner meet the Rubric requirements for at 'Sufficient' or better, and we are ready to seek customer agreement.")
    next_question: str = Field(default="", description="If finalize is False, the next targeted question using our skills (Data-Insight-Question, Why Change, Triple Metric) to probe/educate. Otherwise empty.")

class DiscoveryState(BaseModel):
    triage_done: bool = False
    is_related: bool = True
    mode: str = "discovery" # 'discovery' or 'why_change'
    draft_context: str = ""
    draft_quantify: str = ""
    draft_scope: str = ""
    questions_asked: list[str] = Field(default_factory=list)
    answers_received: list[str] = Field(default_factory=list)
    confidence_assessments: list[str] = Field(default_factory=list)
    accuracy_assessments: list[str] = Field(default_factory=list)
    question_types: list[str] = Field(default_factory=list)
    question_categories: list[str] = Field(default_factory=list)
    question_count: int = 0
    next_question: str = ""
    confirmed: bool = False
    is_refining: bool = False # Flag to bypass hard cap finalization trigger during refinements
    agreement_attempts: int = 0 # Count of discovery agreement loops
    
    # Why Change fields
    wc_target: str = ""
    wc_business_goals: str = ""
    wc_known_needs: str = ""
    wc_unconsidered_needs: str = ""
    wc_trends_changes: str = ""
    wc_previous_approach: str = ""
    wc_consequences: str = ""
    wc_new_way: str = ""
    wc_likely_outcomes_project: str = ""
    wc_likely_outcomes_business_unit: str = ""
    wc_likely_outcomes_corporate: str = ""
    wc_primary_biases: list[str] = Field(default_factory=list)
    wc_questions_asked: list[str] = Field(default_factory=list)
    wc_answers_received: list[str] = Field(default_factory=list)
    wc_question_count: int = 0
    wc_next_question: str = ""
    wc_confirmed: bool = False
    wc_agreement_attempts: int = 0 # Count of why change agreement loops
