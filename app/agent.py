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
from google import genai
from google.genai import types
from google.adk.workflow import Workflow, START, node
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from app.app_utils.skills import data_insight_question, why_change_pitch, triple_metric_scaling

from app.app_utils.schemas import (
    TriageResponse,
    ProblemStatementDraft,
    DiscoveryResponse,
    AgreementResponse,
    WhyChangePlannerDraft,
    WhyChangeLoopResponse,
    DiscoveryState
)

# --- LLM Client ---
client = genai.Client()
MODEL_NAME = "gemini-2.5-flash"


# --- Helper ---
def to_event(text: str) -> Event:
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return Event(
        output=content,
        content=content
    )

# --- Nodes ---

@node
async def triage_and_draft(ctx: Context, node_input: str):
    state = ctx.state
    
    # Replay optimization: skip LLM call if triage has already executed and set state
    if state.get("triage_done"):
        ctx.route = "related"
        return None
    
    prompt = f"""You are a Google Cloud AI Forward Deployed Engineer (FDE).
A customer is starting a conversation with this initial message:
"{node_input}"

Triage and prepare for this request by analyzing it through three key lenses:
1. **Business Initiatives**: What are their high-level business goals or priority outcomes?
2. **Financial Metrics**: Are there any mentions of costs, productivity loss, or ROI expectations?
3. **External Factors**: What external/internal constraints are blocking them (e.g. data governance, compliance, custom architectures)?

Is the request related to business, IT, cloud, AI, data engineering, software, or workflow automation?
Classify the target mode:
- Set target_mode = 'why_change' ONLY if the user's initial message explicitly requests a business pitch, pitch deck content, or a business case to justify adopting Google Cloud (e.g., 'Draft a pitch for my leadership team to adopt GCP', 'Write a presentation on why GCP is better').
- For all other messages—including those describing technical challenges, workarounds, blockers, or prototype requests—you MUST set target_mode = 'discovery' to define the Problem Statement first.

Generate your response conforming to the required schema.

Adhere to these style instructions for the first question (only if target_mode is 'discovery'):
- Adopt a warm, conversational, and collaborative partner persona, like a consultant over coffee.
- Begin the message by introducing yourself as their AI Forward Deployed Engineer discovery partner. Welcomingly invite them to collaborate on mapping out their project's objectives and aligning on a clear, refined problem statement they can agree on.
- Following the introduction, ask exactly one targeted, natural first question to start exploring their main goals or business objectives. Keep it brief and friendly."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TriageResponse,
            temperature=0.1
        )
    )
    
    data = TriageResponse.model_validate_json(response.text)
    
    if not data.is_related:
        state["triage_done"] = True
        ctx.route = "unrelated"
        return to_event(data.polite_decline)
    
    state["is_related"] = True
    state["triage_done"] = True
    
    # We always start in discovery mode to align on the problem statement first
    ctx.route = "related"
    state["mode"] = "discovery"
    state["draft_context"] = data.general_draft
    state["draft_quantify"] = "TBD"
    state["draft_scope"] = "TBD"
    state["next_question"] = data.first_question
    state["question_count"] = 1
    state["questions_asked"] = [data.first_question]
    state["question_types"] = ["Background"]
    state["question_categories"] = ["Goals"]
        
    return None

@node
async def decline_node(ctx: Context, node_input: types.Content | str):
    ctx.route = "done"
    if isinstance(node_input, types.Content):
        return Event(output=node_input, content=node_input)
    content = types.Content(role="model", parts=[types.Part.from_text(text=str(node_input))])
    return Event(output=content, content=content)

@node
async def ask_question_node(ctx: Context):
    state = ctx.state
    question = state.get("next_question", "")
    count = state.get("question_count", 0)
    
    yield to_event(question)
    yield RequestInput(
        interrupt_id=f"question_{count}",
        message=question
    )

def _find_user_response(ctx: Context, interrupt_id: str) -> str | None:
    if not ctx.session or not ctx.session.events:
        return None
    for event in reversed(ctx.session.events):
        if event.author == "user" and event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_response and part.function_response.id == interrupt_id:
                    resp = part.function_response.response
                    if isinstance(resp, dict):
                        return resp.get("result", "")
                    return str(resp)
    return None

def _get_user_input(ctx: Context, interrupt_id: str, node_input: types.Content | str) -> str:
    # 1. Search session events history for the function response to the active interrupt
    user_msg = _find_user_response(ctx, interrupt_id) or ""
    
    # 2. Fallback to ctx.resume_inputs
    if not user_msg and ctx.resume_inputs and interrupt_id in ctx.resume_inputs:
        resp = ctx.resume_inputs[interrupt_id]
        if isinstance(resp, dict):
            user_msg = resp.get("result", "")
        else:
            user_msg = str(resp)
            
    # 3. Fallback to node_input (e.g. for unit tests)
    if not user_msg:
        if isinstance(node_input, types.Content):
            text_parts = []
            for part in node_input.parts:
                if part.text:
                    text_parts.append(part.text)
            user_msg = "".join(text_parts)
        else:
            user_msg = str(node_input)
            
    return user_msg


@node
async def discovery_loop(ctx: Context, node_input: types.Content | str):
    state = ctx.state
    count = state.get("question_count", 0)
    
    interrupt_id = f"question_{count}"
    user_msg = _get_user_input(ctx, interrupt_id, node_input)

        
    answers = state.get("answers_received", [])
    answers.append(user_msg)
    state["answers_received"] = answers
    
    questions = state.get("questions_asked", [])
    
    # Prune history to last 3 turns to optimize token usage
    history_str = ""
    recent_questions = questions[-3:]
    recent_answers = answers[-3:]
    for q, a in zip(recent_questions, recent_answers):
        history_str += f"Q: {q}\nA: {a}\n\n"
        
    prompt = f"""You are a Google Cloud AI Forward Deployed Engineer (FDE).
We are conducting a discovery call with a customer.
Here is the history of questions and answers:
{history_str}

Current draft:
- Context: {state.get('draft_context')}
- Quantify: {state.get('draft_quantify')}
- Scope: {state.get('draft_scope')}

Analyze the latest customer response: "{user_msg}"

Your first task is to perform an assessment of the customer's response based on our discovery methodology:
1. **Customer Confidence**: Evaluate their certainty and alignment (e.g., High, Medium, Low).
2. **Customer Accuracy**: Assess where they fall on the accuracy focus:
   - `Symptom-focused`: Discussing surface symptoms (slower engineering, delays, manual steps) rather than core issues.
   - `Cause-and-Impact`: Explaining the root cause (why it happens) and operational impact (governance blockers, architecture mismatches).
   - `Justification-focused`: Explaining the business case, ROI, why they think the solution works, or proving value.

Formulate the next targeted discovery question based on this assessment. Use the following specialized discovery frameworks to guide your focus:

A. **Probing Strategies (Slide 5)**:
   - If they are **Symptom-focused** or exhibit **Low Confidence**, probe on **Problem vs. Symptoms** (find the connection between symptom and problem, highlight "silent killers" like industry trends) or **Cause & Impact** (ask what gives them confidence they identified the right source of pain, or check "pain radiation" - how the issue affects other business parts indirectly).
   - If they lack business or roadmap justification, probe on **Justification** (solution justification, why they looked at those options, ROI/value metrics, implementation challenges, or timeline).

B. **Reference Scoping Questions (Slide 4)**:
   Depending on which section of the problem statement (Context, Quantify, Scope) needs detail, refer to these targeted prompt templates:
   - **Goals**: What business outcome is intended? Why a priority now? How is success measured?
   - **What**: What does the current workflow look like? What operational/technical challenges exist? What systems/integrations are involved?
   - **Who**: Who owns the environment? Who is most impacted? Who evaluates/approves?
   - **When**: Is there an event/deadline driving this? What is the expected timeline?
   - **Where**: Where are the challenges occurring (applications, regions, workloads)? Where is the friction?
   - **How Bad**: What risks are created? What happens if nothing changes? What blockers/constraints exist?
   - **Benefits**: What would improve if resolved? What metrics change? How is success validated?
   - **Solutions**: What technical/architecture/compliance/security requirements are non-negotiable? Are there dependencies?

Refine the problem statement draft:
- Context: (business objectives, current environment, stakeholders, assumptions)
- Quantify: (risks/blockers, dependencies/constraints, success criteria)
- Scope: (timelines, technical requirements)

Each section of the draft must be summarized in a few clear, high-impact sentences. Avoid unnecessary filler or boilerplate text.

IMPORTANT: Prioritize asking about any section (Context, Quantify, Scope) that is currently 'TBD' or empty in the draft. Do not ask multiple consecutive questions about the same category (e.g. asking 3 questions in a row about consequences or risks). If a section like Scope is empty or 'TBD', your next question MUST target that missing section (e.g. asking about timeline, technical requirements, or system dependencies).

Determine if we have enough information to get final alignment. We want to be highly thorough in our discovery, exploring details for Context, Quantify, and Scope. Set finalize = True if:
- All sections (Context, Quantify, Scope) have thorough, concrete details (no placeholders or 'TBD').
- Crucially, set finalize = True early ONLY if the customer explicitly requests to stop (e.g., 'stop', 'I'm done answering', 'no more questions') or repeatedly refuses to answer business context questions. Otherwise, be thorough and continue probing to gather complete details.
- If we have reached {count} questions.

If finalize is False, formulate the next targeted scoping/solution question.
Generate response conforming to the required schema.

Adhere to these styling instructions for formulation of the next question:
- Keep the tone warm, empathetic, collaborative, and highly conversational.
- Build directly upon the customer's previous response. Use active listening cues (e.g., acknowledging what they just shared) before introducing the next aspect.
- Do not repeat introductory welcomes, greetings, or launch phrases (e.g., avoid "To kick things off", "To get started", "To start with") since the conversation is already ongoing. Ask the question directly.
- Ask exactly one targeted question at a time. Do not compile checklists or multi-part questions (e.g., do not ask 'For timeline, do you have target date? Also, are there dependencies?'). Keep it simple and human-like."""


    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DiscoveryResponse,
            temperature=0.2,
            tools=[data_insight_question, why_change_pitch, triple_metric_scaling]
        )
    )
    
    try:
        data = DiscoveryResponse.model_validate_json(response.text)
    except Exception:
        fallback_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiscoveryResponse,
                temperature=0.2
            )
        )
        data = DiscoveryResponse.model_validate_json(fallback_response.text)
    
    # Save assessments
    confidences = state.get("confidence_assessments", [])
    confidences.append(data.customer_confidence)
    state["confidence_assessments"] = confidences
    
    accuracies = state.get("accuracy_assessments", [])
    accuracies.append(data.customer_accuracy)
    state["accuracy_assessments"] = accuracies
    
    # Save refined draft to state
    state["draft_context"] = data.refined_draft.context
    state["draft_quantify"] = data.refined_draft.quantify
    state["draft_scope"] = data.refined_draft.scope
    
    # Clear refinement override once the user message has been processed
    is_refining = state.get("is_refining", False)
    state["is_refining"] = False
    
    if data.finalize or (count >= 15 and not is_refining):
        ctx.route = "finalize"
        return None
    
    # Continue discovery loop
    state["next_question"] = data.next_question
    state["question_count"] = count + 1
    
    questions.append(data.next_question)
    state["questions_asked"] = questions
    
    q_types = state.get("question_types", [])
    q_types.append(data.next_question_type)
    state["question_types"] = q_types
    
    q_categories = state.get("question_categories", [])
    q_categories.append(data.next_question_category)
    state["question_categories"] = q_categories
    
    ctx.route = "ask_question"
    return None

@node
async def get_agreement(ctx: Context, node_input: types.Content | str | None = None):
    state = ctx.state
    draft_summary = f"""[Context]
{state.get('draft_context')}

[Quantify]
{state.get('draft_quantify')}

[Scope]
{state.get('draft_scope')}"""

    message = f"\nDo you agree with this refined problem statement? (Yes/No or provide feedback):\n\n{draft_summary}"
    yield to_event(message)
    yield RequestInput(
        interrupt_id="agreement",
        message=message
    )

@node
async def process_agreement(ctx: Context, node_input: types.Content | str):
    state = ctx.state
    
    user_msg = _get_user_input(ctx, "agreement", node_input)

        
    prompt = f"""Analyze the customer's response to the problem statement agreement request.
Customer response: "{user_msg}"

Conform your output to the AgreementResponse schema."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AgreementResponse,
            temperature=0.1
        )
    )
    
    data = AgreementResponse.model_validate_json(response.text)
    
    if data.agreed:
        ctx.route = "agreed"
        state["confirmed"] = True
        return None
    
    # Handle edits/feedback: route back to discovery loop
    attempts = state.get("agreement_attempts", 0) + 1
    state["agreement_attempts"] = attempts
    
    if attempts >= 3:
        ctx.route = "agreed"
        state["confirmed"] = True
        force_msg = "We have completed several refinement rounds. Let's lock in this version as our working problem statement and transition to the Why Change phase."
        return to_event(force_msg)
        
    state["is_refining"] = True
    count = state.get("question_count", 0) + 1
    state["question_count"] = count
    state["next_question"] = f"Let's refine the draft based on your feedback: '{data.edits}'. Can you provide more details about this change?"
    
    questions = state.get("questions_asked", [])
    questions.append(state["next_question"])
    state["questions_asked"] = questions
    
    ctx.route = "disagreed"
    disagree_msg = "Understood. Let's adjust the draft based on your feedback."
    return to_event(disagree_msg)

@node
async def why_change_start_node(ctx: Context):
    state = ctx.state
    state["mode"] = "why_change"
    
    # Replay optimization: skip initial why change planner draft LLM call if it has already run
    if state.get("wc_target") and state.get("wc_target") != "TBD":
        return None
        
    state["wc_question_count"] = 1
     # Seed the planner from discovery state (which is guaranteed to be finalized and confirmed)
    business_goals = state.get("draft_context", "")
    consequences = state.get("draft_quantify", "")
    new_way = state.get("draft_scope", "")
    
    # Compile Discovery Q&A history
    discovery_qs = state.get("questions_asked", [])
    discovery_as = state.get("answers_received", [])
    discovery_history = ""
    for q, a in zip(discovery_qs, discovery_as):
        discovery_history += f"Q: {q}\nA: {a}\n\n"
        
    prompt = f"""You are a Google Cloud AI Forward Deployed Engineer (FDE).
We are transitioning to the **Why Change** phase.
Our goal is to construct a compelling "Why Change" pitch planner for the customer to defeat status quo bias and highlight unconsidered needs.

Here is the preceding Discovery conversation history:
{discovery_history}

Here is the source context we have collected:
- Context / Initial Goals: {business_goals}
- Discovery Quantifications / Consequences: {consequences}
- Discovery Scoping / Solutions (New Way): {new_way}

Generate the initial draft of the Why Change Planner:
1. **Target**: Identify the target persona and industry based on the context.
2. **Business Goal**: Populate using the context goals.
3. **Known Needs**: State the customer's known/recognized needs.
4. **Unconsidered Needs**: Identify what the customer might have missed or underappreciated.
5. **Trends & Changes**: Formulate initial industry trends.
6. **Previous Approach**: Identify what they attempted or might have attempted.
7. **Consequences**: Populate using the risks/costs context.
8. **New Way**: Describe how Google Cloud addresses these gaps.
9. **Likely Outcomes**: Formulate initial expected outcomes split into Project, Business Unit, and Corporate metrics.

Also, formulate the first targeted question to ask the customer about their previous approach or current setup (status quo) to understand why they want to change. Do not introduce yourself or use welcome greetings (since the conversation is already ongoing and we just aligned on the problem statement). Instead, transition seamlessly using a brief, natural segue (e.g., acknowledging the alignment we just reached and inviting them to explore the underlying trends or setup) before asking the question directly. Keep the tone warm and collaborative.
Conform to the WhyChangeLoopResponse schema."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=WhyChangeLoopResponse,
            temperature=0.2,
            tools=[data_insight_question, why_change_pitch, triple_metric_scaling] # Enable tools
        )
    )
    
    try:
        data = WhyChangeLoopResponse.model_validate_json(response.text)
    except Exception:
        fallback_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WhyChangeLoopResponse,
                temperature=0.2
            )
        )
        data = WhyChangeLoopResponse.model_validate_json(fallback_response.text)
    
    # Save initial planner
    state["wc_business_goals"] = data.refined_planner.business_goals
    state["wc_known_needs"] = data.refined_planner.known_needs
    state["wc_unconsidered_needs"] = data.refined_planner.unconsidered_needs
    state["wc_trends_changes"] = data.refined_planner.trends_changes
    state["wc_previous_approach"] = data.refined_planner.previous_approach
    state["wc_consequences"] = data.refined_planner.consequences
    state["wc_new_way"] = data.refined_planner.new_way
    state["wc_likely_outcomes_project"] = data.refined_planner.likely_outcomes_project
    state["wc_likely_outcomes_business_unit"] = data.refined_planner.likely_outcomes_business_unit
    state["wc_likely_outcomes_corporate"] = data.refined_planner.likely_outcomes_corporate
    state["wc_target"] = data.refined_planner.target
    
    state["wc_next_question"] = data.next_question
    
    ctx.route = "ask"
    return None

@node
async def why_change_ask_node(ctx: Context):
    state = ctx.state
    question = state.get("wc_next_question", "")
    count = state.get("wc_question_count", 1)
    
    yield to_event(question)
    yield RequestInput(
        interrupt_id=f"wc_question_{count}",
        message=question
    )

@node
async def why_change_loop(ctx: Context, node_input: types.Content | str):
    state = ctx.state
    count = state.get("wc_question_count", 1)
    
    interrupt_id = f"wc_question_{count}"
    user_msg = _get_user_input(ctx, interrupt_id, node_input)
    
    answers = state.get("wc_answers_received", [])
    answers.append(user_msg)
    state["wc_answers_received"] = answers
    
    questions = state.get("wc_questions_asked", [])
    questions.append(state.get("wc_next_question", ""))
    state["wc_questions_asked"] = questions
    
    # Prune history to last 3 turns to optimize token usage
    history_str = ""
    recent_questions = questions[-3:]
    recent_answers = answers[-3:]
    for q, a in zip(recent_questions, recent_answers):
        history_str += f"Q: {q}\nA: {a}\n\n"
        
    # Compile Discovery Q&A history
    discovery_qs = state.get("questions_asked", [])
    discovery_as = state.get("answers_received", [])
    discovery_history = ""
    for q, a in zip(discovery_qs, discovery_as):
        discovery_history += f"Q: {q}\nA: {a}\n\n"

    prompt = f"""You are a Google Cloud AI Forward Deployed Engineer (FDE).
We are constructing a "Why Change" pitch planner for the customer to defeat status quo bias and highlight unconsidered needs.

Here is the preceding Discovery conversation history:
{discovery_history}

Here is the Why Change Planner Rubric:
- **Business Goals**: Starts story in customer's world, 2-3 specific business goals, relates to stakeholder's perspective.
- **Known Needs vs. Unconsidered Needs**: Explicitly contrasts their recognized/known needs with the unconsidered/unseen problems/opportunities we introduced.
- **Trends & Changes**: Real but not obvious trends/changes creating business pressure, relates to stakeholders' interests, includes sufficient detail to establish credibility.
- **Previous Approaches**: Describes previous approach, explains why it no longer works in light of trends/changes, shows how it no longer supports goals.
- **Consequences**: Uses descriptive language to convey the negative impact of continuing the previous approach (quantified risks/consequences of status quo).
- **New Way**: Creates an "Aha" moment by contrasting the previous approach with the new way, describes what the customer will be able to do differently using action, DO-level statements, presents the new way without stating solution capabilities or features.
- **Likely Outcomes**: Cascades values across three tiers:
  1. Project level: operational efficiency, speed, error reduction.
  2. Business Unit level: team productivity, collaboration, cost.
  3. Corporate level: revenue, CSAT, competitive positioning.

Here is the history of the Why Change discussion:
{history_str}

Current Why Change Planner Draft:
- Target: {state.get('wc_target')}
- Business Goals: {state.get('wc_business_goals')}
- Known Needs: {state.get('wc_known_needs')}
- Unconsidered Needs: {state.get('wc_unconsidered_needs')}
- Trends & Changes: {state.get('wc_trends_changes')}
- Previous Approach: {state.get('wc_previous_approach')}
- Consequences: {state.get('wc_consequences')}
- New Way: {state.get('wc_new_way')}
- Likely Outcomes (Project): {state.get('wc_likely_outcomes_project')}
- Likely Outcomes (Business Unit): {state.get('wc_likely_outcomes_business_unit')}
- Likely Outcomes (Corporate): {state.get('wc_likely_outcomes_corporate')}

Analyze the customer's latest response: "{user_msg}"

Refine the Why Change Planner Draft based on this new information. Ensure each section is clear, concise, and meets the 'Sufficient' or 'Excellent' standard in the Rubric.
Perform self-evaluation of the refined draft against the Rubric and identify the primary cause of Status Quo Bias active in the response: Preference Stability, Cost of Change, Selection Difficulty, Anticipated Regret / Blame.

Determine if the planner is complete. We want to be highly thorough, ensuring all sections in the planner meet the 'Sufficient' or 'Excellent' Rubric standard before finalizing. Set finalize = True if:
- All sections (Business Goals, Needs, Trends, Consequences, New Way, Outcomes) have thorough, concrete details meeting the Rubric.
- Crucially, set finalize = True early ONLY if the customer explicitly requests to stop (e.g., 'stop', 'I'm done answering', 'no more questions') or repeatedly refuses to answer business context questions. Otherwise, be thorough and continue probing using your skills.
If finalize is False, formulate the next targeted question using our skills:
- **Data-Insight-Question**: Analyze external trends, identify unconsidered needs/risks supported by data points, and craft a provocative question linked to Google Cloud.
- **Why Change**: Target recent disruptions, analyze why status quo is risky, and explain shortfall of current approaches.
- **Triple Metric**: Cascade metrics to Project, Business Unit, and Corporate levels.

You have access to Google Search grounding to lookup real industry trends, statistics, or Google Cloud solutions.

Adhere to these styling instructions:
- Keep the tone warm, collaborative, and highly conversational.
- Ask exactly one targeted question at a time. Do not compile checklists.
- Do not use welcome greetings or launch phrases. Ask directly.

Generate response conforming to the WhyChangeLoopResponse schema."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=WhyChangeLoopResponse,
            temperature=0.2,
            tools=[data_insight_question, why_change_pitch, triple_metric_scaling] # Enable tools
        )
    )
    
    try:
        data = WhyChangeLoopResponse.model_validate_json(response.text)
    except Exception:
        fallback_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WhyChangeLoopResponse,
                temperature=0.2
            )
        )
        data = WhyChangeLoopResponse.model_validate_json(fallback_response.text)
    
    # Update state
    state["wc_business_goals"] = data.refined_planner.business_goals
    state["wc_known_needs"] = data.refined_planner.known_needs
    state["wc_unconsidered_needs"] = data.refined_planner.unconsidered_needs
    state["wc_trends_changes"] = data.refined_planner.trends_changes
    state["wc_previous_approach"] = data.refined_planner.previous_approach
    state["wc_consequences"] = data.refined_planner.consequences
    state["wc_new_way"] = data.refined_planner.new_way
    state["wc_likely_outcomes_project"] = data.refined_planner.likely_outcomes_project
    state["wc_likely_outcomes_business_unit"] = data.refined_planner.likely_outcomes_business_unit
    state["wc_likely_outcomes_corporate"] = data.refined_planner.likely_outcomes_corporate
    state["wc_target"] = data.refined_planner.target
    
    biases = state.get("wc_primary_biases", [])
    biases.append(data.primary_bias)
    state["wc_primary_biases"] = biases
    
    is_refining = state.get("is_refining", False)
    state["is_refining"] = False
    
    if data.finalize or (count >= 12 and not is_refining):
        ctx.route = "finalize"
        return None
        
    state["wc_next_question"] = data.next_question
    state["wc_question_count"] = count + 1
    ctx.route = "ask"
    return None

@node
async def why_change_agreement(ctx: Context, node_input: types.Content | str | None = None):
    state = ctx.state
    planner_summary = f"""**WHY CHANGE PLANNER**
* **Target**: {state.get('wc_target')}
* **Business Goals**: {state.get('wc_business_goals')}
* **Known Needs**: {state.get('wc_known_needs')}
* **Unconsidered Needs**: {state.get('wc_unconsidered_needs')}
* **Trends & Changes**: {state.get('wc_trends_changes')}
* **Previous Approach**: {state.get('wc_previous_approach')}
* **Consequences**: {state.get('wc_consequences')}
* **New Way**: {state.get('wc_new_way')}
* **Likely Outcomes (Project)**: {state.get('wc_likely_outcomes_project')}
* **Likely Outcomes (Business Unit)**: {state.get('wc_likely_outcomes_business_unit')}
* **Likely Outcomes (Corporate)**: {state.get('wc_likely_outcomes_corporate')}"""

    message = f"\nDo you agree with this refined Why Change Planner draft? (Yes/No or provide feedback):\n\n{planner_summary}"
    yield to_event(message)
    yield RequestInput(
        interrupt_id="wc_agreement",
        message=message
    )

@node
async def process_wc_agreement(ctx: Context, node_input: types.Content | str):
    state = ctx.state
    user_msg = _get_user_input(ctx, "wc_agreement", node_input)
    
    prompt = f"""Analyze the customer's response to the Why Change Planner agreement request.
Customer response: "{user_msg}"

Conform your output to the AgreementResponse schema."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AgreementResponse,
            temperature=0.1
        )
    )
    
    data = AgreementResponse.model_validate_json(response.text)
    
    if data.agreed:
        ctx.route = "agreed"
        state["wc_confirmed"] = True
        return None
    
    # Handle feedback/edits: route back to why change loop
    attempts = state.get("wc_agreement_attempts", 0) + 1
    state["wc_agreement_attempts"] = attempts
    
    if attempts >= 3:
        ctx.route = "agreed"
        state["wc_confirmed"] = True
        force_msg = "We have completed several refinement rounds. Let's lock in this version as our finalized Why Change Planner and prepare the reasoning summary."
        return to_event(force_msg)
        
    state["is_refining"] = True
    count = state.get("wc_question_count", 1) + 1
    state["wc_question_count"] = count
    state["wc_next_question"] = f"Let's refine the Why Change Planner based on your feedback: '{data.edits}'. Can you tell me more about this adjustment?"
    
    ctx.route = "disagreed"
    return to_event("Understood. Let's adjust the Why Change Planner based on your feedback.")

@node
async def reasoning_summary_node(ctx: Context):
    state = ctx.state
    
    questions = state.get("questions_asked", [])
    answers = state.get("answers_received", [])
    confidences = state.get("confidence_assessments", [])
    accuracies = state.get("accuracy_assessments", [])
    q_types = state.get("question_types", [])
    q_categories = state.get("question_categories", [])
    
    history_str = ""
    for i, (q, a) in enumerate(zip(questions, answers)):
        conf = confidences[i] if i < len(confidences) else "N/A"
        acc = accuracies[i] if i < len(accuracies) else "N/A"
        q_type = q_types[i] if i < len(q_types) else "N/A"
        q_cat = q_categories[i] if i < len(q_categories) else "N/A"
        history_str += f"Q: {q}\nA: {a}\n[Assessment] Confidence: {conf}, Focus: {acc}\n[Classification] Type: {q_type}, Category: {q_cat}\n\n"
        
    wc_planner_str = ""
    if state.get("wc_confirmed"):
        wc_planner_str = f"""

Final Why Change Planner aligned:
- Target: {state.get('wc_target')}
- Business Goals: {state.get('wc_business_goals')}
- Known Needs: {state.get('wc_known_needs')}
- Unconsidered Needs: {state.get('wc_unconsidered_needs')}
- Trends & Changes: {state.get('wc_trends_changes')}
- Previous Approach: {state.get('wc_previous_approach')}
- Consequences: {state.get('wc_consequences')}
- New Way: {state.get('wc_new_way')}
- Likely Outcomes (Project): {state.get('wc_likely_outcomes_project')}
- Likely Outcomes (Business Unit): {state.get('wc_likely_outcomes_business_unit')}
- Likely Outcomes (Corporate): {state.get('wc_likely_outcomes_corporate')}"""

    prompt = f"""You are a Google Cloud AI Forward Deployed Engineer (FDE).
We have successfully aligned on a problem statement with the customer using the **REALIGN** (Research, Evaluate, Ask, Listen, Inspect, Get Agreement, Next Steps) framework for Problem-Minded Discovery.

Here is the history of the discovery session questions, answers, assessments, and classifications:
{history_str}

Final problem statement aligned:
- Context: {state.get('draft_context')}
- Quantify: {state.get('draft_quantify')}
- Scope: {state.get('draft_scope')}{wc_planner_str}

Generate a structured "Discovery Reasoning Summary" for the customer.
Your output must be organized around the **ALIGN** discovery stages:
- **Ask (A) & Listen (L)**: Present the question and answer exchanges.
- **Inspect (I)**: For each exchange, analyze the customer's Confidence and Accuracy focus, list the Question Type & Environment Category classifications (which are provided in the history), state the concrete Information Gained, and describe the Impact on the Aligned Problem Statement.
- **Get Agreement (G)**: 
  1. Display the final aligned Problem Statement draft (covering Context, Quantify, Scope) that was agreed upon.
  2. Classify where the customer's problem falls on the **Problem Type Continuum** based on the discovery journey:
     - `Unseen`: The customer is not aware of the problem at all or its business impact.
     - `Undefined`: The customer is aware of pain but cannot confidently describe its cause, extent, or outcomes.
     - `Unsure`: The customer is aware of pain and can confidently describe the problem and its root causes. They know multiple solution options exist but are unsure which to choose.
     - `Unresolved`: The customer knows the problem, root cause, and has a preferred solution, and is now trying to select the provider of that solution (e.g. Google Cloud).
  3. Provide a brief justification explaining why the customer fits this classification based on their confidence and accuracy levels.
- **Why Change Pitch**: If the Why Change planner was completed (indicated by Why Change state parameters being active), construct a compelling 1-2 paragraph storytelling pitch following the Google Cloud 'Why Change Story Circle' format:
  * Stated Goal: Introduce the customer's business objective (derived from Business Goals).
  * Friction/Challenge: Contrast it with the recognized and unconsidered challenges they face (derived from Known/Unconsidered Needs).
  * Previous Approach: Explain the limitations of their status quo setup (derived from Previous Approach).
  * Consequences: Highlight the risks of continuing the status quo (derived from Consequences).
  * New Way: Introduce the transformative new opportunity enabled by Google Cloud (derived from New Way).
  * Resolution: Resolve with the strategic cascaded metrics (derived from Likely Outcomes).
  Clearly indicate which sentences or sections of the planner were used to build each part of this narrative pitch.

Format the output cleanly using structured subheadings and markdown tables to make it look premium, structured, and professional."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    
    summary_suffix = ""
    if state.get("wc_confirmed"):
        biases_identified = ", ".join(set(state.get("wc_primary_biases", []))) or "None"
        summary_suffix = f"""\n\n### Aligned Why Change Planner:
* **Target**: {state.get('wc_target')}
* **Business Goals**: {state.get('wc_business_goals')}
* **Known Needs**: {state.get('wc_known_needs')}
* **Unconsidered Needs**: {state.get('wc_unconsidered_needs')}
* **Trends & Changes**: {state.get('wc_trends_changes')}
* **Previous Approach**: {state.get('wc_previous_approach')}
* **Consequences**: {state.get('wc_consequences')}
* **New Way**: {state.get('wc_new_way')}
* **Likely Outcomes (Project)**: {state.get('wc_likely_outcomes_project')}
* **Likely Outcomes (Business Unit)**: {state.get('wc_likely_outcomes_business_unit')}
* **Likely Outcomes (Corporate)**: {state.get('wc_likely_outcomes_corporate')}
* **Active Status Quo Biases Addressed**: {biases_identified}"""

    summary = f"""Great! We have aligned on the problem statement using the REALIGN framework.

Next Steps: I will prepare a proposal outlining our proposed architecture and timeline, and schedule a review call next week.

Here is a summary of the discovery reasoning and how the information was mapped under the ALIGN process:

{response.text}{summary_suffix}

Thank you for the discovery session. Looking forward to our next meeting!

[Session Status: Concluded Successfully]"""

    return to_event(summary)

# --- Workflow Definition ---

root_agent = Workflow(
    name="root_agent",
    state_schema=DiscoveryState,
    edges=[
        (START, triage_and_draft),
        (triage_and_draft, {
            "related": ask_question_node,
            "unrelated": decline_node
        }),
        
        # Discovery Loop:
        (ask_question_node, discovery_loop),
        (discovery_loop, {"ask_question": ask_question_node, "finalize": get_agreement}),
        
        # Discovery Agreement Phase:
        (get_agreement, process_agreement),
        (process_agreement, {"agreed": why_change_start_node, "disagreed": ask_question_node}),
        
        # Why Change Loop:
        (why_change_start_node, why_change_ask_node),
        (why_change_ask_node, why_change_loop),
        (why_change_loop, {"ask": why_change_ask_node, "finalize": why_change_agreement}),
        
        # Why Change Agreement Phase:
        (why_change_agreement, process_wc_agreement),
        (process_wc_agreement, {"agreed": reasoning_summary_node, "disagreed": why_change_ask_node}),
    ]
)

app = App(
    root_agent=root_agent,
    name="app",
)


