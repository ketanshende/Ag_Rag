import os
import json
import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage

# ─── State Definition ─────────────────────────────────────
class AgRagState(TypedDict):
    farmer_query: str
    enriched_query: str
    crop: str
    pest_or_weed: str
    region: str
    constraints: str
    retrieved_docs: str
    sources: List[str]
    draft_answer: str
    grade: str
    grade_reason: str
    missing_info: str
    refined_query: str
    retry_count: int
    final_answer: str

# ─── System Prompt ────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert agricultural technology advisor 
specializing in USA farming practices. You help farmers find modern, 
sustainable, and autonomous alternatives to conventional pest management 
and weeding practices. Answer based ONLY on retrieved context provided.
Always cite specific named sources for each recommendation."""

MAX_RETRIES = 2

# ─── Intent Detection ─────────────────────────────────────
def is_agricultural_query(query, llm):
    prompt = f"""A user sent this message to an agricultural technology 
advisory tool for USA farmers:

"{query}"

Classify this message and return JSON only — no preamble, no markdown:
{{
    "is_agricultural": true or false,
    "reason": "why you classified it this way",
    "conversational_response": "if not agricultural, a friendly short response that explains this tool helps USA farmers find alternatives to conventional farming practices — otherwise empty string"
}}

Examples of AGRICULTURAL: questions about crops, pests, weeds, 
farm equipment, herbicides, pesticides, irrigation, harvesting, 
soil management, organic certification, farm costs, weed control,
biological alternatives, precision agriculture, autonomous farming

Examples of NOT AGRICULTURAL: greetings, general chat, 
coding questions, math problems, personal questions,
questions completely unrelated to farming"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        return parsed.get("is_agricultural", True), parsed.get("conversational_response", "")
    except:
        return True, ""


# ─── Build Graph ──────────────────────────────────────────
def build_graph(groq_key, langsmith_key):

    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "ag-rag-system"

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_key,
        temperature=0
    )
    search = DuckDuckGoSearchResults()

    def enrich_query(state):
        prompt = f"""A USA farmer asked: "{state["farmer_query"]}"

Return JSON only — no preamble, no markdown, pure JSON:
{{
    "crop": "crop mentioned or unknown",
    "pest_or_weed": "pest or weed mentioned or unknown",
    "region": "US state or region mentioned or unknown",
    "constraints": "constraints like organic certified, budget, farm size — or none",
    "enriched_query": "expanded 15-20 word search query for finding alternative agricultural technologies from USA extension services and USDA"
}}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            for k in ["crop", "pest_or_weed", "region", "constraints"]:
                state[k] = parsed.get(k, "unknown")
            state["enriched_query"] = parsed.get("enriched_query", state["farmer_query"])
        except:
            state["enriched_query"] = state["farmer_query"]
            for k in ["crop", "pest_or_weed", "region", "constraints"]:
                state[k] = "unknown"
        return state

    def retrieve(state):
        base_query = state.get("refined_query") if state.get("retry_count", 0) > 0 else state.get("enriched_query")
        queries = [
            base_query,
            f"{state['pest_or_weed']} {state['crop']} alternatives USA extension service",
            f"{state['pest_or_weed']} {state['crop']} biological mechanical control cost USDA"
        ]
        all_results = ""
        sources_used = []
        for q in queries:
            try:
                result = search.invoke(q)
                all_results += f"\n\nSEARCH: {q}\nRESULTS: {result}\n"
                sources_used.append(q)
            except:
                pass
        state["retrieved_docs"] = all_results
        state["sources"] = sources_used
        return state

    def generate_answer(state):
        prompt = f"""Using ONLY the retrieved context below, answer the farmer question.
Do not invent information not present in the context.
Follow the output format exactly — do not use numbered lists across sections.

FARMER QUESTION: {state["farmer_query"]}
FARMER CONTEXT:
- Crop: {state["crop"]}
- Pest or Weed: {state["pest_or_weed"]}
- Region: {state["region"]}
- Constraints: {state["constraints"]}

RETRIEVED CONTEXT:
{state["retrieved_docs"]}

Output format — follow exactly:

**Operation Identified:** [what the farmer is managing]
**Current Practice:** [what they currently use]

---

**Alternative Technologies Found:**

For each alternative:
**[Technology Name]** | [Category] | [USA Availability]
- How it works: ...
- What it replaces: ...
- Key benefits: ...
- Limitations: ...
- Approximate cost: ...
- Source: [named organization or URL — not just a date]

---

**Best Recommendation for Your Situation:**
[Specific advice based on crop, region, scale, and constraints]

**Questions to Explore Next:**
- [Question 1]
- [Question 2]
- [Question 3]"""

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        state["draft_answer"] = response.content
        return state

    def grade_answer(state):
        prompt = f"""Evaluate this answer for a USA farmer.

FARMER QUESTION: {state["farmer_query"]}
FARMER CONSTRAINTS: {state["constraints"]}
GENERATED ANSWER: {state["draft_answer"]}

Grade as POOR if ANY of these are true:
- No specific named technologies or products mentioned
- Sources are just dates with no named organizations or URLs
- No approximate costs mentioned
- Answer does not address the specific farm scale mentioned
- Answer is fewer than 200 words
- Recommendations not confirmed as available in the USA

Grade as GOOD only if ALL of these are true:
- At least 2 specific named technologies or products mentioned
- At least one source is a named organization like a university, USDA, or company
- Farm scale and constraints are addressed
- Answer is specific and actionable

Return JSON only — no preamble, no markdown:
{{
    "grade": "good or poor",
    "reason": "specific reason",
    "missing": "exactly what is missing — be very specific"
}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            state["grade"] = parsed.get("grade", "poor")
            state["grade_reason"] = parsed.get("reason", "")
            state["missing_info"] = parsed.get("missing", "")
        except:
            state["grade"] = "poor"
            state["grade_reason"] = "Parse error"
            state["missing_info"] = "Retry with more specific query"
        return state

    def refine_query(state):
        prompt = f"""Generate a better search query for the missing information.
Return JSON only — no preamble, no markdown:
{{
    "refined_query": "new targeted 15-25 word search query"
}}

Missing info: {state["missing_info"]}
Original query: {state["enriched_query"]}
Crop: {state["crop"]}
Pest or weed: {state["pest_or_weed"]}
Region: {state["region"]}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            state["refined_query"] = parsed.get(
                "refined_query",
                state["enriched_query"] + " USA specific recommendations"
            )
        except:
            state["refined_query"] = state["enriched_query"] + " USA alternatives"
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state

    def output(state):
        if state["grade"] == "good":
            state["final_answer"] = state["draft_answer"]
        else:
            state["final_answer"] = (
                "⚠️ **Note:** I searched multiple times but could not find "
                "complete information matching all your requirements. "
                "The answer below is the best available — please verify "
                "with your local extension service before making decisions."
                "\n\n---\n\n"
                + state["draft_answer"]
            )
        return state

    def decide(state):
        if state["grade"] == "good":
            return "output"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "output"
        return "refine_query"

    wf = StateGraph(AgRagState)
    for name, fn in [
        ("enrich_query", enrich_query),
        ("retrieve", retrieve),
        ("generate_answer", generate_answer),
        ("grade_answer", grade_answer),
        ("refine_query", refine_query),
        ("output", output)
    ]:
        wf.add_node(name, fn)

    wf.set_entry_point("enrich_query")
    wf.add_edge("enrich_query", "retrieve")
    wf.add_edge("retrieve", "generate_answer")
    wf.add_edge("generate_answer", "grade_answer")
    wf.add_edge("refine_query", "retrieve")
    wf.add_edge("output", END)
    wf.add_conditional_edges(
        "grade_answer",
        decide,
        {"output": "output", "refine_query": "refine_query"}
    )
    return wf.compile(), llm


# ─── Streamlit Interface ──────────────────────────────────
def main():
    st.set_page_config(
        page_title="AgTech Alternatives Finder",
        page_icon="🌾",
        layout="wide"
    )

    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        langsmith_key = st.text_input("LangSmith API Key", type="password", placeholder="lsv2_...")
        st.markdown("---")
        st.markdown("**Get your free keys:**")
        st.markdown("🔑 [Groq Console](https://console.groq.com)")
        st.markdown("📊 [LangSmith](https://smith.langchain.com)")
        st.markdown("---")
        st.caption(
            "Searches trusted USA agricultural sources including USDA, "
            "university extension services, ATTRA, and SARE."
        )

    st.title("🌾 AgTech Alternatives Finder")
    st.markdown(
        "Find modern, sustainable alternatives to conventional pest "
        "management and weeding practices — tailored for USA farmers."
    )

    if not groq_key or not langsmith_key:
        st.info("👈 Please enter your API keys in the sidebar to get started.")
        st.markdown("---")
        st.markdown("### Example Questions")
        st.markdown("""
- *I spray chlorpyrifos on my 80 acre corn farm in Iowa for aphids. What biological alternatives exist?*
- *I use glyphosate for weeds in soybeans in Illinois. I am organic certified. What are my options?*
- *I have fall armyworm on my corn in Texas. What autonomous technologies can help?*
- *I hand weed my 5 acre vegetable farm in California. What mechanical alternatives work at my scale?*
        """)
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("metadata", {}).get("is_agricultural"):
                meta = message["metadata"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Crop", meta.get("crop", "-").title())
                col2.metric("Pest / Weed", meta.get("pest_or_weed", "-").title())
                col3.metric("Region", meta.get("region", "-").title())
                col4.metric("Refinements", meta.get("retry_count", 0))
                with st.expander("🔍 How this answer was evaluated"):
                    st.markdown(f"**Grade:** {meta.get('grade', '-').upper()}")
                    st.markdown(f"**Reason:** {meta.get('grade_reason', '-')}")
                    if meta.get("retry_count", 0) > 0:
                        st.markdown(f"**What triggered retry:** {meta.get('missing_info', '-')}")

    farmer_input = st.chat_input(
        "Ask about pest management, weeding alternatives, or anything farming related..."
    )

    if farmer_input:
        with st.chat_message("user"):
            st.markdown(farmer_input)
        st.session_state.messages.append({"role": "user", "content": farmer_input})

        graph, llm = build_graph(groq_key, langsmith_key)
        is_ag, conv_response = is_agricultural_query(farmer_input, llm)

        with st.chat_message("assistant"):
            if not is_ag:
                st.markdown(conv_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": conv_response,
                    "metadata": {"is_agricultural": False}
                })
            else:
                with st.spinner("🔍 Searching trusted agricultural sources..."):
                    try:
                        result = graph.invoke(AgRagState(
                            farmer_query=farmer_input,
                            enriched_query="", crop="", pest_or_weed="",
                            region="", constraints="", retrieved_docs="",
                            sources=[], draft_answer="", grade="",
                            grade_reason="", missing_info="",
                            refined_query="", retry_count=0, final_answer=""
                        ))

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Crop", result["crop"].title())
                        col2.metric("Pest / Weed", result["pest_or_weed"].title())
                        col3.metric("Region", result["region"].title())
                        col4.metric("Refinements", result["retry_count"])
                        st.markdown("---")
                        st.markdown(result["final_answer"])

                        with st.expander("🔍 How this answer was evaluated"):
                            st.markdown(f"**Grade:** {result['grade'].upper()}")
                            st.markdown(f"**Reason:** {result['grade_reason']}")
                            if result["retry_count"] > 0:
                                st.markdown(f"**Retries:** {result['retry_count']}")
                                st.markdown(f"**What triggered retry:** {result['missing_info']}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["final_answer"],
                            "metadata": {
                                "is_agricultural": True,
                                "crop": result["crop"],
                                "pest_or_weed": result["pest_or_weed"],
                                "region": result["region"],
                                "retry_count": result["retry_count"],
                                "grade": result["grade"],
                                "grade_reason": result["grade_reason"],
                                "missing_info": result["missing_info"]
                            }
                        })

                    except Exception as e:
                        error_msg = f"Something went wrong: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "metadata": {"is_agricultural": False}
                        })

if __name__ == "__main__":
    main()
