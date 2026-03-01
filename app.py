
import os
import json
import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
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
Always cite the specific source of each recommendation."""

MAX_RETRIES = 2

# ─── Build Graph Function ─────────────────────────────────
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
    search = DuckDuckGoSearchRun()

    def enrich_query(state):
        prompt = f"""A USA farmer asked: "{state['farmer_query']}"
Return JSON only with these fields: crop, pest_or_weed, region, constraints, enriched_query"""
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            content = response.content.strip()
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'): content = content[4:]
            parsed = json.loads(content.strip())
            state.update({k: parsed.get(k, 'unknown') for k in ['crop','pest_or_weed','region','constraints']})
            state['enriched_query'] = parsed.get('enriched_query', state['farmer_query'])
        except:
            state['enriched_query'] = state['farmer_query']
            for k in ['crop','pest_or_weed','region','constraints']: state[k] = 'unknown'
        return state

    def retrieve(state):
        query = state.get('refined_query') if state.get('retry_count',0) > 0 else state.get('enriched_query')
        try:
            results = search.run(query)
            state['retrieved_docs'] = results
            state['sources'] = [f'DuckDuckGo: {query}']
        except:
            state['retrieved_docs'] = 'Search failed'
            state['sources'] = []
        return state

    def generate_answer(state):
        prompt = f"""Using ONLY this retrieved context, answer the farmer question.
Farmer: {state['farmer_query']} | Crop: {state['crop']} | Pest/Weed: {state['pest_or_weed']} | Region: {state['region']} | Constraints: {state['constraints']}
Context: {state['retrieved_docs']}
Structure: Operation Identified, Current Practice, Alternative Technologies Found, Best Recommendation, Questions to Explore Next"""
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        state['draft_answer'] = response.content
        return state

    def grade_answer(state):
        prompt = f"""Evaluate this answer for a USA farmer. Return JSON only with grade (good/poor), reason, missing.
Question: {state['farmer_query']} | Constraints: {state['constraints']} | Answer: {state['draft_answer']}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            content = response.content.strip()
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'): content = content[4:]
            parsed = json.loads(content.strip())
            state['grade'] = parsed.get('grade', 'poor')
            state['grade_reason'] = parsed.get('reason', '')
            state['missing_info'] = parsed.get('missing', '')
        except:
            state['grade'] = 'poor'
            state['grade_reason'] = 'Parse error'
            state['missing_info'] = 'Retry'
        return state

    def refine_query(state):
        prompt = f"""Generate a better search query for missing info. Return JSON with refined_query only.
Missing: {state['missing_info']} | Original query: {state['enriched_query']}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            content = response.content.strip()
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'): content = content[4:]
            parsed = json.loads(content.strip())
            state['refined_query'] = parsed.get('refined_query', state['enriched_query'] + ' USA')
        except:
            state['refined_query'] = state['enriched_query'] + ' USA alternatives'
        state['retry_count'] = state.get('retry_count', 0) + 1
        return state

    def output(state):
        if state['grade'] == 'good':
            state['final_answer'] = state['draft_answer']
        else:
            state['final_answer'] = "⚠️ Could not find complete information. Best available answer below — verify with local extension service.\n\n---\n\n" + state['draft_answer']
        return state

    def decide(state):
        if state['grade'] == 'good': return 'output'
        if state.get('retry_count', 0) >= MAX_RETRIES: return 'output'
        return 'refine_query'

    wf = StateGraph(AgRagState)
    for name, fn in [('enrich_query',enrich_query),('retrieve',retrieve),('generate_answer',generate_answer),('grade_answer',grade_answer),('refine_query',refine_query),('output',output)]:
        wf.add_node(name, fn)
    wf.set_entry_point('enrich_query')
    wf.add_edge('enrich_query', 'retrieve')
    wf.add_edge('retrieve', 'generate_answer')
    wf.add_edge('generate_answer', 'grade_answer')
    wf.add_edge('refine_query', 'retrieve')
    wf.add_edge('output', END)
    wf.add_conditional_edges('grade_answer', decide, {'output': 'output', 'refine_query': 'refine_query'})
    return wf.compile()

# ─── Streamlit Interface ──────────────────────────────────
def main():
    st.set_page_config(page_title="AgTech Alternatives Finder", page_icon="🌾", layout="wide")

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
        st.caption("This tool searches trusted USA agricultural sources to find alternatives to conventional farming practices.")

    st.title("🌾 AgTech Alternatives Finder")
    st.markdown("Find modern, sustainable alternatives to conventional pest management and weeding practices — tailored for USA farmers.")

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

    st.markdown("### Describe Your Situation")
    farmer_query = st.text_area(
        label="question",
        placeholder="Example: I currently spray herbicide on my corn farm in Iowa for weed control. I want to explore autonomous or biological alternatives that are cost effective.",
        height=120,
        label_visibility="collapsed"
    )

    if st.button("Find Alternatives", type="primary"):
        if not farmer_query:
            st.warning("Please describe your situation first.")
            return

        with st.spinner("🔍 Searching trusted agricultural sources..."):
            try:
                g = build_graph(groq_key, langsmith_key)
                result = g.invoke(AgRagState(
                    farmer_query=farmer_query, enriched_query='', crop='',
                    pest_or_weed='', region='', constraints='',
                    retrieved_docs='', sources=[], draft_answer='',
                    grade='', grade_reason='', missing_info='',
                    refined_query='', retry_count=0, final_answer=''
                ))

                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Crop", result["crop"].title())
                col2.metric("Pest / Weed", result["pest_or_weed"].title())
                col3.metric("Region", result["region"].title())
                col4.metric("Search Refinements", result["retry_count"])

                st.markdown("---")
                st.markdown("### 🌱 Alternatives Found")
                st.markdown(result["final_answer"])

                with st.expander("🔍 How this answer was evaluated"):
                    st.markdown(f"**Final Grade:** {result['grade'].upper()}")
                    st.markdown(f"**Grade Reason:** {result['grade_reason']}")
                    if result['retry_count'] > 0:
                        st.markdown(f"**Retries:** {result['retry_count']} — system searched again with a more targeted query")
                        st.markdown(f"**What triggered retry:** {result['missing_info']}")

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.markdown("Check your API keys and try again.")

if __name__ == "__main__":
    main()
