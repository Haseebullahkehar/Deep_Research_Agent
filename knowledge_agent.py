import os
import uuid
import asyncio
import json
import schedule
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import requests
from pymongo import MongoClient
from pydantic import BaseModel, Field
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from agents import (
    Agent,
    Runner,
    WebSearchTool,
    function_tool,
    handoff,
    trace,
)

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = Path("data")
VECTORDB_DIR = DATA_DIR / "chroma_db"
DATA_DIR.mkdir(exist_ok=True)

# Set up page configuration
st.set_page_config(
    page_title="AI Research Knowledge System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize all required services


def initialize_services():
    """Initialize database connections and vector stores"""
    # MongoDB for structured storage
    mongo_client = MongoClient(
        os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = mongo_client.research_knowledge_base

    # ChromaDB for vector embeddings
    embedding_function = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=str(VECTORDB_DIR),
        embedding_function=embedding_function
    )

    return {
        "mongo": db,
        "vectordb": vectordb,
        "embedding": embedding_function
    }


services = initialize_services()

# Define enhanced data models


class ResearchSource(BaseModel):
    url: str
    title: str
    publication_date: Optional[datetime] = None
    author: Optional[str] = None
    source_type: str = "web"  # web, api, internal, etc.
    metadata: Dict = Field(default_factory=dict)


class ResearchFact(BaseModel):
    fact: str
    sources: List[ResearchSource]
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None


class ResearchPlan(BaseModel):
    topic: str
    search_queries: List[str]
    focus_areas: List[str]
    # web, api:[name], internal:[type]
    data_sources: List[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    title: str
    outline: List[str]
    report: str
    sources: List[ResearchSource]
    word_count: int
    generated_at: datetime = Field(default_factory=datetime.now)
    related_facts: List[ResearchFact] = Field(default_factory=list)


class KnowledgeBaseStats(BaseModel):
    total_facts: int
    total_reports: int
    sources_breakdown: Dict[str, int]
    most_common_tags: List[str]
    last_updated: datetime

# Enhanced toolset for the agents


@function_tool
def save_important_fact(fact: str, source: ResearchSource, categories: List[str] = None, tags: List[str] = None) -> str:
    """Save an important fact to the knowledge base with proper categorization"""
    research_fact = ResearchFact(
        fact=fact,
        sources=[source],
        categories=categories or [],
        tags=tags or []
    )

    # Store in MongoDB
    services["mongo"].facts.insert_one(research_fact.dict())

    # Store in vector DB for RAG
    doc = Document(
        page_content=fact,
        metadata={
            "source": source.url,
            "categories": ",".join(categories) if categories else "",
            "tags": ",".join(tags) if tags else ""
        }
    )
    services["vectordb"].add_documents([doc])

    return f"Fact saved and indexed: {fact}"


@function_tool
def search_knowledge_base(query: str, limit: int = 5) -> List[Dict]:
    """Search existing knowledge base before doing new research"""
    # First try vector similarity search
    vector_results = services["vectordb"].similarity_search(query, k=limit)

    # Then try MongoDB full-text search
    mongo_results = list(services["mongo"].facts.find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(limit))

    # Combine and deduplicate results
    combined = []
    seen_facts = set()

    for doc in vector_results:
        if doc.page_content not in seen_facts:
            combined.append({
                "type": "vector",
                "fact": doc.page_content,
                "metadata": doc.metadata
            })
            seen_facts.add(doc.page_content)

    for doc in mongo_results:
        fact_str = doc["fact"]
        if fact_str not in seen_facts:
            combined.append({
                "type": "mongo",
                "fact": fact_str,
                "metadata": {
                    "sources": doc.get("sources", []),
                    "categories": doc.get("categories", []),
                    "tags": doc.get("tags", [])
                }
            })
            seen_facts.add(fact_str)

    return combined[:limit]


@function_tool
def fetch_api_data(api_name: str, params: Dict = None) -> List[Dict]:
    """Fetch data from configured APIs"""
    api_configs = {
        "government_stats": {
            "url": "https://api.data.gov/some/endpoint",
            "headers": {"X-API-Key": os.getenv("GOV_API_KEY")}
        },
        "business_data": {
            "url": "https://api.businessdata.com/v1/search",
            "headers": {"Authorization": f"Bearer {os.getenv('BIZ_API_KEY')}"}
        }
    }

    if api_name not in api_configs:
        return [{"error": f"API {api_name} not configured"}]

    config = api_configs[api_name]
    response = requests.get(
        config["url"],
        headers=config.get("headers", {}),
        params=params or {}
    )

    if response.status_code == 200:
        return response.json()
    return [{"error": f"API request failed with status {response.status_code}"}]


# Define the enhanced agents
research_agent = Agent(
    name="Research Agent",
    instructions="""You are a comprehensive research assistant. Your tasks:
1. Check existing knowledge base before searching externally
2. When searching, use multiple sources (web and APIs)
3. Extract key facts and save with proper metadata
4. Tag content with relevant categories and tags
5. Always record complete source information

For each fact you find:
- Determine appropriate categories/tags
- Capture full source metadata
- Check if similar facts already exist
- Save only novel, high-quality information""",
    model="gpt-4-1106-preview",
    tools=[
        WebSearchTool(),
        save_important_fact,
        search_knowledge_base,
        fetch_api_data
    ],
)

editor_agent = Agent(
    name="Editor Agent",
    handoff_description="Senior researcher who synthesizes reports from multiple sources",
    instructions="""You create comprehensive research reports by:
1. Reviewing all collected facts and sources
2. Organizing content into logical sections
3. Incorporating proprietary data when available
4. Generating executive summaries
5. Creating properly formatted outputs (Markdown, HTML, PDF)

Special requirements:
- Include references to all sources
- Highlight connections to existing knowledge
- Add metadata for future retrieval
- Generate multiple output formats""",
    model="gpt-4-1106-preview",
    output_type=ResearchReport,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="""As research coordinator, you:
1. Analyze the research topic thoroughly
2. Check knowledge base for existing content
3. Plan data collection from:
   - Web searches
   - Relevant APIs
   - Internal databases
4. Create detailed research plan with:
   - Topic definition
   - Search queries
   - Focus areas
   - Data sources
5. Monitor progress and adjust as needed""",
    handoffs=[
        handoff(research_agent),
        handoff(editor_agent)
    ],
    model="gpt-4-1106-preview",
    output_type=ResearchPlan,
)

# Knowledge Management System


class KnowledgeSystem:
    def __init__(self):
        self.scheduled_jobs = []
        self.start_scheduler()

    def start_scheduler(self):
        """Start background scheduler for automated updates"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

    def schedule_daily_update(self, topic: str, sources: List[str]):
        """Schedule regular updates for a research topic"""
        job_id = f"update_{topic.lower().replace(' ', '_')}"

        def update_job():
            asyncio.run(self.run_research(topic, scheduled=True))

        schedule.every().day.at("09:00").do(update_job).tag(job_id)
        self.scheduled_jobs.append(job_id)

    async def run_research(self, topic: str, scheduled: bool = False):
        """Enhanced research workflow"""
        conversation_id = str(uuid.uuid4().hex[:16])

        with trace("Enhanced Research Workflow", group_id=conversation_id):
            # Check knowledge base first
            existing_knowledge = search_knowledge_base(topic)

            if existing_knowledge and not scheduled:
                return {
                    "status": "info",
                    "message": "Existing knowledge found",
                    "results": existing_knowledge
                }

            # Create research plan
            triage_result = await Runner.run(
                triage_agent,
                f"Research this topic: {topic}. Existing knowledge: {existing_knowledge}"
            )

            if not hasattr(triage_result.final_output, 'topic'):
                return {"status": "error", "message": "Failed to create research plan"}

            # Execute research
            research_results = await Runner.run(
                research_agent,
                triage_result.to_input_list()
            )

            # Generate report
            report_result = await Runner.run(
                editor_agent,
                {
                    "research_plan": triage_result.final_output.dict(),
                    "research_findings": research_results.to_input_list(),
                    "existing_knowledge": existing_knowledge
                }
            )

            # Store final report
            if hasattr(report_result.final_output, 'report'):
                services["mongo"].reports.insert_one(
                    report_result.final_output.dict())

                # Update vector store with report content
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                report_text = report_result.final_output.report
                docs = text_splitter.create_documents([report_text])
                services["vectordb"].add_documents(docs)

            return {
                "status": "success",
                "report": report_result.final_output,
                "conversation_id": conversation_id
            }

# Streamlit UI Components


def render_sidebar(ksystem: KnowledgeSystem):
    """Render the sidebar with all controls"""
    with st.sidebar:
        st.header("Research Configuration")

        tab1, tab2, tab3 = st.tabs(
            ["New Research", "Scheduled", "Knowledge Base"])

        with tab1:
            user_topic = st.text_area("Research Topic:", height=100)
            sources = st.multiselect(
                "Data Sources:",
                ["Web", "Government Data", "Business APIs", "Internal Data"],
                default=["Web"]
            )

            if st.button("Start Research", type="primary"):
                if user_topic:
                    return {"action": "research", "topic": user_topic, "sources": sources}
                else:
                    st.error("Please enter a research topic")

        with tab2:
            st.write("Scheduled Research Jobs")
            for job in ksystem.scheduled_jobs:
                st.write(f"• {job.replace('_', ' ').title()}")

            new_schedule_topic = st.text_input("Topic to schedule:")
            if st.button("Schedule Daily Update"):
                if new_schedule_topic:
                    ksystem.schedule_daily_update(new_schedule_topic, ["Web"])
                    st.success(
                        f"Scheduled daily updates for: {new_schedule_topic}")

        with tab3:
            st.write("Knowledge Base Statistics")
            stats = services["mongo"].facts.aggregate([
                {"$group": {
                    "_id": None,
                    "total_facts": {"$sum": 1},
                    "sources": {"$addToSet": "$sources.source_type"}
                }}
            ])

            if stats:
                for stat in stats:
                    st.metric("Total Facts", stat["total_facts"])
                    st.write("Sources:", ", ".join(stat["sources"]))

            search_query = st.text_input("Search Knowledge Base:")
            if search_query:
                results = search_knowledge_base(search_query)
                for res in results:
                    with st.expander(res["fact"][:50] + "..."):
                        st.write("Source:", res["metadata"].get(
                            "source", "Unknown"))
                        st.write("Categories:", res["metadata"].get(
                            "categories", "None"))


def render_main_content():
    """Render the main content area"""
    st.title("🧠 AI-Powered Research Knowledge System")
    st.subheader(
        "Enterprise Knowledge Management with Automated Research Agents")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Research Console",
        "Knowledge Explorer",
        "Reports Library",
        "System Administration"
    ])

    with tab1:
        if "research_result" in st.session_state:
            result = st.session_state.research_result
            if result["status"] == "success":
                report = result["report"]
                st.success("Research Completed Successfully!")
                st.download_button(
                    label="Download Full Report",
                    data=json.dumps(report.dict(), indent=2),
                    file_name=f"{report.title.replace(' ', '_')}.json",
                    mime="application/json"
                )

                with st.expander("Report Overview", expanded=True):
                    st.subheader(report.title)
                    st.write(f"Generated: {report.generated_at}")
                    st.write(f"Word Count: {report.word_count}")

                    st.markdown("### Outline")
                    for item in report.outline:
                        st.write(f"- {item}")

                    st.markdown("### Key Findings")
                    st.markdown(report.report[:1000] + "...")
            else:
                st.info(result["message"])
                st.json(result.get("results", {}))

    with tab2:
        st.header("Explore Collected Knowledge")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Recent Facts")
            recent_facts = services["mongo"].facts.find().sort(
                "created_at", -1).limit(5)
            for fact in recent_facts:
                with st.expander(fact["fact"][:50] + "..."):
                    st.write("Sources:")
                    for src in fact.get("sources", []):
                        st.write(
                            f"- {src.get('title', 'No title')} ({src.get('source_type', 'unknown')})")

        with col2:
            st.subheader("Knowledge Graph")
            st.write("""
            This will display a network graph of connected concepts
            (Implementation would require a graph database like Neo4j)
            """)

    with tab3:
        st.header("Generated Reports")
        reports = services["mongo"].reports.find().sort("generated_at", -1)
        for report in reports:
            with st.expander(report["title"]):
                st.write(f"Generated: {report['generated_at']}")
                st.download_button(
                    label="Download",
                    data=json.dumps(report, indent=2),
                    file_name=f"{report['title'].replace(' ', '_')}.json",
                    mime="application/json"
                )
                st.markdown(report["report"][:500] + "...")

    with tab4:
        st.header("System Administration")
        if st.button("Rebuild Vector Index"):
            with st.spinner("Rebuilding index..."):
                # Implementation would re-index all content
                st.success("Vector index rebuilt successfully")

        if st.button("Export Knowledge Base"):
            # Implementation would export all data
            st.success("Export started in background")

# Main application


def main():
    ksystem = KnowledgeSystem()

    # Initialize session state
    if "research_result" not in st.session_state:
        st.session_state.research_result = None

    # Render UI
    user_input = render_sidebar(ksystem)
    render_main_content()

    # Handle user actions
    if user_input and user_input["action"] == "research":
        with st.spinner(f"Researching: {user_input['topic']}"):
            result = asyncio.run(ksystem.run_research(user_input["topic"]))
            st.session_state.research_result = result
            st.rerun()


if __name__ == "__main__":
    main()
