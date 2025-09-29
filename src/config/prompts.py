"""
RAG System Prompts Configuration
Centralized prompt management for different analysis types and use cases.
"""

from typing import Dict, Any
from enum import Enum

class AnalysisType(Enum):
    SUMMARY = "summary"
    DEEP_THINKING = "deep_thinking"
    DETAILED = "detailed"
    STANDARD = "standard"

class PromptTemplates:
    """Centralized prompt templates for the RAG system"""
    
    # Base system prompts
    BASE_SYSTEM_PROMPT = """You are an intelligent document analysis assistant. Your role is to provide accurate, helpful answers based on the provided context documents."""
    
    # Analysis type specific prompts
    PROMPTS = {
        AnalysisType.SUMMARY: {
            "system": """You are a research assistant providing clear, concise summaries from multiple document sources.""",
            "user_template": """Instructions for your Summary response:
1. Provide a focused, well-structured answer that directly addresses the question
2. Use clear bullet points and organize information logically
3. Include the most relevant details and key examples from the documents
4. Mention specific sources when referencing information
5. Keep the response comprehensive but concise
6. Focus on actionable insights and main conclusions
7. Avoid unnecessary elaboration while maintaining accuracy

This search spans across multiple previously uploaded document sets from persistent storage.

Context from multiple sources:
{context}

Question: {input}

Please provide a clear, focused summary that addresses the question:""",
            "retrieval_params": {
                "k": 5,
                "score_threshold": 0.7
            }
        },
        
        AnalysisType.DEEP_THINKING: {
            "system": """You are an expert research analyst conducting comprehensive "Deep Thinking" analysis across multiple document sources.""",
            "user_template": """Instructions for your Deep Thinking response:
1. Provide an exhaustive, detailed analysis that explores the question from multiple angles
2. Structure your response with clear sections, subsections, and detailed bullet points
3. Include specific details, examples, evidence, and data from the documents
4. Analyze connections, relationships, and patterns between different concepts
5. Present multiple perspectives, compare different viewpoints, and analyze contradictions
6. Identify underlying principles, methodologies, and theoretical frameworks
7. Provide historical context, current state, and future implications when relevant
8. Include actionable recommendations, best practices, and implementation strategies
9. Discuss limitations, potential risks, and areas requiring further investigation
10. Synthesize insights across all sources to provide comprehensive understanding
11. Use specific citations and mention file names when referencing information
12. Think step-by-step and show your reasoning process

This comprehensive Deep Thinking analysis spans across multiple previously uploaded document sets.

Context from multiple sources:
{context}

Question: {input}

Please provide a thorough Deep Thinking analysis that comprehensively addresses all aspects of the question:""",
            "retrieval_params": {
                "k": 8,
                "score_threshold": 0.6
            }
        },
        
        AnalysisType.DETAILED: {
            "system": """You are an expert research analyst. Provide comprehensive, detailed answers based on the provided context.""",
            "user_template": """Instructions:
1. Give a thorough analysis of the question using all relevant information from the context
2. Structure your response with clear sections and subsections when appropriate
3. Include specific details, examples, and evidence from the documents
4. Explain the implications and connections between different concepts
5. Cite specific sources when referencing information
6. If multiple perspectives exist, present them fairly
7. Identify any limitations or gaps in the available information
8. Provide actionable insights or recommendations when relevant

Context from multiple sources:
{context}

Question: {input}

Please provide a detailed, well-structured response that thoroughly addresses the question:""",
            "retrieval_params": {
                "k": 6,
                "score_threshold": 0.65
            }
        },
        
        AnalysisType.STANDARD: {
            "system": """You are a helpful assistant that answers questions based on provided context.""",
            "user_template": """Answer the following question based only on the provided context.
If you don't know the answer, just say that you don't know.
When referencing information, mention which file/source it came from.

Context:
{context}

Question: {input}""",
            "retrieval_params": {
                "k": 4,
                "score_threshold": 0.75
            }
        }
    }
    
    # Specialized prompts for different scenarios
    GLOBAL_SEARCH_PROMPT = """You are an expert analyst performing cross-document research. You have access to multiple document collections and need to synthesize information across all sources.

Question: {query}

Available Documents from Multiple Sources:
{context}

Please provide a comprehensive answer that:
1. Synthesizes information from all available sources
2. Identifies patterns and connections across documents
3. Provides a well-structured, informative response
4. Notes any conflicting information or gaps
5. Cites specific sources when possible

Focus on providing valuable insights that leverage the breadth of available information."""

    URL_ANALYSIS_PROMPT = """You are analyzing content fetched directly from web URLs. Provide accurate analysis based on the current content.

Question: {query}

Web Content:
{context}

Please analyze the web content and provide:
1. Direct answer to the question
2. Key information from the content
3. Relevant details and examples
4. Assessment of content quality and relevance

Note: This content was fetched in real-time from the web."""

    GOOGLE_SEARCH_PROMPT = """Based on the following Google search results, provide a helpful answer to the user's question.
If the search results don't contain relevant information, say so.
Always mention that this information comes from Google search results and include relevant URLs.

Google Search Results:
{context}

Question: {query}

Please provide a comprehensive answer based on the search results:"""

    GOOGLE_FALLBACK_PROMPT = """The original question could not be answered from the available documents. Here are relevant search results from Google:

Original Question: {query}
Original RAG Response: {original_response}

Google Search Results:
{google_results}

Please provide a helpful response based on the search results, noting that this information comes from web search since the original documents didn't contain the answer."""

    # Error and fallback messages
    INSUFFICIENT_CONTEXT_MESSAGE = "I don't have enough information in the provided documents to answer this question accurately. Please try rephrasing your question or providing additional context."
    
    NO_RELEVANT_DOCS_MESSAGE = "No relevant documents were found for your query. This might mean the question is outside the scope of the available documents."
    
    PROCESSING_ERROR_MESSAGE = "I encountered an error while processing your request. Please try again or contact support if the issue persists."

    @classmethod
    def get_prompt(cls, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Get prompt configuration for a specific analysis type"""
        return cls.PROMPTS.get(analysis_type, cls.PROMPTS[AnalysisType.STANDARD])
    
    @classmethod
    def get_system_prompt(cls, analysis_type: AnalysisType) -> str:
        """Get system prompt for a specific analysis type"""
        return cls.get_prompt(analysis_type)["system"]
    
    @classmethod
    def get_user_template(cls, analysis_type: AnalysisType) -> str:
        """Get user prompt template for a specific analysis type"""
        return cls.get_prompt(analysis_type)["user_template"]
    
    @classmethod
    def get_retrieval_params(cls, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Get retrieval parameters for a specific analysis type"""
        return cls.get_prompt(analysis_type)["retrieval_params"]
    
    @classmethod
    def format_user_prompt(cls, analysis_type: AnalysisType, query: str, context: str) -> str:
        """Format the user prompt with query and context"""
        template = cls.get_user_template(analysis_type)
        return template.format(input=query, context=context)

# Convenience functions for backward compatibility
def get_summary_prompt(query: str, context: str) -> str:
    """Get formatted summary prompt"""
    return PromptTemplates.format_user_prompt(AnalysisType.SUMMARY, query, context)

def get_detailed_prompt(query: str, context: str) -> str:
    """Get formatted detailed prompt"""
    return PromptTemplates.format_user_prompt(AnalysisType.DEEP_THINKING, query, context)

def get_global_search_prompt(query: str, context: str) -> str:
    """Get formatted global search prompt"""
    return PromptTemplates.GLOBAL_SEARCH_PROMPT.format(query=query, context=context)

def get_url_analysis_prompt(query: str, context: str) -> str:
    """Get formatted URL analysis prompt"""
    return PromptTemplates.URL_ANALYSIS_PROMPT.format(query=query, context=context)

def get_google_search_prompt(query: str, context: str) -> str:
    """Get formatted Google search prompt"""
    return PromptTemplates.GOOGLE_SEARCH_PROMPT.format(query=query, context=context)
