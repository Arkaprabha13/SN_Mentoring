import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from together import Together
import io
import base64
from PIL import Image
import PyPDF2
import docx
import json
import os
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalystAgent:
    def __init__(self, api_key: str):
        """Initialize the Data Analyst Agent with Together AI client."""
        self.client = Together(api_key=api_key)
        self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.conversation_history = []
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for API consumption."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return f"Error reading DOCX: {str(e)}"
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive analysis of DataFrame."""
        try:
            analysis = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_summary": {},
                "categorical_summary": {},
                "sample_data": df.head().to_dict('records')
            }
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            # Categorical columns analysis
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    analysis["categorical_summary"][col] = {
                        "unique_count": df[col].nunique(),
                        "top_values": df[col].value_counts().head().to_dict()
                    }
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing dataframe: {e}")
            return {"error": str(e)}
    
    def generate_visualization_code(self, df: pd.DataFrame, chart_type: str, columns: List[str]) -> str:
        """Generate Python code for creating visualizations."""
        code_templates = {
            "histogram": f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
plt.hist(df['{columns[0]}'], bins=30, alpha=0.7, edgecolor='black')
plt.title('Histogram of {columns[0]}')
plt.xlabel('{columns[0]}')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()
""",
            "scatter": f"""
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['{columns[0]}'], df['{columns[1]}'], alpha=0.6)
plt.title('Scatter Plot: {columns[0]} vs {columns[1]}')
plt.xlabel('{columns[0]}')
plt.ylabel('{columns[1]}')
plt.grid(True, alpha=0.3)
plt.show()
""",
            "bar": f"""
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
value_counts = df['{columns[0]}'].value_counts().head(10)
plt.bar(value_counts.index, value_counts.values)
plt.title('Bar Chart of {columns[0]}')
plt.xlabel('{columns[0]}')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
""",
            "correlation": """
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
"""
        }
        return code_templates.get(chart_type, "# Chart type not supported")
    
    def create_visualization(self, df: pd.DataFrame, chart_type: str, columns: List[str]):
        """Create and return visualization based on chart type."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "histogram" and len(columns) >= 1:
                ax.hist(df[columns[0]].dropna(), bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'Histogram of {columns[0]}')
                ax.set_xlabel(columns[0])
                ax.set_ylabel('Frequency')
                
            elif chart_type == "scatter" and len(columns) >= 2:
                ax.scatter(df[columns[0]], df[columns[1]], alpha=0.6)
                ax.set_title(f'Scatter Plot: {columns[0]} vs {columns[1]}')
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])
                
            elif chart_type == "bar" and len(columns) >= 1:
                value_counts = df[columns[0]].value_counts().head(10)
                ax.bar(value_counts.index, value_counts.values)
                ax.set_title(f'Bar Chart of {columns[0]}')
                ax.set_xlabel(columns[0])
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                
            elif chart_type == "correlation":
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    correlation_matrix = numeric_df.corr()
                    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                    ax.set_xticks(range(len(correlation_matrix.columns)))
                    ax.set_yticks(range(len(correlation_matrix.columns)))
                    ax.set_xticklabels(correlation_matrix.columns, rotation=45)
                    ax.set_yticklabels(correlation_matrix.columns)
                    plt.colorbar(im)
                    ax.set_title('Correlation Heatmap')
                else:
                    ax.text(0.5, 0.5, 'Not enough numeric columns for correlation', 
                           ha='center', va='center', transform=ax.transAxes)
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'Error creating chart: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def query_llm(self, messages: List[Dict], stream: bool = True) -> str:
        """Query the Llama 4 Maverick model."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                max_tokens=2048,
                temperature=0.7
            )
            
            if stream:
                full_response = ""
                for token in response:
                    if hasattr(token, 'choices') and len(token.choices) > 0:
                        if hasattr(token.choices[0], 'delta') and hasattr(token.choices[0].delta, 'content'):
                            if token.choices[0].delta.content:
                                full_response += token.choices[0].delta.content
                return full_response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return f"Error communicating with AI model: {str(e)}"
    
    def analyze_with_ai(self, data_summary: str, user_question: str = None) -> str:
        """Use AI to analyze data and answer questions."""
        system_prompt = """You are an expert data analyst with deep knowledge of statistics, data science, and business intelligence. 
        You help users understand their data by providing insights, identifying patterns, and suggesting actionable recommendations.
        
        When analyzing data:
        1. Provide clear, actionable insights
        2. Identify interesting patterns or anomalies
        3. Suggest relevant visualizations
        4. Recommend next steps for analysis
        5. Be specific about what the data shows
        
        Always structure your response with clear sections and bullet points for readability."""
        
        if user_question:
            user_prompt = f"""
            Data Summary:
            {data_summary}
            
            User Question: {user_question}
            
            Please analyze this data and answer the user's specific question. Provide detailed insights and recommendations.
            """
        else:
            user_prompt = f"""
            Data Summary:
            {data_summary}
            
            Please provide a comprehensive analysis of this dataset. Include:
            1. Key insights and patterns
            2. Data quality observations
            3. Suggested visualizations
            4. Recommendations for further analysis
            """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.query_llm(messages)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'data_analysis' not in st.session_state:
        st.session_state.data_analysis = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def main():
    st.set_page_config(
        page_title="AI Data Analyst Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤖 AI Data Analyst Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Llama 4 Maverick** - Upload any document and get intelligent data analysis")
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("🔧 Configuration")
        
        api_key = st.text_input(
            "Together AI API Key",
            type="password",
            help="Enter your Together AI API key to use Llama 4 Maverick"
        )
        
        if api_key:
            if st.session_state.agent is None:
                st.session_state.agent = DataAnalystAgent(api_key)
                st.success("✅ Agent initialized!")
            
            st.header("📋 Analysis Options")
            analysis_depth = st.selectbox(
                "Analysis Depth",
                ["Quick Overview", "Detailed Analysis", "Deep Dive"],
                help="Choose the depth of analysis"
            )
            
            auto_visualize = st.checkbox(
                "Auto-generate visualizations",
                value=True,
                help="Automatically create relevant charts"
            )
        else:
            st.warning("⚠️ Please enter your Together AI API key to continue")
            st.info("Get your free API key at: https://www.together.ai/")
    
    if not api_key:
        st.error("🔑 API key required to proceed")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="section-header">📁 File Upload</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=['csv', 'xlsx', 'xls', 'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'],
            help="Supported formats: CSV, Excel, Text, PDF, Word, Images"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Processing file..."):
                try:
                    file_details = {
                        "filename": uploaded_file.name,
                        "filetype": uploaded_file.type,
                        "filesize": uploaded_file.size
                    }
                    
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
                    st.json(file_details)
                    
                    # Process different file types
                    if uploaded_file.type == "text/csv":
                        df = pd.read_csv(uploaded_file)
                        st.session_state.uploaded_data = df
                        
                    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                        df = pd.read_excel(uploaded_file)
                        st.session_state.uploaded_data = df
                        
                    elif uploaded_file.type == "text/plain":
                        text_content = str(uploaded_file.read(), "utf-8")
                        st.session_state.uploaded_data = {"text_content": text_content}
                        
                    elif uploaded_file.type == "application/pdf":
                        text_content = st.session_state.agent.extract_text_from_pdf(uploaded_file)
                        st.session_state.uploaded_data = {"text_content": text_content}
                        
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text_content = st.session_state.agent.extract_text_from_docx(uploaded_file)
                        st.session_state.uploaded_data = {"text_content": text_content}
                        
                    elif uploaded_file.type.startswith('image/'):
                        image = Image.open(uploaded_file)
                        st.session_state.uploaded_data = {"image": image}
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    logger.error(f"File processing error: {traceback.format_exc()}")
    
    with col2:
        if st.session_state.uploaded_data is not None:
            st.markdown('<div class="section-header">📊 Data Analysis</div>', unsafe_allow_html=True)
            
            # Analyze uploaded data
            if isinstance(st.session_state.uploaded_data, pd.DataFrame):
                df = st.session_state.uploaded_data
                
                # Display basic info
                col2_1, col2_2, col2_3 = st.columns(3)
                with col2_1:
                    st.metric("Rows", df.shape[0])
                with col2_2:
                    st.metric("Columns", df.shape[1])
                with col2_3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Data preview
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Generate analysis
                if st.button("🔍 Analyze Data", type="primary"):
                    with st.spinner("🤖 AI is analyzing your data..."):
                        analysis = st.session_state.agent.analyze_dataframe(df)
                        st.session_state.data_analysis = analysis
                        
                        # Create summary for AI
                        summary = f"""
                        Dataset Analysis:
                        - Shape: {analysis['shape']} (rows, columns)
                        - Columns: {', '.join(analysis['columns'])}
                        - Data Types: {analysis['dtypes']}
                        - Missing Values: {analysis['missing_values']}
                        - Numeric Summary: {analysis.get('numeric_summary', 'No numeric columns')}
                        - Categorical Summary: {analysis.get('categorical_summary', 'No categorical columns')}
                        """
                        
                        ai_insights = st.session_state.agent.analyze_with_ai(summary)
                        
                        st.markdown("### 🤖 AI Insights")
                        st.markdown(ai_insights)
                
                # Visualization section
                if auto_visualize and st.session_state.data_analysis:
                    st.markdown("### 📈 Visualizations")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if len(numeric_cols) > 0:
                        with viz_col1:
                            st.subheader("Distribution")
                            fig = st.session_state.agent.create_visualization(df, "histogram", [numeric_cols[0]])
                            st.pyplot(fig)
                    
                    if len(numeric_cols) > 1:
                        with viz_col2:
                            st.subheader("Correlation")
                            fig = st.session_state.agent.create_visualization(df, "correlation", [])
                            st.pyplot(fig)
                    
                    if len(categorical_cols) > 0:
                        st.subheader("Category Distribution")
                        fig = st.session_state.agent.create_visualization(df, "bar", [categorical_cols[0]])
                        st.pyplot(fig)
            
            elif isinstance(st.session_state.uploaded_data, dict):
                # Handle text or image content
                if "text_content" in st.session_state.uploaded_data:
                    text_content = st.session_state.uploaded_data["text_content"]
                    
                    st.text_area("📄 Document Content", text_content[:1000] + "..." if len(text_content) > 1000 else text_content, height=200)
                    
                    if st.button("🔍 Analyze Document", type="primary"):
                        with st.spinner("🤖 AI is analyzing your document..."):
                            summary = f"Document content (first 2000 chars): {text_content[:2000]}"
                            ai_insights = st.session_state.agent.analyze_with_ai(summary)
                            st.markdown("### 🤖 AI Analysis")
                            st.markdown(ai_insights)
                
                elif "image" in st.session_state.uploaded_data:
                    image = st.session_state.uploaded_data["image"]
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    if st.button("🔍 Analyze Image", type="primary"):
                        with st.spinner("🤖 AI is analyzing your image..."):
                            # For image analysis, we'd need to implement image-to-text functionality
                            # This is a placeholder for image analysis
                            ai_insights = st.session_state.agent.analyze_with_ai("Image uploaded for analysis")
                            st.markdown("### 🤖 AI Analysis")
                            st.markdown(ai_insights)
    
    # Chat interface for follow-up questions
    if st.session_state.uploaded_data is not None:
        st.markdown('<div class="section-header">💬 Ask Questions</div>', unsafe_allow_html=True)
        
        user_question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., What are the main trends in this dataset?"
        )
        
        if st.button("❓ Ask AI", type="secondary") and user_question:
            with st.spinner("🤖 AI is thinking..."):
                if isinstance(st.session_state.uploaded_data, pd.DataFrame):
                    df = st.session_state.uploaded_data
                    analysis = st.session_state.agent.analyze_dataframe(df)
                    summary = f"Dataset with {df.shape[0]} rows and {df.shape[1]} columns. Columns: {list(df.columns)}"
                else:
                    summary = "Uploaded document content"
                
                ai_response = st.session_state.agent.analyze_with_ai(summary, user_question)
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "question": user_question,
                    "answer": ai_response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                st.markdown("### 🤖 AI Response")
                st.markdown(ai_response)
        
        # Display conversation history
        if st.session_state.conversation_history:
            with st.expander("💭 Conversation History"):
                for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                    st.markdown(f"**Q{len(st.session_state.conversation_history)-i}** ({conv['timestamp']}): {conv['question']}")
                    st.markdown(f"**A**: {conv['answer'][:200]}...")
                    st.markdown("---")

if __name__ == "__main__":
    main()
