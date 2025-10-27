import os
from langchain_openai import OpenAI
from sqlalchemy import create_engine, text
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Import settings from the supabase_utils file
from supabase_utils import DATABASE_URL, OPENAI_API_KEY

# Global agent_executor variable
agent_executor = None

def get_agent_executor():
    """
    Initializes and returns the LangChain SQL Agent Executor.
    Caches the agent in a global variable to avoid re-initialization
    on every API request.
    """
    global agent_executor
    if agent_executor is not None:
        return agent_executor

    # Check for missing environment variables
    if not DATABASE_URL:
        print("Error: DATABASE_URL environment variable not set.")
        return None
    
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return None

    try:
        # Create the SQLAlchemy engine
        print("Connecting to database...")
        engine = create_engine(DATABASE_URL)
        
        # Test connection
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Database connection successful.")
        
        # Initialize the SQLDatabase utility
        db = SQLDatabase(engine)
        
        # Initialize the OpenAI LLM
        llm = OpenAI(temperature=0, verbose=True, openai_api_key=OPENAI_API_KEY)
        
        # Initialize the SQLDatabaseToolkit
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        # Create the SQL Agent Executor
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  <-- THIS LINE IS REMOVED
            handle_parsing_errors=True, # Add this to make the agent more robust
                  prefix="""
            You are an AI agent designed to interact with a PostgreSQL database.
            Given an input question, first create a syntactically correct PostgreSQL query to run, then run the query and return the answer.
            You have permissions to perform CREATE, READ, UPDATE, and DELETE operations.
            
            When asked to 'find' or 'list' users, query the 'employee_profiles' table which has rich data.
            The 'profiles' table has basic user info linked to auth.
            
            Example queries:
            - "Find employees with 'React' skill": SELECT * FROM employee_profiles WHERE 'React' = ANY(skills);
            - "Add a new task": INSERT INTO tasks (title, description, created_by, ...) VALUES (...);
            - "Update task 123": UPDATE tasks SET status = 'completed' WHERE id = '123';
            
            NEVER query for all columns from a table. You must query only the relevant columns.
            You MUST wrap all SQL table and column names in double quotes ("") to be case-sensitive and avoid errors with PostgreSQL.
            
            For example:
            Query: "List the full names of all users"
            SQL Query: SELECT "full_name" FROM "profiles";
            
            Query: "What are the skills of Sarah Johnson?"
            SQL Query: SELECT "skills" FROM "employee_profiles" WHERE "user_id" IN (SELECT "id" FROM "profiles" WHERE "full_name" = 'Sarah Johnson');

            If you are asked to delete something, you must confirm the exact query to run.
            If you get an error, try to fix the query and run it again.

            
            ### Final Answer Instructions ###
            After you run the query and get the results, you *must* analyze the results and provide a clear, natural language answer.
            
            1.  *Do not return raw data, tuples, or Python objects.*
            2.  *If the user asks to 'list', 'show', or 'find' items (like employees, tasks, etc.), you MUST present the key information for each item found.*
            3.  *Format the list clearly.* A bulleted list is best. Do not just say "I have the items" or "There are 4 employees." You must show the items.
            
            *Good Answer Example (User: "List all employees"):*
            "I found 4 employees:
            * *Sarah Johnson:* Senior Developer
            * *Mark Lee:* Project Manager
            * *Jane Doe:* UI/UX Designer
            * *Alex Smith:* DevOps Engineer"
            
            *Good Answer Example (User: "Show me high priority tasks"):*
            "Here are the high-priority tasks:
            * *Build E-commerce Product Catalog:* (Status: ongoing)
            * *Migrate to AWS:* (Status: pending)"
            
            If the query returns no results, just say so (e.g., "I couldn't find any high-priority tasks.").
            """
        )
        
        print("LangChain SQL Agent initialized successfully.")
        return agent_executor

    except Exception as e:
        print(f"Error connecting to the database or initializing the agent.")
        print(f"Details: {e}")
        return None

# Initialize the agent on startup
agent_executor = get_agent_executor()