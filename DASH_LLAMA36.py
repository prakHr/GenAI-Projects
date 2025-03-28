import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import pandas as pd
import io
import ollama
import threading
import queue
import json
import plotly.express as px
import base64
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Queue for handling streamed messages
response_queue = queue.Queue()
streaming_done = False  # Flag to check completion of streaming

# Available models
available_models = ["llama3", "llama3.1", "llama3.2"]

# Initialize Local LLM
llm = Ollama(model="llama3")


import json
import re

def chatbot_response_stream(prompt, chat_history2, model_name):
    global streaming_done
    streaming_done = False

    def fetch_stream():
        global streaming_done
        response = ollama.chat(model=model_name, messages=chat_history2 + [
            {'role': 'user', 'content': prompt},
        ], stream=True)  # Enable streaming
        
        for chunk in response:
            response_queue.put(chunk['message']['content'])
        
        streaming_done = True  # Mark completion

    thread = threading.Thread(target=fetch_stream, daemon=True)
    thread.start()


def ask_dataframe2(question, df):
    # Create a prompt template for natural language queries
    prompt_template = PromptTemplate(
        input_variables=["question", "dataframe"],
        template="You are an AI assistant analyzing a dataset. Here is the dataset:\n{dataframe}\n\nQuestion: {question}\nAnswer:"
    )
    query = prompt_template.format(question=question, dataframe=df.to_string())
    rv = llm(query)
    print(f"rv = {rv}")
    return rv 

def ask_dataframe(question,df):
    # Prompt Template for Selecting the Best Plot Type
    prompt_template = PromptTemplate(
        input_variables=["question", "dataframe"],
        template="""
        You are a data visualization expert. Given the following DataFrame:
        {dataframe}

        Identify the most suitable Plotly chart type to answer this question.
        Choose from: 'bar', 'line', 'scatter', 'pie', 'histogram', 'box', 'heatmap'.

        Also, suggest the best x-axis and y-axis columns (or category for pie charts).

        Respond in **strict JSON format only**, with no additional text:
        {{"chart_type": "bar", "x": "Region", "y": "Sales"}}
        """
    )

    query = prompt_template.format(question=question, dataframe=df.head().to_string())  # Limit rows for clarity
    response = llm(query)

    # Debugging: Print the raw response to inspect it
    print("Raw response:", response)

    try:
        # Extract JSON using regex (if extra text is present)
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            response = match.group(0)  # Get only the JSON part

        chart_info = json.loads(response)  # Convert JSON string to Python dict

    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return

    # Extracting chart type and columns
    chart_type = chart_info.get("chart_type", "").strip().lower()
    x_col = chart_info.get("x", None)
    y_col = chart_info.get("y", None)

    # Generate Plotly Chart Based on Suggested Type
    fig = None
    if chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, title=question)
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title=question)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, title=question)
    elif chart_type == "pie":
        fig = px.pie(df, names=x_col, values=y_col, title=question)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_col, title=question)
    elif chart_type == "box":
        fig = px.box(df, y=y_col, title=question)
    elif chart_type == "heatmap":
        fig = px.imshow(df.corr(), title=question)
    else:
        print(f"Could not determine a suitable chart type for: {question}")
        return

    return fig


# Layout
app.layout = dmc.MantineProvider([
    dbc.Container([
        html.H2("ENIQ Chatbot"),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload CSV', className='btn btn-info mb-2'),
            multiple=False
        ),
        html.Button('Clear File', id='clear-file', className='btn btn-warning mb-2'),
        dcc.Store(id='stored-data', data=None),  # Store DataFrame in memory
        
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': model, 'value': model} for model in available_models],
            value='llama3',
            clearable=False
        ),
        dcc.Textarea(id='user-input', placeholder='Type your message here...', style={'width': '100%', 'height': '100px'}),
        dbc.Row([
            dbc.Col(html.Button('Send', id='send-button', n_clicks=0, className='btn btn-primary mt-2'), width=6),
            dbc.Col(html.Button('Clear Chat', id='clear-button', n_clicks=0, className='btn btn-danger mt-2'), width=6),
        ]),
        html.Div(id='chat-history', children=[], style={'marginTop': '20px'}),
        dcc.Graph(id='data-chart'),
        dcc.Store(id='chat-history2', data=[{'role': 'system', 'content': 'Hi!'}]),
        dcc.Interval(id="stream-interval", interval=500, n_intervals=0, disabled=True),
    ])
])

@app.callback(
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def store_uploaded_data(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = io.BytesIO(base64.b64decode(content_string))
        df = pd.read_csv(decoded)
        return df.to_json()  # Store as JSON
    return None

@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    Input('clear-file', 'n_clicks'),
    prevent_initial_call=True
)
def clear_file(n_clicks):
    triggered_id = ctx.triggered_id
    
    if triggered_id == "clear-file" and n_clicks > 0:
        return None  # Clear stored data

@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-history2', 'data'),
     Output("stream-interval", "disabled"),
     Output('data-chart', 'figure')], 
    [Input('send-button', 'n_clicks'),
     Input('clear-button', 'n_clicks'),
     Input("stream-interval", "n_intervals")],
    [State('user-input', 'value'),
     State('chat-history', 'children'),
     State('chat-history2', 'data'),
     State('model-dropdown', 'value'),
     State('stored-data', 'data'),
     State('data-chart', 'figure')]  # Add previous figure to prevent disappearing charts
)
def update_chat(n_clicks, clear_clicks, n_intervals, user_message, chat_history, chat_history2, model_name, stored_data, prev_fig):
    global streaming_done
    
    triggered_id = ctx.triggered_id
    
    # Handle clear chat
    if triggered_id == "clear-button" and clear_clicks > 0:
        while not response_queue.empty():
            response_queue.get()
        return [], [{'role': 'system', 'content': 'Hi!'}], True, {}

    # Handle new user message
    if triggered_id == "send-button" and n_clicks > 0 and user_message:
        chat_history2.append({'role': 'user', 'content': user_message})
        chat_history2.append({'role': 'assistant', 'content': ""})  # Placeholder for streamed response
        chatbot_response_stream(user_message, chat_history2, model_name)
        
        df = pd.read_json(stored_data) if stored_data else None
        fig = prev_fig  # Preserve previous chart unless overwritten
        
        if df is not None:
            answer = ask_dataframe2(user_message, df)
            chat_history2[-1]['content'] += answer  # Append instead of overwriting
            new_fig = ask_dataframe(user_message, df)
            if new_fig:  # Only update if a new chart is generated
                fig = new_fig
        
        return chat_history, chat_history2, False, fig  # Keep streaming enabled

    # Handle streaming updates
    if triggered_id == "stream-interval":
        new_content = ""
        while not response_queue.empty():
            new_content += response_queue.get()
        
        if new_content:
            chat_history2[-1]["content"] += new_content  # Append to last assistant message
        
        chat_history_display = [
            dmc.Alert(msg["content"], title=msg["role"].capitalize(), color="violet" if msg["role"] == "user" else "blue")
            for msg in chat_history2
        ]
        
        return chat_history_display, chat_history2, streaming_done, prev_fig  # Preserve the existing chart
    
    return chat_history, chat_history2, True, prev_fig  # Preserve the chart on no update


if __name__ == '__main__':
    app.run(debug=True)
