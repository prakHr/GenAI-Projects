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
import re
# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Queue for handling streamed messages
response_queue = queue.Queue()
streaming_done = False  # Flag to check completion of streaming

# Available models
available_models = ["llama3", "llama3.1", "llama3.2","gemma3:1b" ]



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


def ask_normal_dataframe(question,df,model_name):
    # Initialize Local LLM
    llm = Ollama(model=model_name)

    prompt_template = PromptTemplate(
        input_variables=["question", "dataframe"],
        template="You are an AI assistant analyzing a question. Here is the dataset:\n{dataframe}\n\nQuestion: is this {question} related to dataset\n. Answer in one word either only yes or no"
    )
    query = prompt_template.format(question=question, dataframe=df.to_string())
    return llm.invoke(query)


def ask_dataframe2(question, df,model_name):
    # Initialize Local LLM
    llm = Ollama(model=model_name)

    # Create a prompt template for natural language queries
    prompt_template = PromptTemplate(
        input_variables=["question", "dataframe"],
        template="You are an AI assistant analyzing a dataset. Here is the dataset:\n{dataframe}\n\nQuestion: {question}\nAnswer:"
    )
    query = prompt_template.format(question=question, dataframe=df.to_string())
    rv = llm.invoke(query)
    # print(f"rv = {rv}")
    return rv 

def ask_dataframe(question, df, model_name):
    yes_or_no = ask_normal_dataframe(question,df,model_name)
    print(f"yes_or_no = {yes_or_no}")
    if 'no' in yes_or_no.lower():
        return []

    llm = Ollama(model=model_name)

    # Prompt Template for Selecting Multiple Plot Types
    prompt_template = PromptTemplate(
        input_variables=["question", "dataframe"],
        template="""
        You are a data visualization expert. Given the following DataFrame:
        {dataframe}

        Identify **up to seven** relevant Plotly chart types to best answer this question:
        "{question}"

        Choose from: 'bar', 'line', 'scatter', 'pie', 'histogram', 'box', 'heatmap'.

        - Select appropriate columns for the x-axis and y-axis (or category for pie charts).
        - Ensure the selections are meaningful based on the data types and question.

        Respond in **strict JSON format only**, with no additional text:
        [
            {{"chart_type": "bar", "x": "<column_name>", "y": "<column_name>" }},
            {{"chart_type": "pie", "x": "<column_name>", "y": "<column_name>" }}
        ]
        """
    )


    query = prompt_template.format(question=question, dataframe=df.head().to_string())
    response = llm.invoke(query)

    # Debugging: Print raw response
    # print("Raw response:", response)

    try:
        import re
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            response = match.group(0)

        chart_info_list = json.loads(response)  # Convert JSON string to Python list

    except json.JSONDecodeError as e:
        # print("Error parsing JSON:", e)
        return []
    # from pprint import pprint
    # pprint(chart_info_list)

    figures = []
    for chart_info in chart_info_list:
        chart_type = chart_info.get("chart_type", "").strip().lower()
        x_col = chart_info.get("x", None)
        y_col = chart_info.get("y", None)
        # import re

        # def extract_valid_column(column_name, df):
            # import re
            # What is the correlation between petal length and sepal width?
            # match = re.findall(r'\b\w+\b', column_name)  # Extract individual words
            # valid_columns = [col for col in match if col in df.columns]  # Keep only valid columns
            # return valid_columns[0] if valid_columns else df.columns[0]  # Return first valid column or fallback
            # return valid_columns
        # y_cols = extract_valid_column(y_col, df)
        # x_cols = extract_valid_column(x_col, df)
        # print(f"x_cols={x_cols}")
        # print(f"y_cols={y_cols}")

        # for y_col in y_cols:
            # for x_col in x_cols:
        fig = None
        if chart_type == "bar":
            try:
                fig = px.bar(df, x=x_col, y=y_col, title=f"{question} - {chart_type}")
            except:
                fig = None
        elif chart_type == "line":
            try:
                fig = px.line(df, x=x_col, y=y_col, title=f"{question} - {chart_type}")
            except:
                fig = None
        elif chart_type == "scatter":
            try:
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{question} - {chart_type}")
            except:
                fig = None
        elif chart_type == "pie":
            try:
                fig = px.pie(df, names=x_col, values=y_col, title=f"{question} - {chart_type}")
            except:
                fig = None
        elif chart_type == "histogram":
            try:
                fig = px.histogram(df, x=x_col, title=f"{question} - {chart_type}")
            except:
                fig = None
        elif chart_type == "box":
            try:
                fig = px.box(df, y=y_col, title=f"{question} - {chart_type}")
            except:
                fig = None
        elif chart_type == "heatmap":
            try:
                fig = px.imshow(df.corr(), title=f"{question} - {chart_type}")
            except:
                fig = None
        if fig:
            figures.append(fig)

    return figures


# Layout
app.layout = dmc.MantineProvider(
    theme={
        "colorScheme": "light",  # Ensure light mode is applied
    },
    children=[
        html.Div(
            style={
                "backgroundImage": "url('/assets/e.webp')",
                "backgroundSize": "cover",
                "backgroundPosition": "center",
                "height": "100%",  # Ensure it covers the full page
                "width": "100%",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "padding": "20px",
            },
            children=[
                dbc.Container([
                    html.H2("ENIQ Chatbot"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button('Upload CSV', className='btn btn-primary mb-2'),
                        multiple=False
                    ),
                    html.Button('Clear File', id='clear-file', className='btn btn-warning mb-2', style={'display': 'None'}),
                    dcc.Store(id='stored-data', data=None),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[{'label': model, 'value': model} for model in available_models],
                        value="gemma3:1b",
                        clearable=False,
                    ),
                    # dcc.Textarea(id='user-input', placeholder='Type your query here...', style={'width': '100%', 'height': '100px'}),
                    dmc.Textarea(
                        id='user-input',
                        placeholder='Type your query here...',
                        autosize=True,
                        minRows=3,
                        maxRows=6,
                        style={'width': '100%'}
                    ),
                    dbc.Row([
                        dbc.Col(html.Button('Send', id='send-button', n_clicks=0, className='btn btn-primary mt-2'), width=6),
                        dbc.Col(html.Button('Clear Chat', id='clear-button', n_clicks=0, className='btn btn-secondary mt-2'), width=4),
                    ]),
                    html.Div(id='chat-history', children=[], style={'marginTop': '20px'}),
                    html.Div(id='data-charts'),
                    dcc.Store(id='chat-history2', data=[{'role': 'system', 'content': 'Hi!'}]),
                    dcc.Interval(id="stream-interval", interval=500, n_intervals=0, disabled=True),
                ], fluid=True, style={"backgroundColor": "rgba(255, 255, 255, 0.8)", "padding": "20px", "borderRadius": "10px"})
            ]
        )
    ]
)


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
        # print(f"df = {df}")
        return df.to_json()  # Store as JSON
    return None

@app.callback(
    [Output('stored-data', 'data', allow_duplicate=True),
    Output('upload-data', 'contents', allow_duplicate=True)],
    [Input('clear-file', 'n_clicks'),
    Input('upload-data', 'contents')],
    prevent_initial_call=True
)
def clear_file(n_clicks,contents):
    triggered_id = ctx.triggered_id
    
    # if triggered_id == "clear-file" and n_clicks > 0:
    # print("called here")
    # print(f"contents = {contents}")
    # print("*"*100)
    return n_clicks,contents  # Clear stored data
    # return None,None

@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-history2', 'data'),
     Output("stream-interval", "disabled"),
     Output('data-charts', 'children'),
     Output('upload-data', 'contents')],  # Update multiple charts
    [Input('send-button', 'n_clicks'),
     Input('clear-button', 'n_clicks'),
     Input("stream-interval", "n_intervals")],
    [State('user-input', 'value'),
     State('chat-history', 'children'),
     State('chat-history2', 'data'),
     State('model-dropdown', 'value'),
     State('stored-data', 'data'),
     State('data-charts', 'children'),
     State('upload-data', 'contents')]  # Preserve previous charts
)
def update_chat(n_clicks, clear_clicks, n_intervals, user_message, chat_history, chat_history2, model_name, stored_data, prev_charts,contents):
    global streaming_done
    
    triggered_id = ctx.triggered_id

    # Handle clear chat
    if triggered_id == "clear-button" and clear_clicks > 0:
        while not response_queue.empty():
            response_queue.get()
        return [], [{'role': 'system', 'content': 'Hi!'}], True, [],None  # Clear charts

    # Handle new user message
    if triggered_id == "send-button" and n_clicks > 0 and user_message:
        chat_history2.append({'role': 'user', 'content': user_message})
        chat_history2.append({'role': 'assistant', 'content': ""})  # Placeholder

        chatbot_response_stream(user_message, chat_history2, model_name)
        
        df = pd.read_json(stored_data) if stored_data else None
        charts = prev_charts  # Preserve previous charts

        if df is not None:
            answer = ask_dataframe2(user_message, df, model_name)
            chat_history2[-1]['content'] += answer
            new_figs = ask_dataframe(user_message, df, model_name)

            # Convert figures to dcc.Graph components
            new_graphs = [dcc.Graph(figure=fig, style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}) for fig in new_figs]
            prev_charts = prev_charts if prev_charts is not None else []
            charts = new_graphs + prev_charts  # Append new charts instead of replacing

        return chat_history, chat_history2, False, charts,contents  # Keep streaming enabled

    # Handle streaming updates
    if triggered_id == "stream-interval":
        new_content = ""
        while not response_queue.empty():
            new_content += response_queue.get()
        
        if new_content:
            chat_history2[-1]["content"] += new_content  # Append to last assistant message

        # chat_history_display = [
        #     dmc.Alert(msg["content"], title=msg["role"].capitalize(), color="violet" if msg["role"] == "user" else "blue")
        #     for msg in chat_history2
        # ]
        chat_history_display = []
        for msg in chat_history2:
            if msg["role"] == "assistant":
            # if "```" in msg["content"]:  # Check if it's a code block
                chat_history_display.append(dcc.Markdown(msg["content"], style={"white-space": "pre-wrap"}))
            else:
                chat_history_display.append(dmc.Alert(msg["content"], title=msg["role"].capitalize(), color="blue"))

        return chat_history_display, chat_history2, streaming_done, prev_charts,contents  # Preserve charts

    return chat_history, chat_history2, True, prev_charts,contents  # Preserve charts on no update



if __name__ == '__main__':
    app.run(debug=True)
