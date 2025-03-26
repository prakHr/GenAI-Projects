import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import ollama
import threading
import queue

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Queue for handling streamed messages
response_queue = queue.Queue()
streaming_done = False  # Flag to check completion of streaming

# Streaming response generator
def chatbot_response_stream(prompt, chat_history2):
    global streaming_done
    streaming_done = False

    def fetch_stream():
        global streaming_done
        response = ollama.chat(model="llama3", messages=chat_history2 + [
            {'role': 'user', 'content': prompt},
        ], stream=True)  # Enable streaming
        
        for chunk in response:
            response_queue.put(chunk['message']['content'])

        streaming_done = True  # Mark completion

    thread = threading.Thread(target=fetch_stream, daemon=True)
    thread.start()

# Wrap the entire layout in MantineProvider
app.layout = dmc.MantineProvider([
    dbc.Container([
        html.H2("ENIQ Chatbot"),
        dcc.Textarea(id='user-input', placeholder='Type your message here...', style={'width': '100%', 'height': '100px'}),
        dbc.Row([
            dbc.Col(html.Button('Send', id='send-button', n_clicks=0, className='btn btn-primary mt-2'), width=6),
            dbc.Col(html.Button('Clear Chat', id='clear-button', n_clicks=0, className='btn btn-danger mt-2'), width=6),
        ]),
        html.Div(id='chat-history', children=[], style={'marginTop': '20px'}),
        dcc.Store(id='chat-history2', data=[{'role': 'system', 'content': 'Hi!'}]),
        dcc.Interval(id="stream-interval", interval=500, n_intervals=0, disabled=True),
    ])
])

@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-history2', 'data'),
     Output("stream-interval", "disabled")], 
    [Input('send-button', 'n_clicks'),
     Input('clear-button', 'n_clicks'),
     Input("stream-interval", "n_intervals")],
    [State('user-input', 'value'),
     State('chat-history', 'children'),
     State('chat-history2', 'data')]
)
def update_chat(n_clicks, clear_clicks, n_intervals, user_message, chat_history, chat_history2):
    global streaming_done

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Clear chat when "Clear Chat" is clicked
    if triggered_id == "clear-button" and clear_clicks > 0:
        while not response_queue.empty():
            response_queue.get()  # Empty the queue
        return [], [{'role': 'system', 'content': 'Hi!'}], True  # Reset history and stop streaming

    # Start chatbot response when "Send" is clicked
    if triggered_id == "send-button" and n_clicks > 0 and user_message:
        chat_history2.append({'role': 'user', 'content': user_message})
        chat_history2.append({'role': 'assistant', 'content': ""})  # Placeholder for streamed response
        chatbot_response_stream(user_message, chat_history2)
        return chat_history, chat_history2, False  # Enable streaming

    # Handle streaming updates
    if triggered_id == "stream-interval":
        new_content = ""
        while not response_queue.empty():
            new_content += response_queue.get()

        if new_content:
            chat_history2[-1]["content"] += new_content  # Append to assistant's message
        
        chat_history_display = [
            dmc.Alert(
                msg["content"], 
                title=msg["role"].capitalize(), 
                color="violet" if msg["role"] == "user" else "blue"
            ) for msg in chat_history2
        ]

        if streaming_done:
            return chat_history_display, chat_history2, True  # Stop streaming
        return chat_history_display, chat_history2, False  # Continue streaming

    return chat_history, chat_history2, True  

if __name__ == '__main__':
    app.run(debug=True)
