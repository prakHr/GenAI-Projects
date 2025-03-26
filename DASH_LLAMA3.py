import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import ollama
import dash_mantine_components as dmc

# Simple chatbot logic
def chatbot_response(prompt, chat_history2):
    stream = ollama.chat(model="llama3", messages=chat_history2 + [
        {'role': 'user', 'content': prompt},
    ])
    return stream['message']['content']  # Ensure correct key access



# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("ENIQ Chatbot"),
    dcc.Textarea(id='user-input', placeholder='Type your message here...', style={'width': '100%', 'height': '100px'}),
    html.Button('Send', id='send-button', n_clicks=0, className='btn btn-primary mt-2'),
    html.Div(id='chat-history', children=[], style={'marginTop': '20px'}),
    dcc.Store(id='chat-history2', data=[{'role': 'system', 'content': 'Hi!'}]),  # Use dcc.Store to store chat history
    # Chat history inside a carousel
    dbc.Carousel(
        id = "chatter-history",
        items=[],
        controls=True,       # Show previous/next buttons
        indicators=True,     # Show dot indicators
        interval=3000,       # Auto-slide every 3 seconds
        ride="carousel",
        style={"width": "80%", "margin": "auto", "marginTop": "50px"}
    )
])



@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-history2', 'data'),
     Output('chatter-history','items')],  # Ensure chat-history2 is updated properly
    Input('send-button', 'n_clicks'),
    State('user-input', 'value'),
    State('chat-history', 'children'),
    State('chat-history2', 'data'),  # Retrieve stored chat history
    State('chatter-history','items')
)
def update_chat(n_clicks, user_message, chat_history, chat_history2,carousel_slides):
    if n_clicks > 0 and user_message:
        bot_reply = chatbot_response(user_message, chat_history2)
        
        chat_history2.append({'role': 'user', 'content': user_message})
        chat_history2.append({'role': 'assistant', 'content': bot_reply})
        # chat_history2.append({'role': 'user', 'content': "I respond to your answer if I know about it, otherwise I simply say 'I don't know!'"})
        chat_history_display = []
        # Convert text list into carousel items
        carousel_slides = []
        for i,msg in enumerate(chat_history2):
            text = f"{msg['content']}"
            if ((msg['role'])=='user'):
                jcomponent = dmc.MantineProvider(
                    dmc.Alert(
                       text,
                       title=f"{msg['role'].capitalize()}",
                       color="violet",
                    )
                )

                chat_history_display.append(jcomponent) 
                # chat_history_display.append(html.Div(f"{msg['role'].capitalize()}: {msg['content']}"))
            else:
                jcomponent = dmc.MantineProvider(
                    dmc.Alert(
                       text,
                       title=f"{msg['role'].capitalize()}",
                       color="blue",
                    )
                )
                chat_history_display.append(jcomponent)
            carousel_slides.append({"key": str(i), "caption": text})
                # chat_history_display.append(html.Div(f"{msg['role'].capitalize()}: {msg['content']}"))
        # Convert list to carousel slides
        # carousel_slides = [dmc.CarouselSlide(child) for child in chat_history_display]
        from pprint import pprint
        pprint(carousel_slides)
        return chat_history_display, chat_history2, carousel_slides  # Update both components

    return chat_history, chat_history2,carousel_slides  # Return current state if no new message

if __name__ == '__main__':
    app.run(debug=True)
