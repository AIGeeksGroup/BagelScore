import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# Read data
df = pd.read_csv('/Users/yinshuo/Documents/1y/code/BAGELScore/Edit_1k_0924/1000_results.csv')

# Create Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define application layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Image Editing Quality Evaluation Dashboard", className="text-center my-4"),
            html.P("This dashboard displays the relationship and distribution of EditScore and GPT scores", className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filter Options"),
                dbc.CardBody([
                    html.P("Select Score Range:"),
                    dcc.RangeSlider(
                        id='score-range-slider',
                        min=0,
                        max=1,
                        step=0.05,
                        marks={i/10: str(i/10) for i in range(0, 11)},
                        value=[0, 1]
                    ),
                    html.Div(id='score-range-output', className="mt-2"),
                    
                    html.P("Select Metrics to Display:", className="mt-3"),
                    dcc.Checklist(
                        id='metrics-checklist',
                        options=[
                            {'label': ' EditScore', 'value': 'editscore'},
                            {'label': ' GPT Total Score', 'value': 'gpt_total_score'},
                            {'label': ' Editing Accuracy', 'value': 'gpt_editing_accuracy'},
                            {'label': ' Visual Quality', 'value': 'gpt_visual_quality'},
                            {'label': ' Content Preservation', 'value': 'gpt_content_preservation'},
                            {'label': ' Style Consistency', 'value': 'gpt_style_consistency'},
                            {'label': ' Overall Effect', 'value': 'gpt_overall_effect'}
                        ],
                        value=['editscore', 'gpt_total_score'],
                        inline=True
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("EditScore and GPT Score Scatter Plot"),
                dbc.CardBody([
                    dcc.Graph(id='scatter-plot')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Score Distribution"),
                dbc.CardBody([
                    dcc.Graph(id='distribution-plot')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Correlation Heatmap"),
                dbc.CardBody([
                    dcc.Graph(id='correlation-heatmap')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("EditScore Component Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='components-analysis')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Table"),
                dbc.CardBody([
                    html.Div(id='data-table')
                ])
            ])
        ], width=12)
    ])
], fluid=True)

# Callback functions
@app.callback(
    Output('score-range-output', 'children'),
    Input('score-range-slider', 'value')
)
def update_score_range_output(value):
    return f'Selected range: {value[0]} to {value[1]}'

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('distribution-plot', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('components-analysis', 'figure'),
     Output('data-table', 'children')],
    [Input('score-range-slider', 'value'),
     Input('metrics-checklist', 'value')]
)
def update_graphs(score_range, selected_metrics):
    # Filter data based on score range
    filtered_df = df[(df['gpt_total_score'] >= score_range[0]) & 
                     (df['gpt_total_score'] <= score_range[1])]
    
    # Create scatter plot
    scatter_fig = px.scatter(
        filtered_df, 
        x='editscore', 
        y='gpt_total_score',
        hover_data=['sequence_number', 'image_name'],
        color='image_rls',
        color_continuous_scale='Viridis',
        title='Relationship between EditScore and GPT Score'
    )
    scatter_fig.update_layout(
        xaxis_title='EditScore',
        yaxis_title='GPT Total Score',
        coloraxis_colorbar_title='Image RLS'
    )
    
    # Create distribution plot
    dist_data = []
    for metric in selected_metrics:
        if metric in filtered_df.columns:
            dist_data.append(
                go.Violin(
                    y=filtered_df[metric],
                    name=metric,
                    box_visible=True,
                    meanline_visible=True
                )
            )
    
    dist_fig = go.Figure(data=dist_data)
    dist_fig.update_layout(
        title='Score Distribution',
        yaxis_title='Score',
        xaxis_title='Metrics'
    )
    
    # Create correlation heatmap
    corr_cols = [col for col in [
        'image_rls', 'image_cosine_sim', 'text_similarity', 
        'gpt_editing_accuracy', 'gpt_visual_quality', 
        'gpt_content_preservation', 'gpt_style_consistency', 
        'gpt_overall_effect', 'gpt_total_score', 
        'editscore'
    ] if col in filtered_df.columns]
    
    corr_df = filtered_df[corr_cols].corr()
    
    corr_fig = px.imshow(
        corr_df,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title='Correlation Analysis of Evaluation Metrics'
    )
    corr_fig.update_layout(
        height=600
    )
    
    # Create EditScore component analysis
    components_fig = make_subplots(rows=1, cols=3, subplot_titles=(
        'Image RLS vs EditScore', 'Image Cosine Sim vs EditScore', 'Text Similarity vs EditScore'
    ))
    
    components_fig.add_trace(
        go.Scatter(
            x=filtered_df['image_rls'],
            y=filtered_df['editscore'],
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name='Image RLS'
        ),
        row=1, col=1
    )
    
    components_fig.add_trace(
        go.Scatter(
            x=filtered_df['image_cosine_sim'],
            y=filtered_df['editscore'],
            mode='markers',
            marker=dict(color='green', opacity=0.6),
            name='Image Cosine Sim'
        ),
        row=1, col=2
    )
    
    components_fig.add_trace(
        go.Scatter(
            x=filtered_df['text_similarity'],
            y=filtered_df['editscore'],
            mode='markers',
            marker=dict(color='red', opacity=0.6),
            name='Text Similarity'
        ),
        row=1, col=3
    )
    
    components_fig.update_layout(
        height=400,
        title_text='EditScore Component Analysis',
        showlegend=False
    )
    
    # Create data table
    table_cols = ['sequence_number', 'image_name', 'editscore', 'gpt_total_score'] + [
        col for col in selected_metrics if col not in ['editscore', 'gpt_total_score']
    ]
    
    table_data = filtered_df.sort_values(by='gpt_total_score', ascending=False).head(10)[
        [col for col in table_cols if col in filtered_df.columns]
    ]
    
    table = dbc.Table.from_dataframe(
        table_data, 
        striped=True, 
        bordered=True, 
        hover=True,
        responsive=True,
        className="mt-3"
    )
    
    return scatter_fig, dist_fig, corr_fig, components_fig, table

# Run the application
if __name__ == '__main__':
    print("Starting interactive dashboard...")
    print("Access the dashboard at http://127.0.0.1:8050/")
    app.run(debug=True, port=8050)