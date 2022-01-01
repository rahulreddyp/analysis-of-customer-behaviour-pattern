import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from analysis.SVMregressor import *
from analysis.randomforest import *
from plotlyvisualization import *
from dash.dependencies import Output, Input, State


def dashboard():
    pdf = read_dataset(Path('..', 'datasets', 'processed', 'product.csv'))
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ])

    customer_Insight = html.Div([dbc.Container([
        dbc.Row([
            html.H1(children='Akatsuki Analytics'),
            html.H2(children='Customer Insights', style={'text-align': 'center'}),
        ],
            style={'display': 'flex', 'flex-direction': 'column', 'align-content': 'center'}),
        html.Hr(),

        html.Div(html.H4("Pie Chart showing category wise customer distribution")),
        html.Div([
            html.Div([
                dcc.Dropdown(id="dropdown", value='Education',
                             options=[{"label": "Education", "value": 'Education'},
                                      {"label": "Age", "value": 'Age'},
                                      {"label": "Marital_Status", "value": 'Marital_Status'}], style={
                        "width": "170px"}),

            ], style={'display': 'flex', 'align-items': 'center', 'width': '300px', 'justify-content': 'center'}),
            html.Div([dcc.Graph(id='pie')],
                     style={})
        ], style={'display': 'flex', 'flex-direction': 'row', 'padding': '20px', 'border': '1px solid black',
                  'border-radius': '10px', 'background': '#bcc6cc', 'justify-content': 'center'}),
        html.Hr(),
        html.Div(html.H4("Plots summarizing on various customer attributes")),
        html.Div([
            html.Div([
                dbc.Label("Choose X - Customer Category"),
                dcc.Dropdown(id="dropdown1", value='Education',
                             options=[{"label": "Education", "value": 'Education'}, {"label": "Age", "value": 'Age'},
                                      {"label": "Marital_Status", "value": 'Marital_Status'}],
                             style={
                                 "width": "200px"}
                             ),
            ]),
            html.Div([dbc.Label("Choose Y data"),
                      dcc.Dropdown(id="dropdown2", value='numofvisits',
                                   options=[{"label": "Amount spent", "value": 'amountspent'},
                                            {"label": "Number of visit", "value": 'numofvisits'},
                                            {"label": "Total Number of Purchaces", "value": 'totalpurchaces'},
                                            {"label": "Income", "value": "Income"}],
                                   style={
                                       "width": "200px"}
                                   )]),
        ], style={'display': 'flex',
                  'flex-direction': 'row',
                  'justify-content': 'space-evenly', 'padding-bottom': '20px'}),
        html.Div([dcc.RadioItems(id='plottype',
                                 options=[
                                     {'label': '  Bar plot' + "\t", 'value': 'bp'},
                                     {'label': '  Line plot' + "\t", 'value': 'lp'},
                                     {'label': '  Composite line bar', 'value': 'clb'}
                                 ],
                                 value='bp',
                                 labelStyle={'display': 'inline-block', 'margin-right': '15px', 'font-weight': 300},
                                 style={'display': 'inline-block', 'margin-left': '7px'}
                                 )], style={'text-align': 'center', 'padding-bottom': '10px'}),
        html.Div([
            dcc.Graph(id='barline', style={'width': '530px'}),
            dcc.Graph(id='table', style={'width': '530px'}),
        ], style={'display': 'flex', 'justify-content': 'space-around', 'padding': '20px', 'border': '1px solid black',
                  'border-radius': '10px', 'background': '#bcc6cc'}),
        html.Hr(),
        html.Div(html.H4("Age Distribution over customer education")),
        html.Div([
            dbc.Label("Select Plot"),
            dcc.Dropdown(id="dropdown3", value=2,
                         options=[{"label": "Iciecle", "value": 1}, {"label": "sunburst", "value": 2}],
                         style={
                             "width": "200px"}
                         ),
        ], style={'padding-top': '10px', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}),
        html.Div([
            dcc.Graph(id='hirar')
        ])])])

    product_insight = html.Div([dbc.Container([
        dbc.Row([
            html.H1(children='Akatsuki Analytics'),
            html.H2(children='Product Insights', style={'text-align': 'center'})
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-content': 'center'}),
        html.Hr(),
        html.Div(html.H4("Average amount spent on each product")),
        html.Div([
            html.Div([
                dbc.Col(dcc.Graph(id='horbar', figure=horizontalbar(pdf))),
            ], style={'display': 'flex', 'flex-direction': 'row', 'padding': '20px', 'border': '1px solid black',
                      'border-radius': '10px', 'background': '#bcc6cc', 'justify-content': 'center'}),

            html.Hr(),
            html.Div(html.H4("Heatmap showing correlation among all products")),
            html.Div([
                dbc.Col(dcc.Graph(id='heatmap', figure=heatmap(pdf))),
            ], style={'display': 'flex', 'flex-direction': 'row', 'padding': '20px', 'border': '1px solid black',
                      'border-radius': '10px', 'background': '#bcc6cc', 'justify-content': 'center'})
        ])
    ]),
    ]),

    purchase_insight = html.Div([dbc.Container([
        dbc.Row([
            html.H1(children='Akatsuki Analytics'),
            html.H2(children='Product Insights', style={'text-align': 'center'})
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-content': 'center'}),
        html.Hr(),
        html.Div(html.H4("Funnel Chart showing average amount spent on each mode of purchase")),
        html.Div([
            dbc.Col(dcc.Graph(id='funnel', figure=funnel())),
        ]),
        html.Div(html.H4("Pie chart showing the percentage of number of purchases made in each category")),
        dbc.Col(dcc.Graph(id='ppie', figure=piemarketing())),
    ])])
    marketing = html.Div([dbc.Container([
        dbc.Row([
            html.H1(children='Akatsuki Analytics'),
            html.H2(children='Campaign Analysis', style={'text-align': 'center'})
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-content': 'center'}),
        html.Hr(),
        html.Div(html.H4("Bar graph showing total responses in each campaign")),
        html.Div(
            dcc.Graph(id='bm', figure=barmarketing())
            , style={'padding': '20px', 'border': '1px solid black',
                     'border-radius': '10px', 'background': '#bcc6cc'}),
        html.Hr(),
    ])])
    analysis = html.Div([dbc.Container([
        dbc.Row([
            html.H1(children='Akatsuki Analytics'),
            html.H2(children='Analysis', style={'text-align': 'center'})
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-content': 'center'}),
        html.Hr(),
        html.Div(html.H4("Analysis showing relation between customer category and products")),
        html.Div([dcc.Dropdown(id="dropdown4", value="Education",
                               options=[{"label": "Education", "value": "Education"},
                                        {"label": "AgeCategory", "value": "AgeCategory"},
                                        {"label": "Marital_Status", "value": "Marital_Status"}]
                               , style={
                "width": "170px"}),

                  ], style={'display': 'flex', 'align-items': 'center', 'width': '300px', 'justify-content': 'center'}),
        html.Div(dcc.RadioItems(id='groupby',
                                options=[
                                    {'label': ' mean', 'value': 'mean'},
                                    {'label': ' max', 'value': 'max'},
                                    {'label': ' min', 'value': 'min'}
                                ],
                                value='mean',
                                labelStyle={'display': 'inline-block', 'margin-right': '15px', 'font-weight': 300},
                                style={'display': 'inline-block', 'margin-left': '7px'}
                                )
                 ),
        dbc.Button('Show Plot', color='primary', id='eb', style={'margin-bottom': '1em'}, block=True),
        html.H3(id='cpa', children='Click on bars for more details', style={'text-align': 'center'}),
        html.Div(
            dcc.Graph(id='sb')
            , style={'padding': '20px', 'border': '1px solid black',
                     'border-radius': '10px', 'background': '#bcc6cc'}),
        html.Hr(),
        html.Div([
            html.H4("Customer Campaign Analysis"),
            dcc.Dropdown(id="dropdown5", value="Education",
                         options=[{"label": "Education", "value": "Education"},
                                  {"label": "AgeCategory", "value": "AgeCategory"},
                                  {"label": "Marital_Status", "value": "Marital_Status"}]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='gb'))
            ]),
        ]),
        html.Div([
            html.H4("Customer Purchase Analysis"),
            html.Div([
                dbc.Label(id='sli', children='Select Purchase Category'),
                dcc.Slider(id="slider1", min=1, max=3, step=1, value=1,
                           marks={1: 'Education', 2: 'Age', 3: 'Marital Status'}),

            ]),
            html.Div([

                dbc.Label(id='sl', children='Select type of purchase'),
                dcc.Slider(id="slider2", min=1, max=4, step=1, value=1,
                           marks={1: 'DealsPurchases', 2: 'WebPurchases', 3: 'CatalogPurchases', 4: 'StorePurchases'}
                           ),
            ]),
            html.Div([
                dbc.Col(dcc.Graph(id='lp'))
            ]),
        ])
    ])])

    marketing_insight = html.Div([dbc.Container([
    ])])

    prediction = html.Div([dbc.Container([
        dbc.Row([
            html.H1(children='Akatsuki Analytics'),
            html.H2(children='Prediction', style={'text-align': 'center'})
        ], style={'display': 'flex', 'flex-direction': 'column', 'align-content': 'center'}),
        html.Hr(),
        html.Div([dcc.RadioItems(id='ma',
                                 options=[
                                     {'label': 'Married', 'value': 1},
                                     {'label': 'Unmarried', 'value': 0},
                                 ],
                                 value=1,
                                 labelStyle={'display': 'inline-block', 'margin-right': '15px', 'font-weight': 300},
                                 style={'display': 'inline-block', 'margin-left': '7px'}
                                 ),
                  html.Br(),
                  dbc.Label(children='Provide Income:  '),
                  dcc.Input(id='income', type="text"),
                  html.Br(),
                  html.Br(),
                  dbc.Label(id='sle', children='Select type of Degree:'),
                  dcc.Slider(id="slider7", min=1, max=5, step=1, value=1,
                             marks={1: 'Graduation', 2: 'PhD', 3: 'Master', 4: 'Basic', 5: '2n Cycle'}
                             ),
                  html.Br(),
                  html.Br(),
                  dbc.Label(children='Provide Age:  '),
                  dcc.Input(id='Age', type="text"),
                  html.Br(),
                  dbc.Button('Analyse', color='primary', id='ab', style={'margin-bottom': '1em'}, block=True),
                  html.Hr(),
                  html.H3(id='final'),
                  html.Hr(),

                  html.H5(
                      'Table showing accuracy metrics of Random Forest Classifier which classfies the potential customer'),

                  dcc.Graph(figure=potentialbar('ValidCustomer'), style={'width': '530px', 'height': '250px'}),
                  html.H5(
                      'Table showing accuracy metrics of Random Forest Classifier that predicts on what product customer is likely going to spend money '),

                  dcc.Graph(figure=potentialbar('MostSpntOn'), style={'width': '530px', 'height': '250px'}),
                  html.H5(
                      'Table showing error metrics of SVM Regressor that predicts the amount customer is going to purchase '),
                  dcc.Graph(figure=svmMetricts(), style={'width': '530px', 'height': '250px'})

                  ])
    ]),
    ])

    card1 = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4("Customer Insights", className="card-title"),
                    html.P("Visualizations of customer data",
                           className="card-text",
                           ),
                    dcc.Link('View Visualizations', href='/customer')
                ]
            ),
        ],
        style={"width": "18rem", "height": "15rem"},
    )

    card2 = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4("Product Insights", className="card-title"),
                    html.P(
                        "Visualizations of products data",
                        className="card-text",
                    ),
                    dcc.Link('View Visualizations', href='/product')
                ]
            ),
        ],
        style={"width": "18rem", "height": "15rem"},
    )

    card3 = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4("Analysis", className="card-title"),
                    html.P(
                        "Plots showing the analysis",
                        className="card-text",
                    ),
                    dcc.Link('View Visualizations', href='/analysis')
                ],
            ),
        ],
        style={"width": "18rem", "height": "15rem"},
    )

    card4 = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4("Purchase Insights", className="card-title"),
                    html.P("Visualizations of customer purchase data",
                           className="card-text",
                           ),
                    dcc.Link('View Visualizations', href='/purchase')
                ]
            ),
        ],
        style={"width": "18rem", "height": "15rem"},
    )

    card5 = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4("Marketing Insights", className="card-title"),
                    html.P("Visualizations of Marketing data",
                           className="card-text",
                           ),
                    dcc.Link('View Visualizations', href='/marketing1')
                ]
            ),
        ],
        style={"width": "18rem", "height": "15rem"},
    )

    card6 = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4("Predictions", className="card-title"),
                    html.P("Various predictions made on total customer purchases",
                           className="card-text",
                           ),
                    dcc.Link('View Predictions', href='/panalysis')
                ]
            ),
        ],
        style={"width": "18rem", "height": "15rem"},
    )

    index_page = html.Div([dbc.Container([
        dbc.Row([
            html.H1("Akatsuki Analytics"),
            html.H2("Customer Spending Pattern", style={'text-align': 'center'}),
        ],
            style={'display': 'flex', 'flex-direction': 'column', 'align-content': 'center'}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(card1, width="auto", style={'padding-bottom': '50px'}),
                dbc.Col(card2, width="auto"),
                dbc.Col(card3, width="auto"),
                dbc.Col(card4, width="auto"),
                dbc.Col(card5, width="auto"),
                dbc.Col(card6, width="auto"),
            ],
            style={'display': 'flex',
                   'justify-content': 'space-around', 'padding-bottom': '50px'})
    ])
    ],
        style={'background-color': '#D3D3D3', 'align': 'center',
               'display': 'flex',
               'justify-content': 'center',
               })

    @app.callback(dash.dependencies.Output('page-content', 'children'),
                  [dash.dependencies.Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/customer':
            return customer_Insight
        elif pathname == '/product':
            return product_insight
        elif pathname == '/analysis':
            return analysis
        elif pathname == '/purchase':
            return purchase_insight
        elif pathname == '/marketing':
            return marketing_insight
        elif pathname == '/panalysis':
            return prediction
        elif pathname == '/marketing1':
            return marketing
        else:
            return index_page

    @app.callback(Output('pie', 'figure'), [Input('dropdown', 'value')])
    def renderpie(category):
        df = read_dataset(Path('..', 'datasets', 'processed', 'cusanalysis.csv'))
        tmp = df.groupby([category], as_index=False, sort=False)[category].count()
        tmp['tmp'] = df[category].unique()
        return pie(tmp, 'tmp', category)

    @app.callback(Output('barline', 'figure'),
                  [Input('dropdown1', 'value'), Input('dropdown2', 'value'), Input('plottype', 'value'),
                   Input('barline', 'clickData')])
    def renderbar(x, y, pt, data):
        df = read_dataset(Path('..', 'datasets', 'processed', 'cusanalysis.csv'))
        tmp = df.groupby(x, as_index=False, sort=False)[y].mean()
        tmp[x] = tmp[x].unique()
        if not (data is None):
            print(data['points'][0]['label'])
        if pt == 'bp':
            return bar(tmp, x, y)
        if pt == 'lp':
            return line(tmp, x, y)
        if pt == 'clb':
            return compositelinebar(tmp, x, y)

    @app.callback(Output('hirar', 'figure'), [Input('dropdown3', 'value')])
    def hirar(value):
        df = read_dataset(Path('..', 'datasets', 'processed', 'customerhirarchy.csv'))
        if value == 1:
            return iciecle(df)
        else:
            return sunburst(df)

    @app.callback(Output('sb', 'figure'), [Input('eb', 'n_clicks')],
                  [State('dropdown4', 'value'), State('groupby', 'value')])
    def subplots(clicks, dd='Education', rb='mean'):
        dataframe = read_dataset(Path('..', 'datasets', 'processed', 'customerproducatanalysis.csv'))
        return subplotss(dataframe, dd, rb)

    @app.callback(Output('table', 'figure'), [Input('dropdown1', 'value'), Input('barline', 'clickData')])
    def rendertable(dv, barselected):
        df = read_dataset(Path('..', 'datasets', 'processed', 'cusanalysis.csv'))
        if not (barselected is None):
            return tabel(df=df, fd=dv, bv=barselected['points'][0]['label'])
        return tabel(df=df, fd=dv, bv='Graduation')

    @app.callback(Output('cpa', 'children'), Input('sb', 'clickData'), Input('dropdown4', 'value'))
    def renderpara(points, dv):
        dataframe = read_dataset(Path('..', 'datasets', 'processed', 'customerproducatanalysis.csv'))
        ybars = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        if not (points is None):
            dataframe = dataframe[dataframe[dv] == points['points'][0]['label']]
            timesbaught = len(dataframe[ybars[points['points'][0]['curveNumber']]]) - (
                    dataframe[ybars[points['points'][0]['curveNumber']]] == 0).sum(axis=0)
            avgamount = get_column_mean(dataframe, ybars[points['points'][0]['curveNumber']])
            avgamount = round(avgamount, 2)
            type = points['points'][0]['label']
            product = ybars[points['points'][0]['curveNumber']]
            product = product.replace('Mnt', '')
            product = product.replace('Products', '')
            product = product.replace('Prods', '')
            return f'Customer with {type} education bought {product} {timesbaught} times and spent an average of $ {avgamount} on the product'
        return 'Click on bars for more details'

    @app.callback(dash.dependencies.Output('final', 'children'), [Input('ab', 'n_clicks')],
                  [State('ma', 'value'), State('income', 'value'), State('slider7', 'value'), State('Age', 'value')])
    def renderprid(clicks, ms, inc, edu, age):
        edlist = [0, 0, 0, 0, 0]
        edlist[edu - 1] = 1
        plist = []
        df = read_dataset(Path('..', 'datasets', 'processed', 'analysis.csv'))
        plist.append(ms)
        print(inc)
        plist.append((int(inc) - df['Income'].min()) / (df['Income'].max() - df['Income'].min()))
        print(edu)
        for e in edlist:
            plist.append(e)
        print(age)
        plist.append((int(age) - df['Age'].min()) / (df['Age'].max() - df['Age'].min()))
        print(plist)
        rfp = random_forest_classifier(plist)
        svmr = np.round(svm(plist))
        print("customer will spend amount on")
        print(random_forest_classifier2(plist))
        rfc2 = random_forest_classifier2(plist)
        if rfp[0] == 'no':
            return f"Is not a potential customer and he might spend total amount of {svmr} and customer is most " \
                   f"likely to spend money on {rfc2[0]} "
        if rfp[0] == 'yes':
            return f"Is a potential customer and he might spend total amount of {svmr} and customer is most likely to " \
                   f"spend money on {rfc2[0]} "

    @app.callback(Output('lp', 'figure'), [Input('slider1', 'value'), Input('slider2', 'value')])
    def renderlp(sv1, sv2):
        pc = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
        cc = ['Education', 'AgeCategory', 'Marital_Status']

        sv1 = sv1 - 1
        sv2 = sv2 - 1
        print('Testing')
        print(sv1)
        print(pc[sv2])
        print('working')
        return scatter(cc[sv1], pc[sv2])

    @app.callback(Output('gb', 'figure'), Input('dropdown5', 'value'))
    def rendergb(ct):
        return groupedbar(ct)

    return app


if __name__ == "__main__":
    dashboard().run_server(debug=True, dev_tools_ui=False)
