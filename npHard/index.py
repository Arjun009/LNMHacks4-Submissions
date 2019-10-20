# Importing Libraries

import dash_core_components as dcc
import dash
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import dash_html_components as html
import base64
import webbrowser as wb
from grad_cam  import build_model,build_guided_model,compute_saliency
from PIL import Image
from base64 import decodestring
import numpy as np
import os
import urllib
import datafr
import ast
import dash_daq as daq
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
import pickle
from flask import Flask, Response
import cv2
import plotly.graph_objs as go




# external JS
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    "https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js",
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]

# external CSS stylesheets
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    },

    {
        'href': 'https://fonts.googleapis.com/css?family=Varela',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }

    
]



# Making app as an object of dash class. Invoking Dash function.

# Provide your name and content.



app = dash.Dash(__name__,meta_tags=[
    {
        'name': 'description',
        'content': 'My description'
    },
    {
        'http-equiv': 'X-UA-Compatible',
        'content': 'IE=edge'
    }
],
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)

server = app.server

##app.scripts.config.serve_locally = True
##app.css.config.serve_locally = True


# Your webpage title

app.title = 'LnmHacks 	&#x1F525;'

app.config['suppress_callback_exceptions']=True




'''

Here to write <div> , <h1>, ,<p>, etc. tags we need to specify them as html.Div, html.H1, html.P (As we are importing this tags from html library)


'''




# Your main layout

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

PLOTLY_LOGO = "/"



app.config['suppress_callback_exceptions']=True






# Update the index

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':

        # Default path is / . It will render Page_1_layout
        
        return page_1_layout

    if pathname=="/webcam":

        return page_2_layout
    else:
        return []
    # You could also return a 404 "URL not found" page here



# Root page

page_1_layout = html.Div([


        # Creating a Navbar

        dbc.Navbar(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/keras-logo-small-wb-1.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("LnmHacks", className="ml-4")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/",
            ),

            
            dbc.NavbarToggler(id="navbar-toggler"),
           
        ],
        color="dark",      
        dark=True,
        sticky="top",
    ),


    
     html.Div([    
     html.Div([

        dcc.Interval(id="interval", interval=250, n_intervals=0),
        html.Div([


        html.Div(id='variables',children=[],style={"display":"None"}),

         # Inside this class we are making dcc.Upload which will upload our image.
            
       dbc.Card(
            [
                dbc.CardHeader("LnmHacks \u2b50"),
                dbc.CardBody(
                    [html.Div([




                html.Div([

                 html.Div([
                  html.H5('Choose Model 		\ud83d\udcbb'),
                       
                 html.Div([
                            html.Div([dcc.Dropdown(id='dropdown-1',
                                                        
                            options=[{'label': 'Vgg16', 'value': 0},
                            {'label': 'ResNet50', 'value': 1},
                                     {'label': 'Nasnet', 'value': 2},
                                     {'label': 'Mobilenet', 'value': 3}
                             ],
                            value=0,
                            ),],className='three columns',style={'margin-bottom':'10px'}),
                            ],className="twelve columns",style={'margin-bottom':'10px'}),

                    ],className='twelve columns'),



                         html.H5('No. of Prediction \ud83d\udc68\u200d\ud83d\udcbb'),

                                     html.Div([
                                      html.Div([
                                      daq.Knob(
                                      id="knob",
                                      size=70,
                                      value=5,
                                      color="#ffaa00",
                                      max=10,
                                      min=1,
                                    ),

                                      ],className="three columns"),
                                       ],className="twelve columns",style={'margin-bottom':'10px'}),
                
                


                  ],className='twelve columns'),
            html.H5('Choose Image 	\ud83d\uddc3\ufe0f'),
            dcc.Upload(
                    id='upload-data',
                    children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
            ,' 	\ud83d\udcc1'
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin-bottom':'100%',
                        'margin': '1%'
                    },
                    # Allow multiple files to be uploaded
                   accept='image/*'
                )],className="ten columns offset-by-one")  ]), ]
        ),],style={ 'margin-top':'3%'}),   


         html.Div(id='output-image',style={"margin-top":"10px","margin-bottom":"10px"}),

        dbc.Card(
            [
                dbc.CardHeader("Select other probability/model to compare with the original prediction \ud83d\udc41\ufe0f "),
                dbc.CardBody(
                    [
                        
                     html.Div([
                   
                             
                     html.Div([dcc.Dropdown(id='dropdown',
                            options=[],
                        ),],style={"margin-top":"10px","margin-":"10px"},className='five columns'),

                     ]),
                      
                    ]
                ),
            ]
        ),


          html.Div(id='output-image-1',style={"margin-top":"10px","margin-":"10px"}),


    ],className='ten columns offset-by-one'),
     ]),

])






@app.callback(dash.dependencies.Output('dropdown', 'options'),
[Input('lolo', 'children')])
def set_cities_options(lolo):
    try:

##        pkl_file = open('myfile.pkl', 'rb')
##        mydict2 = pickle.load(pkl_file)
##        pkl_file.close()
        #o=abc()


        d=lolo[0]
        d=ast.literal_eval(d)
    
       
        print("\n\n",type(d),"\n11111111111111111111111111")
        return [{'label': k, 'value': k} for k in d.keys()]
        
    except Exception as e:
        return []








# Each time you give input as image in id ('upload-data') this function app.callback will fire and give you output in id ('output-image')

@app.callback(Output('output-image', 'children'),
              [Input('upload-data','contents')],
              [State('upload-data', 'filename'),State('variables', 'children'),State('dropdown-1','value'),State('dropdown-1','options'),State('knob','value')])
def update_graph_interactive_images(content,new_filename,dictvariables,number,labels,knob):


  
    if (content is not None):
      
        
        # Base64 string of the input image
        
        string = content.split(';base64,')[-1]
        
        imgdata = base64.b64decode(string)
        
        filename = 'some_image.png'  # I assume you have a way of picking unique filenames
        
        with open(filename, 'wb') as f:
            f.write(imgdata)

        #image = Image.fromstring('RGB',(200,200),decodestring(string))


       # Do You processing here ----------------



        basewidth = 224
        img = Image.open('some_image.png')
        img=img.convert('RGB')
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        datafr.H=224
        img = img.resize((224, 224),resample=0)
        datafr.W=224
        img.save('some_image.png')



        number=number
        model = build_model(number)
        guided_model = build_guided_model(number)
        layer_name=''


        modelNamesChoosen=''
        if number==1:
            layer_name='res5c_branch2c'
            modelNamesChoosen="ResNet50"
        elif number==0:
            layer_name='block5_conv3'
            modelNamesChoosen="VGG16"
        elif number==2:
            layer_name='normal_add_2_12'
            modelNamesChoosen="NasNetMobile"
        elif number==3:
            layer_name='conv_dw_13'
            modelNamesChoosen="MobileNet"
            




       
            
        gradcam, gb, guided_gradcam,dictionaries = compute_saliency(model, guided_model,layer_name=layer_name,img_path=img, cls=-1, visualize=False, save=True,string=number,knob=knob)


        img1=''
        with open("assets/gradcam.jpg", "rb") as imageFile:
            img1=base64.b64encode(imageFile.read()).decode("utf-8")
        img2=''
        with open("assets/guided_backprop.jpg", "rb") as imageFile:
            img2=base64.b64encode(imageFile.read()).decode("utf-8")
        img3=''
        with open("assets/guided_gradcam.jpg", "rb") as imageFile:
            img3=base64.b64encode(imageFile.read()).decode("utf-8")
        
        #print(datafr.d[0][1])
        #ac=str(list(dictvariables[d])[0])
        dictvariables=[]
        dictvariables.append(1)
        dictvariables.append(dictionaries)

##
##        output = open('myfile.pkl', 'wb')
##        pickle.dump(dictionaries, output)
##        output.close()

        # Manipulate here
        print("Yes1","\n\n")

        # You can write your python code or call a class/ function of ocr.

        # Your picked image will be stored as some_image.png locally. You can pick this image and manipulate.


        '''

        working example


        img1=''
        with open("some_image.png", "rb") as imageFile:
        img1=base64.b64encode(imageFile.read()).decode("utf-8")

        Now you can send this encoded base64 string to model or convert it into numpy array.


        '''
        

       ##################################
       
##        img1=''
##        with open("some_image.png", "rb") as imageFile:
##            img1=base64.b64encode(imageFile.read()).decode("utf-8")
       


        # Lets say your output text here I'm taking as base64 string of image


    
        filename = "some_image.png"
        parser = createParser(filename)
        metadata = extractMetadata(parser)
        linestring=[]
        counterline=0
        print("yes2","\n\n")
        for line in metadata.exportPlaintext():
            if counterline==0:
                linestring.append(html.Div([html.H5(line)]))
                counterline+=1
            else:
                linestring.append(html.Div([html.P(line)]))

        #datafr.loader+=10
        try:
            linestring=linestring.split("\n")[1:]
            linestring="\n".join(linestring)
        except:
            pass




        print("YES3.1","\n\n")
        ac=list(dictionaries)[0]
        print("YES3","\n\n")

 
        originalcard=html.Div([
        dbc.Card([
        dbc.CardBody(
            [dbc.CardTitle(new_filename)]
        ),
        dbc.CardImg(
            src=(
                'data:image/png;base64,{}'.format(string))
        ),
         dbc.CardBody([
                dbc.CardText(
                   html.P("Original Image")),]),],
    style={"max-width": "250px"},
),
],className="twelve columns")
        

        cardgradcam=dbc.Card(
    [
        dbc.CardBody(
            [dbc.CardTitle("Gradcam")]
        ),
        dbc.CardImg(
            src=(
                'data:image/jpg;base64,{}'.format(img1)
            )
        ),
         dbc.CardBody(
            [
                dbc.CardText(
                   ac
                ),
            ]
        ),
    ],
    style={"max-width": "250px"},
)

        cardprop=dbc.Card(
    [
        dbc.CardBody(
            [dbc.CardTitle("Guided Backpropogation")]
        ),
        dbc.CardImg(
            src=(
                'data:image/jpg;base64,{}'.format(img2)
            )
        ),
         dbc.CardBody(
            [
                dbc.CardText(
                   ac
                ),
            ]
        ),
    ],
    style={"max-width": "250px"},
)
        cardguided=dbc.Card(
    [
        dbc.CardBody(
            [dbc.CardTitle("Guided Gradcam")]
        ),
        dbc.CardImg(
            src=(
                'data:image/jpg;base64,{}'.format(img3)
            )
        ),
         dbc.CardBody(
            [
                dbc.CardText(
                   ac
                ),
            ]
        ),
    ],
    style={"max-width": "250px"},
)
        
    

 
##        org=[html.Div([html.Div([dbc.Card([
##                dbc.CardHeader("vizard \u2b50"),
##                dbc.CardBody(
##                    [   html.Div([
##                        html.Div([originalcard],className="five columns offset-by-one"),
##                        html.Div([
##                            html.Div([html.Div(linestring),],className='eight columns'),],className="six columns"),],className='row'),]),
##            ])],className="twelve columns"),
##            dbc.Card([
##                dbc.CardHeader(str(labels)+" thinks that it is "+ ac),
##                dbc.CardBody([html.Div([html.Div([
##            html.Div([
##                cardgradcam
##                ],className="four columns"),
##            html.Div([
##                cardprop
##                ],className="four columns"),
##            html.Div([
##                cardguided
##                ],className="four columns"),
##            ],className="row"),
##                    ],className="eleven columns offset-by-one",style={"margin-top":"10px","margin-bottom":"10px"}),
##                         ]),]),
##            html.Div([
##         dbc.Card(
##            [dbc.CardHeader("vizard \u2b50"),
##                dbc.CardBody(
##                    [html.Div(""),
##                    ]
##                ),
##            ]
##        ),],style={"margin-top":"10px","margin-":"10px"})])]
##        

        print("Library Changed,\n\n")

        #print(org)



        x=[]
        y=[]
        title="Probability For "+str(modelNamesChoosen)+" Model"
        for i in dictionaries:
            y.append(i)
            x.append(dictionaries[i][1])
        fig=[ dcc.Graph(
    id='example-graph-1',
    figure={
        'data': [
            {'x': y, 'y': x, 'type': 'bar', 'name': 'Probability'},
        ],
        'layout': {
            'title': title
        }
    }
)]
        

        

        return [

            html.Div([

                html.Div(id='lolo',children=[str(dictionaries)],style={'display':'None'}),

                 dbc.Card(
            [
                dbc.CardHeader("vizard \u2b50"),
                dbc.CardBody(
                    [   html.Div([



                        html.Div([
                        originalcard],className="five columns offset-by-one"),
                        html.Div([

                            html.Div([
                                
                                html.Div(linestring),
                                ],className='eight columns'),
                            ],className="six columns"),

                        ],className='row'




                                 ),
                    ]
                ),
            ]
        )

                ],className="twelve columns"),



            dbc.Card(
            [
                dbc.CardHeader(str(modelNamesChoosen)+" thinks that it is "+ac),
                dbc.CardBody(
                    [
                        html.Div([
                         html.Div([
            html.Div([
                cardgradcam
                ],className="four columns"),


            html.Div([
                cardprop

                ],className="four columns"),
            html.Div([

                cardguided
                ],className="four columns"),

            ],className="row"),



                    ],className="eleven columns offset-by-one",style={"margin-top":"10px","margin-":"10px"}),



                         ]),]
            ),

            html.Div([
         dbc.Card(
            [
                dbc.CardHeader("vizard \u2b50"),
                dbc.CardBody(
                    [
                        html.Div(fig),
                    ]
                ),
            ]
        ),],style={"margin-top":"10px","margin-":"10px"})




            ]








@app.callback(Output('output-image-1', 'children'),
              [Input('dropdown', 'value')],
              [State('dropdown-1','value'),
               State('dropdown-1','options'),
               State('knob','value'),
               State('lolo','children')])
def update_graph_interactive_image(content,number,label,knob,dictvariables):
    if (content is not None):
        print(content)
##        for i in label:
##            if i['value']==number:
##                dictvariables[label2]=i['label']
##        dictvariables[check]=0
        
        #datafr.loader=0
        #dictvariables[d]=dict()
        #dictvariables[predict]=knob
        #dictvariables[flags1]=1
        img=Image.open('some_image.png')
        print(img)
        number=number
        print('-----------')
        print(dictvariables[0],type(dictvariables[0]))
        dictvariables=ast.literal_eval(dictvariables[0])
        aa=dictvariables[content][0]
        model = build_model(number)
        guided_model = build_guided_model(number)

        modelNamesChoosen=''
        if number==1:
            layer_name='res5c_branch2c'
            modelNamesChoosen="ResNet50"
        elif number==0:
            layer_name='block5_conv3'
            modelNamesChoosen="VGG16"
        elif number==2:
            layer_name='normal_add_2_12'
            modelNamesChoosen="NasNetMobile"
        elif number==3:
            layer_name='conv_dw_13'
            modelNamesChoosen="MobileNet"
            
        gradcam, gb, guided_gradcam,dictionaries = compute_saliency(model, guided_model,layer_name=layer_name,img_path=img, cls=aa, visualize=False, save=True,string=number,knob=knob)
        
        img1=''
        with open("assets/gradcam.jpg", "rb") as imageFile:
            img1=base64.b64encode(imageFile.read()).decode("utf-8")
        img2=''
        with open("assets/guided_backprop.jpg", "rb") as imageFile:
            img2=base64.b64encode(imageFile.read()).decode("utf-8")
        img3=''
        with open("assets/guided_gradcam.jpg", "rb") as imageFile:
            img3=base64.b64encode(imageFile.read()).decode("utf-8")
      
        ac=content
##        try:
##            img_data = open('some_image.png', 'rb').read()
##            msg = MIMEMultipart()
##            msg['Subject'] = 'Keras Visualtion Tool Report'
##            tempstring="The probability predicted by the "+datafr.label2+" model are"+str(datafr.d1)
##            text = MIMEText(tempstring)
##            msg.attach(text)
##            image_data = MIMEImage(img_data, name=os.path.basename('some_image.png'))
##            msg.attach(image_data)
##            img_data = open('assets/gradcam.png', 'rb').read()
##            image_data = MIMEImage(img_data, name=os.path.basename('assets/gradcam.png'))
##            msg.attach(image_data)
##            img_data = open('assets/guided_backprop.png', 'rb').read()
##            image_data = MIMEImage(img_data, name=os.path.basename('assets/guided_backprop.png'))
##            msg.attach(image_data)
##            img_data = open('assets/guided_gradcam.png', 'rb').read()
##            image_data = MIMEImage(img_data, name=os.path.basename('assets/guided_gradcam.png'))
##            msg.attach(image_data)
##            FROM = "vizard.keras@gmail.com"
##            TO = str(email).split(",")
##
##            import smtplib
##            try:
##                server = smtplib.SMTP('smtp.gmail.com', 587)
##            except:
##                 server = smtplib.SMTP('smtp.gmail.com', 465)
##            server.starttls()
##            server.login("#youremail", "#yourpassword")
##            server.sendmail(FROM, TO, msg.as_string())
##            server.quit()
##        except Exception as e:
##            print('mail error'+str(e))


        ##for firebase no need
        '''
        imageBlob=bucket.blob("gradcam/"+'gradcam_'+datafr.label1+"_"+content+"_"+datafr.new_filename.split(".")[0])
        imageBlob.upload_from_filename('assets/gradcam.jpg')
        imageBlob=bucket.blob("guided_backprop/"+'guided_backprop_'+"_"+datafr.label1+content+"_"+datafr.new_filename.split(".")[0])
        imageBlob.upload_from_filename('assets/guided_backprop.jpg')
        imageBlob=bucket.blob("guided_gradcams/"+'guided_gradcam_'+"_"+datafr.label1+content+"_"+datafr.new_filename.split(".")[0])
        imageBlob.upload_from_filename('assets/guided_gradcam.jpg')

        '''
        ###datafr.loader+=10
        cardgradcam=dbc.Card(
    [
        dbc.CardBody(
            [dbc.CardTitle("Gradcam")]
        ),
        dbc.CardImg(
            src=(
                'data:image/jpg;base64,{}'.format(img1)
            )
        ),
         dbc.CardBody(
            [
                dbc.CardText(
                   ac
                ),
            ]
        ),
    ],
    style={"max-width": "250px"},
)

        cardprop=dbc.Card(
    [
        dbc.CardBody(
            [dbc.CardTitle("Guided Backpropogation")]
        ),
        dbc.CardImg(
            src=(
                'data:image/jpg;base64,{}'.format(img2)
            )
        ),
         dbc.CardBody(
            [
                dbc.CardText(
                   ac
                ),
            ]
        ),
    ],
    style={"max-width": "250px"},
)
        cardguided=dbc.Card(
    [
        dbc.CardBody(
            [dbc.CardTitle("Guided Gradcam")]
        ),
        dbc.CardImg(
            src=(
                'data:image/jpg;base64,{}'.format(img3)
            )
        ),
         dbc.CardBody(
            [
                dbc.CardText(
                   ac
                ),
            ]
        ),
    ],
    style={"max-width": "250px"},
)
        #nnn = ToastNotifier() 
        #datafr.loader+=10
        #nnn.show_toast("vizard ", "Task is Completed", duration = 5, 
         #icon_path ="assets/favicon.ico") 

        #dictvariables[check]=1






        x=[]
        y=[]
        title="Probability For "+str(modelNamesChoosen)+" Model"
        for i in dictionaries:
            y.append(i)
            x.append(dictionaries[i][1])
        fig=[ dcc.Graph(
    id='example-graph-2',
    figure={
        'data': [
            {'x': y, 'y': x, 'type': 'bar', 'name': 'Probability'},
        ],
        'layout': {
            'title': title
        }
    }
)]
     
        return [        dbc.Card(
            [
                dbc.CardHeader(str(modelNamesChoosen)+" thinks that it is "+ac),
                dbc.CardBody(
                    [
                        html.Div([
                         html.Div([
            html.Div([
                cardgradcam
                ],className="four columns"),


            html.Div([
                cardprop

                ],className="four columns"),
            html.Div([

                cardguided
                ],className="four columns"),

            ],className="row"),



                    ],className="eleven columns offset-by-one",style={"margin-top":"10px","margin-":"10px"}),



                         ]),]
            ),


             html.Div([
     dbc.Card(
        [
            dbc.CardHeader("vizard \u2b50"),
            dbc.CardBody(
                [
                    html.Div(fig),
                ]
            ),
        ]
    ),],style={"margin-top":"10px","margin-":"10px"}),

     






            ]

        return [
            html.Div([html.Div([cardgradcam]),
            html.Img(src='data:image/jpg;base64,{}'.format(img1)),
                 html.Img(src='data:image/jpg;base64,{}'.format(img2)),
                 html.Img(src='data:image/jpg;base64,{}'.format(img3)),])
                    ]



           
    else:
        return []

        return [html.Div([html.Img(src='data:image/jpg;base64,{}'.format(img1)),
                 html.Img(src='data:image/jpg;base64,{}'.format(img2)),
                 html.Img(src='data:image/jpg;base64,{}'.format(img3)),])]


















# Page



page_2_layout=[]

##page_2_layout=html.Div([
##
##    dbc.Navbar(
##        [
##            html.A(
##                # Use row and col to control vertical alignment of logo / brand
##                dbc.Row(
##                    [
##                        dbc.Col(html.Img(src="assets/keras-logo-small-wb-1.png", height="30px")),
##                        dbc.Col(dbc.NavbarBrand("LnmHacks", className="ml-4")),
##                    ],
##                    align="center",
##                    no_gutters=True,
##                ),
##                href="/",
##            ),
##
##           
##
##            dbc.NavItem(dbc.NavLink("Webcam", href="/webcam")),
##            
##            dbc.NavbarToggler(id="navbar-toggler"),
##           
##        ],
##        color="dark",      
##        dark=True,
##        sticky="top",
##    ),
##
##
##    
##     html.Div([    
##     html.Div([
##
##        dcc.Interval(id="interval", interval=250, n_intervals=0),
##        html.Div([
##
##
##        html.Div(id='variables',children=[],style={"display":"None"}),
##
##         # Inside this class we are making dcc.Upload which will upload our image.
##            
##       dbc.Card(
##            [
##                dbc.CardHeader("LnmHacks \u2b50"),
##                dbc.CardBody(
##                    [html.Div([
##
##                  html.H5('Choose Model 		\ud83d\udcbb'),
##                                      ],className="three columns"),
##                                       ],className="twelve columns",style={'margin-bottom':'10px'}),
##
##                dcc.RadioItems(
##
##                    id='radio-1',
##    options=[
##        {'label': 'MNIST', 'value': 'mnist'},
##        {'label': 'Custom Datasets', 'value': 'custom'}
##    ],
##    value='mnist',
##    labelStyle={'display': 'inline-block'}),
##
##
##            
##            html.Div(id='anotherRadio'),
##            html.Div(id='chooseImage'),
##
##
##             dcc.RadioItems(
##
##                    id='radio-2',
##    options=[
##        {'label': 't-SNE', 'value': 'TSNE'},
##        {'label': 'PCA', 'value': 'PCA'}
##    ],
##    value='TSNE',
##    labelStyle={'display': 'inline-block'}),
##
##            
##            dcc.Dropdown(
##
##                id='dropdown-22',
##    options=[
##        {'label': '3D', 'value': '3'},
##        {'label': '2D', 'value': '2'}
##    ],
##    value='3'
##),  
##
##        html.Div(dcc.Input(id='input-box-1', type='text')),
##
##
##                
##            
##                 
##            
##
##         html.Div(id='output-image-33',style={"margin-top":"10px","margin-bottom":"10px"}),
##
##
##    html.Div([
##
##dbc.Button("Click Image 	\ud83d\udc4b", id="submitthree", className="mr-1",color="warning",outline=True,style={"margin-top":"3%","margin-top":"5px",'border':"#FFC107 2px solid "})
##        ],className="twelve columns offset-by-three"),
##
##    html.Div([
##
##    html.Div(id="finalOutputText",style={"margin-top":"3%"}),
##
##
##    ],className="ten columns offset-by-one"),
##    
##])
##
##@app.callback(Output('anotherRadio', 'children'),
##              [Input('radio-1','value')])
##def update_graph_interactive_images(clicked):
##    if clicked=='mnsit':
##        return []
##    else:
##        return[
##
##                dcc.RadioItems(
##
##                    id='radio-3',
##    options=[
##        {'label': 'Label', 'value': True},
##        {'label': 'No-Label', 'value': False}
##    ],
##    value=True,
##    labelStyle={'display': 'inline-block'}),
##
##
##                
##
##            ]
##
##
##
##
##@app.callback(Output('output-image-33 ', 'children'),
##              [Input('radio-3','value')])
##
##def update_graph_interactive_images(clicked):
##    if clicked:
##
##            html.H5('Choose Label 	\ud83d\uddc3\ufe0f'),
##            
##            dcc.Upload(
##                    id='upload-data-2',
##                    children=html.Div([
##            'Drag and Drop or ',
##            html.A('Select Files')
##            ,' 	\ud83d\udcc1'
##                    ]),
##                    style={
##                        'width': '100%',
##                        'height': '60px',
##                        'lineHeight': '60px',
##                        'borderWidth': '1px',
##                        'borderStyle': 'dashed',
##                        'borderRadius': '5px',
##                        'textAlign': 'center',
##                        'margin-bottom':'100%',
##                        'margin': '1%'
##                    },
##                    # Allow multiple files to be uploaded
##                )],className="ten columns offset-by-one")  ]), ]
##        ),],style={ 'margin-top':'3%'}),   
##
##
##            ]
##
##
##@app.callback(Output('chooseImage', 'children'),
##              [Input('radio-1','value')])
##def update_graph_interactive_images(clicked):
##    if clicked=='mnsit':
##        return []
##    else:
##        return[
##
##
##            html.H5('Choose Image 	\ud83d\uddc3\ufe0f'),
##            
##            dcc.Upload(
##                    id='upload-data-1',
##                    children=html.Div([
##            'Drag and Drop or ',
##            html.A('Select Files')
##            ,' 	\ud83d\udcc1'
##                    ]),
##                    style={
##                        'width': '100%',
##                        'height': '60px',
##                        'lineHeight': '60px',
##                        'borderWidth': '1px',
##                        'borderStyle': 'dashed',
##                        'borderRadius': '5px',
##                        'textAlign': 'center',
##                        'margin-bottom':'100%',
##                        'margin': '1%'
##                    },
##                    # Allow multiple files to be uploaded
##                    multiple=True,
##                   accept='image/*'
##                )],className="ten columns offset-by-one")  ]), ]
##        ),],style={ 'margin-top':'3%'}),   
##
##
##            ]
##
##
##def 2d(df,col,algo):
##     a=html.Div([
##    dcc.Graph(
##        id='life-exp-vs-gdp',
##        figure={
##            'data': [
##                go.Scatter(
##                    x=df['one'],
##                    y=df['two'],
##                    text=algo+"1",
##                    mode='markers',
##                    opacity=0.7,
##                    marker={
##                        'size': 15,
##                        'line': {'width': 0.5, 'color': 'white'}
##                    },
##                    name=i
##                ) for i in [algo+"1",algo+'2']
##            ],
##            'layout': go.Layout(
##                xaxis={ 'title': 'PCA-1'},
##                yaxis={'title': 'PCA-2'},
##                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
##                legend={'x': 0, 'y': 1},
##                hovermode='closest'
##            )
##        }
##    )
##])
##        return a
##    
##
##
##
##
##
##def from_base64(base64_data):
##    nparr = np.fromstring(base64_data, np.uint8)
##    return cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
##
##
##@app.callback(Output('finalOutputText', 'children'),
##              [Input('submitthree','n_clicks')],
##              [State('upload-data-1', 'content'),State('upload-data-2', 'content'),State('radio-1', 'value'),State('radio-2', 'value'),State('radio-3', 'value'),State('dropdown-22', 'value'),State('input-box-1','value')])
##def update_graph_interactive_images(content,filename,mnsit,algo,label,3D,seed):
##    if mnsit=='mnsit':
##        df,col=DR2.mains(algo,int(3D),seed,mnsit,True,None)
##
##        if 3D==2:
##            fig=2d(df,col,algo)
##
##            return [fig]
##        
##
##    else:
##        if content is not None:
##            dict1={'X':[],'Y':[]}
##                
##            if label:
##                for i in content:
##                    z=i.split(';base64,')[-1]
##                    imgdata = base64.b64decode(z)
##                    v=from_base64(imgdata)
##                    dict1['X'].append(v)
##
##                 #labels
##                 z=filename.split(';base64,')[-1]
##                 decoded = base64.b64decode(z).split("\n")
##                 dict1['Y']=decoded
##
##                 get_image_data(dict1)
##
##                 
##                    
##                
##                
##            else:
##                dict1={'X':[]}
##                for i in content:
##                    z=i.split(';base64,')[-1]
##                    imgdata = base64.b64decode(z)
##                    v=from_base64(imgdata)
##                    dict1['X'].append(v)
##                
##            df,col=DR2.mains(algo,int(3D),seed,mnsit,label,dict1)
##
##            if 3D=='2':
##                fig=2d(df,col,algo)
##            
##        else:
##            dict1={'X':[]}
##            df,col=DR2.mains(algo,int(3D),seed,mnsit,False,dict1)
##
##            if 3D=='2':
##                fig=2d(df,col,algo)
##
##        return [fig]
##            
##            


    
    
    

    


if __name__ == '__main__':
    app.run_server(debug=False)

