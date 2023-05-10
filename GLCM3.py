import cv2
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import base64
import numpy as np
from skimage.feature import graycomatrix, graycoprops

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("GLCM特征展示"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            '拖放或 ',
            html.A('选择图片')
        ]),
        style={
            'width': '50%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    html.Div(id='output-glcms')
])

def analyze_image(contents):
    # 将上传的内容解码为图像格式
    encoded_image = contents.split(',')[1]
    decoded_image = cv2.imdecode(np.frombuffer(
        bytes(base64.b64decode(encoded_image)), np.uint8), -1)
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2GRAY)
    # 计算灰度共生矩阵
    glcm = graycomatrix(gray_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    # 计算GLCM特征
    props = ['ASM', 'correlation', 'energy', 'homogeneity', 'dissimilarity', 'contrast']
    features = []
    for prop in props:
        feature_val = graycoprops(glcm, prop)
        features.append(feature_val[0])
    return features

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def update_output_image_upload(contents):
    if contents is not None:
        # 将图像显示在网页上
        return html.Div([
            html.H3('上传的图像：'),
            html.Img(src=contents, style={'width': '400px'}),
        ])

@app.callback(Output('output-glcms', 'children'),
              Input('upload-image', 'contents'))
def update_output_glcms(contents):
    if contents is not None:
        features = analyze_image(contents)
        # 将GLCM特征显示在网页上
        return html.Div([
            html.H3('GLCM特征（0度角）：'),
            html.P(f'ASM(角二阶矩特征)：{features[0][0]:.4f}'),
            html.P(f'Correlation(相关性)：{features[1][0]:.4f}'),
            html.P(f'Energy(能量)：{features[2][0]:.4f}'),
            html.P(f'Homogeneity(同质性)：{features[3][0]:.4f}'),
            html.P(f'Dissimilarity(差异性)：{features[4][0]:.4f}'),
            html.P(f'Contrast(对比度)：{features[5][0]:.4f}'),
            html.H3('GLCM特征（45度角）：'),
            html.P(f'ASM(角二阶矩特征)：{features[0][1]:.4f}'),
            html.P(f'Correlation(相关性)：{features[1][1]:.4f}'),
            html.P(f'Energy(能量)：{features[2][1]:.4f}'),
            html.P(f'Homogeneity(同质性)：{features[3][1]:.4f}'),
            html.P(f'Dissimilarity(差异性)：{features[4][1]:.4f}'),
            html.P(f'Contrast(对比度)：{features[5][1]:.4f}'),
            html.H3('GLCM特征（90度角）：'),
            html.P(f'ASM(角二阶矩特征)：{features[0][2]:.4f}'),
            html.P(f'Correlation(相关性)：{features[1][2]:.4f}'),
            html.P(f'Energy(能量)：{features[2][2]:.4f}'),
            html.P(f'Homogeneity(同质性)：{features[3][2]:.4f}'),
            html.P(f'Dissimilarity(差异性)：{features[4][2]:.4f}'),
            html.P(f'Contrast(对比度)：{features[5][2]:.4f}'),
            html.H3('GLCM特征（135度角）：'),
            html.P(f'ASM(角二阶矩特征)：{features[0][3]:.4f}'),
            html.P(f'Correlation(相关性)：{features[1][3]:.4f}'),
            html.P(f'Energy(能量)：{features[2][3]:.4f}'),
            html.P(f'Homogeneity(同质性)：{features[3][3]:.4f}'),
            html.P(f'Dissimilarity(差异性)：{features[4][3]:.4f}'),
            html.P(f'Contrast(对比度)：{features[5][3]:.4f}'),
        ])
if __name__ == '__main__':
    app.run_server(debug=True,port=8091)
