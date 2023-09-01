#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename     : Style-Converter.py
@description     : 一个简易的sdwebui界面,隐藏了复杂的参数和设置,用于风格转换
@time     : 2023/08/21
@author     : SSL
@Version     : 1.0
'''
import re
import os
import cv2
from PIL import Image, PngImagePlugin
import base64
import json
import requests
import gradio as gr
import io
import datetime
from modules import scripts, shared, shared_items, script_callbacks, extras, errors,sd_models


sd_url = "http://127.0.0.1:7860"

# #controlnet external_code
# external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')


    
# Note: Change symbol hints mapping in `javascript/hints.js` when you change the symbol values.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
refresh_symbol = "\U0001f504"  # 🔄
switch_values_symbol = "\U000021C5"  # ⇅
camera_symbol = "\U0001F4F7"  # 📷
reverse_symbol = "\U000021C4"  # ⇄
tossup_symbol = "\u2934"
trigger_symbol = "\U0001F4A5"  # 💥
open_symbol = "\U0001F4DD"  # 📝
detect_image_size_symbol = '\U0001F4D0'  # 📐

#按钮
class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""
    def __init__(self, **kwargs):
        super().__init__(variant="tool", 
                         elem_classes=kwargs.pop('elem_classes', []) + ["cnet-toolbutton"], 
                         **kwargs)
    def get_block_name(self):
        return "button"
    
# # 设置Gradio界面的主题
# def set_theme(theme):
#     if theme == "dark":
#         gr.set_theme(gr.theme.DARK)
#     else:
#         gr.set_theme(gr.theme.LIGHT)

# json  
def save_json(save_path,data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path,'w') as file:
        json.dump(data,file)
def load_json(file_path):
    assert file_path.split('.')[-1] == 'json'
    with open(file_path,'r') as file:
        data = json.load(file)
    return data 

# 启动读取风格预设
current_dir = scripts.basedir()
style_presets = load_json(current_dir+"/style_presets.json")
style_presets_list = list(style_presets)
# 启动读取checkpoint
root_dir = os.path.dirname( os.path.dirname(current_dir))
checkpoint = load_json(root_dir+"/config.json")["sd_model_checkpoint"]
# 启动读取vae
vae = load_json(root_dir+"/config.json")["sd_vae"]
# 重新读取风格预设
def refresh_presets():
    global style_presets
    style_presets = load_json(current_dir+"/style_presets.json")
    style_presets_list = list(style_presets)
    return gr.Dropdown.update(choices=style_presets_list)

# 设置风格预设参数
def set_parm_presets(i2i_prompt_styles, i2i_checkpoints, i2i_vae):
    params = style_presets[i2i_prompt_styles]
    # print(shared.list_checkpoint_tiles())
    if "sd_model_checkpoint" in params.keys():
        if params["sd_model_checkpoint"] in shared.list_checkpoint_tiles():
            i2i_checkpoints = params["sd_model_checkpoint"]
        else:
            print(u"错误:没有在预设json找到 checkpoint!")
    if "sd_vae" in params.keys():
        if params["sd_vae"] in shared_items.sd_vae_items():
            i2i_vae = params["sd_vae"]
        elif params["sd_vae"] == "None":
            i2i_vae = None
        else:
            print(u"错误:没有在预设json找到 VAE!")
    # print(params)
    payload = params["payload"]
    i2i_prompt_input = payload["prompt"] if "prompt" in payload.keys() else ""
    i2i_negativeprompt_input = payload["negative_prompt"] if "negative_prompt" in payload.keys() else ""
    i2i_width = payload["width"] if "width" in payload.keys() else 960
    i2i_height = payload["height"] if "height" in payload.keys() else 540
    i2i_cfg_scale = payload["cfg_scale"] if "cfg_scale" in payload.keys() else 7
    i2i_denoising_strength = payload["denoising_strength"] if "denoising_strength" in payload.keys() else 0.75
    i2i_sampling_steps = payload["steps"] if "steps" in payload.keys() else 20
    return [i2i_checkpoints,i2i_vae,i2i_prompt_input,i2i_negativeprompt_input,i2i_width,i2i_height,i2i_cfg_scale,i2i_denoising_strength,i2i_sampling_steps]

# # 读取checkpoints,但是在api启动后
# def api_getcheckpoint_titles(): 
#     response = requests.get(url=f'{sd_url}/sdapi/v1/sd-models')
#     if response.status_code != 200:
#         print(response)
#     r = response.json()
#     result = []
#     for i in r:
#         result.append(i["title"])
#     return result
# checkpoint_titles = api_getcheckpoint_titles()
# # 读取vae
# def api_getvae(): 
#     response = requests.get(url=f'{sd_url}/sdapi/v1/sd-vae')
#     if response.status_code != 200:
#         print(response)
#     r = response.json()
#     result = []
#     for i in r:
#         result.append(i["model_name"])
#     return result
# vaes = api_getvae()

# 切模型
# def change_checkpoint(i2i_checkpoints):
#     option_payload = {
#         "sd_model_checkpoint": i2i_checkpoints,
#     }
#     response = requests.post(url=f'{sd_url}/sdapi/v1/options', json=option_payload)
#     if response.status_code != 200:
#         print(response)
#     # r = response.json()
#     return i2i_checkpoints
# def change_vae(i2i_vae):
#     option_payload = {
#         "sd_vae": i2i_vae,
#     }
#     response = requests.post(url=f'{sd_url}/sdapi/v1/options', json=option_payload)
#     if response.status_code != 200:
#         print(response)
#     # r = response.json()
#     return i2i_vae

# numpy 转 base64
def numpy_to_base64(image_np): 
    # data = cv2.imencode('.jpg', image_np)[1]
    # image_bytes = data.tobytes()
    # image_base4 = base64.b64encode(image_bytes).decode('utf8')
    # retval, buffer = cv2.imencode('.jpg', image_np)
    # pic_str = base64.b64encode(buffer)
    # image_base4 = pic_str.decode('utf-8')
    # 读取二进制图片，获得原始字节码，注意 'rb'
    with open(image_np, 'rb') as jpg_file:
        byte_content = jpg_file.read()
    image_base4 = base64.b64encode(byte_content).decode("utf-8")
    return image_base4

# 交换宽高
def switchWidthHeight(width,height):
    return [height,width]
# 测量图片
def detect_image_size(imginput,width,height):
    if imginput is None:
        return [width,height]
    else:
        src = cv2.imread(imginput) 
        image = src.shape 
        return [image[1], image[0]]

# 如果输出图片存在的话，获取输出图的seed
def reuse_seed(i2i_image_output):
    seed = -1
    if i2i_image_output is not None:
        _, pnginfo, _ = extras.run_pnginfo(i2i_image_output)
        try:
            png_list = re.split(" |,",pnginfo)
            # print(png_list)
            index = png_list.index("Seed:")
            if index != -1:
                seed = png_list[index+1]
        except json.decoder.JSONDecodeError:
            if pnginfo:
                errors.report(f"Error parsing JSON generation info: {pnginfo}")
    return seed
    
def api_getoptions(option:str):
    response = requests.get(url=f'{sd_url}/sdapi/v1/options')
    r = response.json()
    return r[option]

# 使用api自带存储在图生图文件夹
# def save_name(infotext:str):
#     save_path = "./outputs/styleconverter-images" # output dir for styleconverter images
#     if api_getoptions("directories_filename_pattern") == "[date]":
#         save_path = save_path + "/" + str(datetime.date.today())
#     os.makedirs(save_path, exist_ok=True)
#     # 列出save_path目录下的所有文件,用"-"拆分文件名，取前缀最大的
#     max_num = 0
#     seed = -1
#     for file in os.listdir(save_path):
#         if re.match(r"^\d+(-\d+)*",file):
#             if int(re.split("-",file)[0]) > max_num:
#                 max_num  = int(re.split("-",file)[0])
#     numstr = str(max_num+1).zfill(5)    # 1 -》 00001
#     try:
#         png_list = re.split(" |,",infotext)
#         index = png_list.index("Seed:")
#         if index != -1:
#             seed = png_list[index+1]
#     except:
#         errors.report(f"Error parsing JSON generation info: {infotext}")
#     png_name = save_path + "/" + numstr + "-" + str(seed) + ".png"
#     return png_name

def api_getimg(api:str,payload:dict):
    response = requests.post(url=f'{sd_url}/{api}', json=payload)
    r = response.json()
    if response.status_code != 200:
        print(response)
        return
    #使用controlnet后会返回controlnet控制图片，结果图在第一张
    result = r['images'][0]
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    # infotext=json.loads(r['info'])['infotexts'][0]#图片信息
    # pnginfo = PngImagePlugin.PngInfo()
    # pnginfo.add_text('parameters', infotext)
    # pngname = save_name(str(infotext))
    # print(u"保存图片："+pngname)
    # image.save(pngname, pnginfo=pnginfo)
    return image



def start_i2i(i2i_prompt_styles,i2i_checkpoints,i2i_vae,i2i_prompt_input,i2i_negativeprompt_input,i2i_image_input,i2i_seed,i2i_width,i2i_height,i2i_cfg_scale,i2i_denoising_strength,i2i_sampling_steps,i2i_controlnet_strength):
    """
    图生图
    """
    #初始化
    base64_bytes = numpy_to_base64(i2i_image_input)
    init_images = [base64_bytes]
    print(i2i_checkpoints)
    payload = {            
        "sampler_name": "DPM++ 2M Karras",
        "override_settings": {
            "sd_model_checkpoint": i2i_checkpoints,
            "sd_vae": i2i_vae,
            "CLIP_stop_at_last_layers": 2
        },
        "init_images": init_images,
        "prompt": i2i_prompt_input,
        "negative_prompt": i2i_negativeprompt_input,
        "seed": i2i_seed,
        "width": i2i_width,
        "height": i2i_height,
        "denoising_strength": i2i_denoising_strength,
        "steps": i2i_sampling_steps,
        "cfg_scale": i2i_cfg_scale,
        "steps": i2i_sampling_steps,
        "save_images": True
    }
    if i2i_prompt_styles != "Default":#有预设
        if "payload" in style_presets[i2i_prompt_styles].keys():
            payload_from_preset = style_presets[i2i_prompt_styles]["payload"]
            if "alwayson_scripts" in payload_from_preset.keys() and "controlnet" in payload_from_preset["alwayson_scripts"].keys():
                print(u"读取预设的controlnet参数！")
                cnet_args = payload_from_preset["alwayson_scripts"]["controlnet"]["args"]
                for i in range(len(cnet_args)):
                    cnet_args[i]["input_image"] = init_images
                    if "weight_max" in cnet_args[i].keys() and "weight_min" in cnet_args[i].keys():
                        cnet_args[i]["weight"] = (cnet_args[i]["weight_max"]-cnet_args[i]["weight_min"])*i2i_controlnet_strength+cnet_args[i]["weight_min"]
                        del cnet_args[i]["weight_max"],cnet_args[i]["weight_min"]
                # 添加cnet参数
                payload.update({
                    "alwayson_scripts":{
                        "controlnet":{
                            "args":cnet_args
                        }
                    }
                })
        else:# 预设里没有payload
            print(u"警告！预设文件中没找到payload")
    else:
        print(u"没有预设，图生图")
    # save_json("savejason1.json",payload)
    image = api_getimg("sdapi/v1/img2img",payload)
    return image  
 
def create_UI():
    # demo_ui
    with gr.Blocks(analytics_enabled=False) as demo_ui:
        with gr.Tab("img2img"):
            with gr.Row(elem_id=f"i2i_toprow", variant="panel", equal_height=True):#上半部分主页面.style(equal_height=True)
                with gr.Column(elem_id=f"i2i_leftcolimn", scale=6):
                #     with gr.Row(elem_classes=["i2i-imageoutput-row"]):
                    i2i_image_output = gr.Image(type="pil", height=540)#.style(height=540)
                with gr.Column(elem_id=f"i2i_rightcolimn", scale=4):
                    with gr.Row():
                        with gr.Row(elem_id=f"i2i_styles_row"):
                            i2i_style_presets = gr.Dropdown(label="Style preset", elem_id=f"i2i_styles", choices = style_presets_list, value="Dafault", multiselect=False)
                            i2i_refresh_button = ToolButton(value=refresh_symbol,visible=True)
                            i2i_refresh_button.click(fn=refresh_presets, outputs=i2i_style_presets)
                        with gr.Column(elem_id=f"i2i_actions_column"):
                            with gr.Row(elem_id=f"i2i_generate_box", elem_classes="generate-box"):
                                i2i_interrupt = gr.Button('Interrupt', elem_id=f"i2i_interrupt", elem_classes="generate-box-interrupt")
                                i2i_skip = gr.Button('Skip', elem_id=f"i2i_skip", elem_classes="generate-box-skip")
                                i2i_submit = gr.Button('Generate', elem_id=f"i2i_generate", variant='primary')
                    with gr.Row():
                        i2i_checkpoints = gr.Dropdown(label="Stable Diffusion checkpoint", elem_id=f"i2i_checkpoints", choices=shared.list_checkpoint_tiles(), value=checkpoint, multiselect=False)
                        i2i_checkpointsrefresh_button = ToolButton(value=refresh_symbol,visible=True)
                        i2i_checkpointsrefresh_button.click(fn=shared.refresh_checkpoints, inputs=[], outputs=[])#shared.refresh_checkpoints()
                        i2i_vae = gr.Dropdown(label="SD VAE", elem_id=f"i2i_vae", choices=shared_items.sd_vae_items(), value=vae, multiselect=False)
                        i2i_vaerefresh_button = ToolButton(value=refresh_symbol,visible=True)
                        i2i_vaerefresh_button.click(fn=shared_items.refresh_vae_list, inputs=[], outputs=[])
                        # i2i_checkpoints.change(fn=change_checkpoint, inputs=[i2i_checkpoints], outputs=[i2i_checkpoints])
                        # i2i_vae.change(fn=change_vae, inputs=[i2i_vae], outputs=[i2i_vae])
                    with gr.Row(elem_classes=["i2i-image-row"], equal_height=True):#.style(equal_height=True):
                        i2i_image_input = gr.Image(type="filepath", height=380)#.style(height=380)
            with gr.Row(elem_id=f"i2i_downrow", variant="panel", equal_height=True):#.style(equal_height=True):#下半部分参数
                with gr.Column(elem_id=f"i2i_prompt_container", scale=6):
                    with gr.Column(scale=2):
                        i2i_prompt_input = gr.Textbox(
                            label="Prompt", 
                            elem_id=f"i2i_prompt", 
                            show_label=False, 
                            lines=3, 
                            placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)", 
                            elem_classes=["prompt"])
                    with gr.Column(scale=2):
                        i2i_negativeprompt_input = gr.Textbox(
                            label="Negative prompt", 
                            elem_id=f"i2i_neg_prompt", 
                            show_label=False, 
                            lines=3, 
                            placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)", 
                            elem_classes=["prompt"])
                    with gr.Row(scale=1, elem_id=f"i2i_seed_container", variant="compact"):
                        i2i_seed = gr.Number(label='Seed', value=-1, elem_id=f"i2i_seed", container=False)
                        #i2i_seed.style(container=False)
                        i2i_random_seed = ToolButton(value=random_symbol, visible=True, elem_id=f"i2i_random_seed", label='Random seed')
                        i2i_random_seed.click(fn=lambda: -1,inputs=None,outputs=i2i_seed,show_progress=False)
                        i2i_reuse_seed = ToolButton(value=reuse_symbol, visible=True, elem_id=f"i2i_reuse_seed", label='Reuse seed')
                        i2i_reuse_seed.click(fn=reuse_seed,inputs=i2i_image_output,outputs=i2i_seed,show_progress=False)
                with gr.Column(elem_id=f"i2i_prompt_container", scale=4):
                    with gr.Row(elem_id=f"i2i_param_container"):
                        with gr.Column(elem_id="i2i_column_size", scale=4):
                            i2i_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=960, elem_id="i2i_width")
                            i2i_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=540, elem_id="i2i_height")
                        with gr.Column(elem_id="i2i_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                            res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="i2i_res_switch_btn")
                            res_switch_btn.click(fn=switchWidthHeight, inputs=[i2i_width,i2i_height], outputs=[i2i_width,i2i_height], show_progress=False)
                            detect_image_size_btn = ToolButton(value=detect_image_size_symbol, elem_id="i2i_detect_image_size_btn")
                            detect_image_size_btn.click(fn=detect_image_size, inputs=[i2i_image_input, i2i_width, i2i_height], outputs=[i2i_width,i2i_height], show_progress=False)
                    i2i_cfg_scale = gr.Slider(minimum=1.0,maximum=30.0,step=0.5,value=7.0,label="CFG scale", elem_id=f"i2i_cfg_scale")
                    i2i_denoising_strength = gr.Slider(minimum=0.0,maximum=1.0,step=0.01,value=0.75,label="Denoising strength", elem_id=f"i2i_denoising_strength")
                    i2i_sampling_steps = gr.Slider(minimum=1,maximum=150,step=1,value=20,label="Sampling steps", elem_id=f"i2i_sampling_steps")
                    i2i_controlnet_strength = gr.Slider(minimum=0.0,maximum=1.0,step=0.05,value=0.15,label="controlnet strength", elem_id=f"i2i_controlnet_strength")
            i2i_style_presets.change(
                fn=set_parm_presets,
                inputs=[i2i_style_presets,i2i_checkpoints,i2i_vae],
                outputs=[i2i_checkpoints,i2i_vae,i2i_prompt_input,i2i_negativeprompt_input,i2i_width,i2i_height,i2i_cfg_scale,i2i_denoising_strength,i2i_sampling_steps]
                )
            i2i_submit.click(
                fn=start_i2i, 
                inputs=[i2i_style_presets,i2i_checkpoints,i2i_vae,i2i_prompt_input,i2i_negativeprompt_input,i2i_image_input,i2i_seed,i2i_width,i2i_height,i2i_cfg_scale,i2i_denoising_strength,i2i_sampling_steps,i2i_controlnet_strength], 
                outputs=i2i_image_output
                )
            # i2i_skip.click(
            #     fn=lambda: shared.state.skip(),
            #     inputs=[],
            #     outputs=[],
            #     )
            # i2i_interrupt.click(
            #     fn=lambda: shared.state.interrupt(),
            #     inputs=[],
            #     outputs=[],
            #     )
        # return demo_ui
        return [(demo_ui, "Style Converter", "style_converter_tab")]

script_callbacks.on_ui_tabs(create_UI)
# script_callbacks.on_ui_tabs([(create_UI(), "Style Converter", "style_converter_tab")])

# create_UI().launch(inbrowser=True)



