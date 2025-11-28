import numpy as np
# from openai import OpenAI
import json
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import time
import torch

import os
import time
# --- NEW IMPORTS ---
from google import genai
from google.genai.errors import APIError
# -------------------

def correct_indentation(code_str):
    lines = code_str.split("\n")
    tabbed_lines = ["\t" + line for line in lines]
    tabbed_text = "\n".join(tabbed_lines)
    return tabbed_text


def replace_tabs_with_spaces(code_str):
    return code_str.replace("\t", "    ")


def untab(text):
    lines = text.split("\n")
    untabbed_lines = []
    for line in lines:
        if line.startswith("\t"):
            untabbed_lines.append(line[1:])
        elif line.startswith("    "):
            untabbed_lines.append(line[4:])
        else:
            untabbed_lines.append(line)
    untabbed_text = "\n".join(untabbed_lines)
    return untabbed_text


def get_methods_from_json(api_json):
    methods = []
    namespace = {}
    for method_info in api_json:

        signature = method_info["signature"]
        if "def" in method_info["implementation"]:
            full_method = method_info["implementation"]
        else:
            full_method = signature + "\n" + method_info["implementation"]
        methods.append(full_method)

    return methods, namespace


# class Generator:
#     def __init__(self, model_name="gpt-4o", temperature=0.7, api_key_path="./api.key"):
#         self.temperature = temperature
#         self.model_name = model_name
#         self.api_key_path = api_key_path
#         with open(self.api_key_path, "r") as file:
#             self.client = OpenAI(api_key=file.read().strip())

#     def remove_substring(self, output, substring):

#         if substring in output:
#             return output.replace(substring, "")
#         else:
#             return output

#     def generate(self, prompt, messages=None):
#         new_messages = None
#         if self.model_name == "gpt-3.5-turbo-instruct":
#             response = self.client.Completion.create(
#                 model=self.model_name,
#                 prompt=prompt,
#                 temperature=self.temperature,
#                 max_tokens=1024,
#                 top_p=0.5,
#                 frequency_penalty=0,
#                 presence_penalty=0,
#                 n=1,
#                 logprobs=11,
#             )

#             result = response.choices[0]["text"].lstrip("\n").rstrip("\n")
#             result = self.remove_substring(result, "```python")
#             result = self.remove_substring(result, "```")

#         else:
#             if not messages:
#                 messages = [{"role": "user", "content": prompt}]
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=messages,
#                     temperature=self.temperature,
#                 )
#             except Exception as e:
#                 time.sleep(60)
#                 return self.generate(prompt, messages)
#             new_messages = messages
#             result = response.choices[0].message.content.lstrip("\n").rstrip("\n")
#             result = self.remove_substring(result, "```python")
#             result = self.remove_substring(result, "```")
#             new_messages.append(
#                 {
#                     "role": response.choices[0].message.role,
#                     "content": result,
#                 }
#             )
#         return result, new_messages

class Generator:
    # Set a Gemini model as the new default
    def __init__(self, model_name="gemini-2.5-flash", temperature=0.7, api_key_path="./api.key"):
        self.temperature = temperature
        self.model_name = model_name
        self.api_key_path = api_key_path
        
        # Load API key from file and set environment variable for the client
        try:
            with open(self.api_key_path, "r") as file:
                api_key = file.read().strip()
            # **CRITICAL**: Set the environment variable for the Google client
            os.environ["GEMINI_API_KEY"] = api_key 
            # self.client = genai.Client()
            self.client = genai.Client(api_key=api_key)
        except FileNotFoundError:
            raise ValueError(f"API key file not found at {api_key_path}. Please create it.")

    def remove_substring(self, output, substring):
        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def generate(self, prompt, messages=None):
        new_messages = None

        try:
            # 1. Build `contents` in the format expected by google-genai
            if messages is None or len(messages) == 0:
                # Simple one-shot text call
                text = "" if prompt is None else str(prompt)
                contents = [genai.types.Part(text=text)]
            else:
                contents = []
                for msg in messages:
                    role = "user" if msg.get("role") == "user" else "model"
                    parts = []

                    content = msg.get("content", "")

                    # Case 1: content is a list of pieces (e.g. [{"type": "text", ...}, {"type": "image_url", ...}])
                    if isinstance(content, list):
                        for piece in content:
                            # Plain string
                            if isinstance(piece, str):
                                parts.append(genai.types.Part(text=piece))

                            # {"type": "text", "text": "..."}
                            elif isinstance(piece, dict) and piece.get("type") == "text":
                                parts.append(
                                    genai.types.Part(text=str(piece.get("text", "")))
                                )

                            # {"type": "image_url", "image_url": {"url": "..."}}
                            elif isinstance(piece, dict) and piece.get("type") == "image_url":
                                img_spec = piece.get("image_url", {})
                                url = img_spec.get("url", "")

                                if url.startswith("data:"):
                                    # data URL like "data:image/png;base64,AAAA..."
                                    try:
                                        header, b64data = url.split(",", 1)
                                        # "data:image/png;base64" -> "image/png"
                                        mime_type = header.split(";")[0].split(":")[1]
                                        img_bytes = base64.b64decode(b64data)
                                        parts.append(
                                            genai.types.Part.from_bytes(
                                                data=img_bytes,
                                                mime_type=mime_type,
                                            )
                                        )
                                    except Exception as e:
                                        # Fallback: if something goes wrong, just send it as text
                                        parts.append(genai.types.Part(text=str(piece)))
                                else:
                                    # Remote URL – best-effort
                                    parts.append(
                                        genai.types.Part.from_uri(
                                            file_uri=url,
                                            mime_type="image/png",
                                        )
                                    )
                            else:
                                # Unknown structure – just stringify
                                parts.append(genai.types.Part(text=str(piece)))
                    else:
                        # Case 2: content is just a string or something simple
                        parts.append(genai.types.Part(text=str(content)))

                    # Wrap role + parts into a Content object
                    contents.append(
                        genai.types.Content(
                            role=role,
                            parts=parts,
                        )
                    )

            # 2. Configuration (temperature etc.)
            config = genai.types.GenerateContentConfig(
                temperature=self.temperature
            )

            # 3. Make the API call
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

        except APIError as e:
            print(f"Gemini API Error: {e}. Retrying after 60 seconds...")
            time.sleep(60)
            return self.generate(prompt, messages)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

        # 4. Process result
        result = response.text.lstrip("\n").rstrip("\n")
        result = self.remove_substring(result, "```python")
        result = self.remove_substring(result, "```")

        # 5. Update history if chat messages were provided
        if messages is not None:
            new_messages = messages
            new_messages.append(
                {
                    "role": "assistant",
                    "content": result,
                }
            )

        return result, new_messages


def docstring_from_json(json_file):
    with open(json_file, "r") as file:
        api_data = json.load(file)

    docstring = ""
    for module in api_data.get("modules", []):
        docstring += f'"""\n'
        docstring += f"{module['description']}\n\n"
        if module["arguments"]:
            docstring += "Args:\n"
            for arg in module["arguments"]:
                docstring += (
                    f"    {arg['name']} ({arg['type']}): {arg['description']}\n"
                )
        if "returns" in module:
            docstring += f"\nReturns:\n"
            docstring += f"    {module['returns']['type']}: {module['returns']['description']}\n\"\"\""
        docstring += f"\n{module['name']}("
        args = [arg["name"] for arg in module["arguments"]]
        docstring += ", ".join(args) + ")\n\n"

    return docstring.strip()

def depth_to_grayscale(depth_map):
    # Ensure depth_map is a NumPy array of type float (if not already)
    depth_map = np.array(depth_map, dtype=np.float32)

    # Get the minimum and maximum depth values
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    
    # Avoid division by zero if the image is constant
    if d_max - d_min == 0:
        normalized = np.zeros_like(depth_map)
    else:
        normalized = (depth_map - d_min) / (d_max - d_min)
    
    # Scale to 0-255 and convert to unsigned 8-bit integer
    grayscale = (normalized * 255).astype(np.uint8)
    
    return grayscale


def box_image(img, boxes):
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for box in boxes:
        x_0, y_0, x_1, y_1 = box[0], box[1], box[2], box[3]
        draw.rectangle([x_0, y_0, x_1, y_1], outline="red", width=8)

    return img1


def dotted_image(img, points):
    # Scale dot size based on image width
    if isinstance(img, np.ndarray):
        img_width = img.shape[1]
        np_img = img.copy()
        img = Image.fromarray(np_img)
        if img.mode == 'F':
            img = depth_to_grayscale(np_img)
            img = Image.fromarray(img)
            img = img.convert('RGB')
    else:
        img_width = img.size[0]

    
    dot_size = int(img_width * 0.02) # 2% of image width
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for pt in points:
        x = pt[0]
        y = pt[1]

        draw.ellipse(
            (x - dot_size, y - dot_size, x + dot_size, y + dot_size),
            fill="red",
            outline="black",
        )
    return img1


def html_embed_image(img, size=300):
    img = img.copy()
    img.thumbnail((size, size))
    with BytesIO() as buffer:
        if img.mode == 'F':
            img = img.convert('RGB')
        img.save(buffer, "jpeg")
        base64_img = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<img style="vertical-align:middle" src="data:image/jpeg;base64,{base64_img}">'
    )
    

class TimeoutException(Exception):
    pass


def set_devices():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device


def timeout_handler(signum, frame):
    raise TimeoutException(
        "The script took too long to run. There is likely a Recursion Error. Ensure that you are not calling a method with infinite recursion."
    )


def remove_python_text(output):
    substring = "```python"
    if substring in output:
        output = output.replace(substring, "")

    substring = "```"
    if substring in output:
        output = output.replace(substring, "")

    return output
