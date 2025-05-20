# PDF2Audio: Technical Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Application Overview](#application-overview)
3. [File Structure](#file-structure)
4. [Core Components](#core-components)
   - [Data Models](#data-models)
   - [PDF Processing](#pdf-processing)
   - [Text Generation](#text-generation)
   - [Audio Generation](#audio-generation)
   - [Instruction Templates](#instruction-templates)
5. [User Interface](#user-interface)
   - [Main Layout](#main-layout)
   - [Input Controls](#input-controls)
   - [Output Display](#output-display)
   - [Editing Features](#editing-features)
6. [Workflow](#workflow)
7. [Key Functions](#key-functions)
8. [Integration Points](#integration-points)
9. [Customization Options](#customization-options)
10. [Conclusion](#conclusion)

## Introduction

PDF2Audio is a Gradio-based web application that converts PDF documents, markdown files, and text files into audio content using OpenAI's GPT models for text generation and text-to-speech (TTS) services. The application allows users to upload documents, select from various instruction templates (podcast, lecture, summary, etc.), and customize the output with different voices and models.

This technical documentation provides a detailed explanation of the `app.py` file, which contains all the functionality of the PDF2Audio application. It is designed to help developers and designers understand the codebase to use it as a foundation for similar applications.

## Application Overview

PDF2Audio follows a straightforward workflow:

1. User uploads one or more PDF, markdown, or text files
2. User selects an instruction template and customizes settings
3. The application extracts text from the uploaded files
4. An LLM (Language Learning Model) processes the text according to the selected template
5. The generated dialogue is converted to audio using OpenAI's TTS service
6. The user can listen to the audio, view the transcript, edit it, and regenerate if needed

The application is built using the Gradio framework, which provides an easy-to-use interface for creating web applications with Python. The backend leverages OpenAI's API for both text generation and text-to-speech conversion.

## File Structure

The entire application is contained within a single `app.py` file, which includes:

- Import statements for required libraries
- Data model definitions
- Instruction templates for different output formats
- Core functionality for text extraction, dialogue generation, and audio synthesis
- Gradio UI components and layout
- Event handlers for user interactions

## Core Components

### Data Models

The application uses Pydantic models to structure the dialogue data:

```python
class DialogueItem(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2"]

class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]
```

These models ensure type safety and provide a structured way to handle the dialogue data throughout the application.

### PDF Processing

PDF processing is handled using the PyPDF library. The application extracts text from uploaded PDF files:

```python
if suffix == ".pdf":
    with file_path.open("rb") as f:
        reader = PdfReader(f)
        text = "\n\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
        combined_text += text + "\n\n"
```

The application also supports markdown and plain text files:

```python
elif suffix in [".txt", ".md", ".mmd"]:
    with file_path.open("r", encoding="utf-8") as f:
        text = f.read()
        combined_text += text + "\n\n"
```

### Text Generation

Text generation is performed using OpenAI's GPT models through the `promptic` library's `llm` decorator. The application uses a custom `conditional_llm` wrapper to dynamically configure the LLM based on user selections:

```python
@conditional_llm(
    model=text_model,
    api_base=api_base,
    api_key=openai_api_key,
    reasoning_effort=reasoning_effort,
    do_web_search=do_web_search,           
)
def generate_dialogue(text: str, intro_instructions: str, text_instructions: str, 
                     scratch_pad_instructions: str, prelude_dialog: str, 
                     podcast_dialog_instructions: str,
                     edited_transcript: str = None, user_feedback: str = None) -> Dialogue:
    # Function body contains the prompt template
```

The `generate_dialogue` function is decorated with `@retry` to handle validation errors and retry the API call if necessary.

### Audio Generation

Audio generation is handled by OpenAI's TTS API through the `get_mp3` function:

```python
def get_mp3(text: str, voice: str, audio_model: str, api_key: str = None,
           speaker_instructions: str ='Speak in an emotive and friendly tone.') -> bytes:
    
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )
 
    with client.audio.speech.with_streaming_response.create(
        model=audio_model,
        voice=voice,
        input=text,
        instructions=speaker_instructions,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()
```

The application uses `concurrent.futures.ThreadPoolExecutor` to parallelize audio generation for each dialogue line, improving performance:

```python
with cf.ThreadPoolExecutor() as executor:
    futures = []
    for line in llm_output.dialogue:
        transcript_line = f"{line.speaker}: {line.text}"
        voice = speaker_1_voice if line.speaker == "speaker-1" else speaker_2_voice
        speaker_instructions=speaker_1_instructions if line.speaker == "speaker-1" else speaker_2_instructions
        future = executor.submit(get_mp3, line.text, voice, audio_model, openai_api_key, speaker_instructions)
        futures.append((future, transcript_line))
        characters += len(line.text)

    for future, transcript_line in futures:
        audio_chunk = future.result()
        audio += audio_chunk
        transcript += transcript_line + "\n\n"
```

### Instruction Templates

The application includes a comprehensive set of instruction templates for different output formats:

```python
INSTRUCTION_TEMPLATES = {
    "podcast": { ... },
    "deep research analysis": { ... },
    "clean rendering": { ... },
    "SciAgents material discovery summary": { ... },
    "lecture": { ... },
    "summary": { ... },
    "short summary": { ... },
    "podcast (French)": { ... },
    "podcast (German)": { ... },
    "podcast (Spanish)": { ... },
    "podcast (Portuguese)": { ... },
    "podcast (Hindi)": { ... },
    "podcast (Chinese)": { ... },
}
```

Each template contains five key components:
1. `intro`: High-level task description
2. `text_instructions`: How to process the input text
3. `scratch_pad`: Hidden brainstorming area for the model
4. `prelude`: Introduction to the main output
5. `dialog`: Main output instructions

These templates guide the LLM in generating appropriate content based on the selected format.

## User Interface

### Main Layout

The Gradio UI is structured with a clean, responsive layout:

```python
with gr.Blocks(title="PDF to Audio", css="""
    #header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px;
        background-color: transparent;
        border-bottom: 1px solid #ddd;
    }
    /* Additional CSS styles */
""") as demo:
    
    cached_dialogue = gr.State()
    
    with gr.Row(elem_id="header"):
        # Header content
    
    with gr.Row(elem_id="main_container"):
        with gr.Column(scale=2):
            # Input controls
        with gr.Column(scale=3):
            # Template selection and customization
    
    # Output components
    # Editing features
```

### Input Controls

The left column contains input controls for file uploading and model selection:

```python
files = gr.Files(label="PDFs (.pdf), markdown (.md, .mmd), or text files (.txt)", 
                file_types=[".pdf", ".PDF", ".md", ".mmd", ".txt"])

openai_api_key = gr.Textbox(
    label="OpenAI API Key",
    visible=True,
    placeholder="Enter your OpenAI API Key here...",
    type="password"
)

text_model = gr.Dropdown(
    label="Text Generation Model",
    choices=STANDARD_TEXT_MODELS,
    value="o3-mini",
    info="Select the model to generate the dialogue text."
)

# Additional input controls for audio model, voices, etc.
```

### Output Display

The application provides several output components:

```python
audio_output = gr.Audio(label="Audio", format="mp3", interactive=False, autoplay=False)
transcript_output = gr.Textbox(label="Transcript", lines=25, show_copy_button=True)
original_text_output = gr.Textbox(label="Original Text", lines=10, visible=False)
error_output = gr.Textbox(visible=False)  # Hidden textbox to store error message
```

### Editing Features

The application includes several features for editing and regenerating content:

1. **Transcript Editing**:
   ```python
   use_edited_transcript = gr.Checkbox(label="Use Edited Transcript", value=False)
   edited_transcript = gr.Textbox(label="Edit Transcript Here", lines=20, visible=False,
                                show_copy_button=True, interactive=False)
   ```

2. **Line-by-Line Editing**:
   ```python
   with gr.Accordion("Edit dialogue line‑by‑line", open=False) as editor_box:
       df_editor = gr.Dataframe(
           headers=["Speaker", "Line"],
           datatype=["str", "str"],
           wrap=True,
           interactive=True,
           row_count=(1, "dynamic"),
           col_count=(2, "fixed"),
       )
   ```

3. **User Feedback**:
   ```python
   user_feedback = gr.Textbox(label="Provide Feedback or Notes", lines=10)
   ```

## Workflow

The application workflow is managed through event handlers that connect UI components to backend functions:

1. **Template Selection**:
   ```python
   template_dropdown.change(
       fn=update_instructions,
       inputs=[template_dropdown],
       outputs=[intro_instructions, text_instructions, scratch_pad_instructions, 
                prelude_dialog, podcast_dialog_instructions]
   )
   ```

2. **Generate Audio**:
   ```python
   submit_btn.click(
       fn=validate_and_generate_audio,
       inputs=[
           files, openai_api_key, text_model, reasoning_effort, do_web_search, audio_model, 
           speaker_1_voice, speaker_2_voice, speaker_1_instructions, speaker_2_instructions,
           api_base, intro_instructions, text_instructions, scratch_pad_instructions, 
           prelude_dialog, podcast_dialog_instructions, edited_transcript, user_feedback,
       ],
       outputs=[audio_output, transcript_output, original_text_output, error_output, cached_dialogue]
   )
   ```

3. **Regenerate with Edits**:
   ```python
   regenerate_btn.click(
       fn=lambda use_edit, edit, *args: validate_and_generate_audio(
           *args[:12],  # All inputs up to podcast_dialog_instructions
           edit if use_edit else "",  # Use edited transcript if checkbox is checked
           *args[12:]  # user_feedback and original_text_output
       ),
       inputs=[
           use_edited_transcript, edited_transcript,
           # Additional inputs
       ],
       outputs=[audio_output, transcript_output, original_text_output, error_output, cached_dialogue]
   )
   ```

4. **Re-render Audio**:
   ```python
   rerender_btn.click(
       fn=render_audio_from_dialogue,
       inputs=[
           cached_dialogue, openai_api_key, audio_model,
           speaker_1_voice, speaker_2_voice, speaker_1_instructions, speaker_2_instructions,
       ],
       outputs=[audio_output, transcript_output],
   )
   ```

## Key Functions

### `validate_and_generate_audio`

This function serves as the entry point for audio generation, validating inputs and handling errors:

```python
def validate_and_generate_audio(*args):
    files = args[0]
    if not files:
        return None, None, None, "Please upload at least one PDF (or MD/MMD/TXT) file before generating audio."
    try:
        audio_file, transcript, original_text, dialogue = generate_audio(*args)
        return audio_file, transcript, original_text, None, dialogue
    except Exception as e:
        return None, None, None, str(e), None
```

### `generate_audio`

This is the core function that orchestrates the entire process:

1. Validates the API key
2. Extracts text from uploaded files
3. Configures and calls the LLM to generate dialogue
4. Processes any user edits or feedback
5. Generates audio for each dialogue line
6. Returns the audio file, transcript, and original text

### `render_audio_from_dialogue`

This function re-renders audio from an existing dialogue without regenerating the text:

```python
def render_audio_from_dialogue(
    cached_dialogue,
    openai_api_key: str,
    audio_model: str,
    speaker_1_voice: str,
    speaker_2_voice: str,
    speaker_1_instructions: str,
    speaker_2_instructions: str,
) -> tuple[str, str]:
    # Function implementation
```

### `save_dialogue_edits`

This function saves edits made in the dataframe editor:

```python
def save_dialogue_edits(df, cached_dialogue):
    if cached_dialogue is None:
        raise gr.Error("Nothing to edit yet – run Generate Audio first.")

    import pandas as pd
    new_dlg = df_to_dialogue(pd.DataFrame(df, columns=["Speaker", "Line"]))

    # regenerate plain transcript so the user sees the change immediately
    transcript_str = "\n".join(f"{d.speaker}: {d.text}" for d in new_dlg.dialogue)

    # Return updated state and transcript
    return new_dlg, gr.update(value=transcript_str), "Edits saved. Press *Re‑render* to hear them."
```

## Integration Points

The application integrates with several external services and libraries:

1. **OpenAI API**: Used for both text generation (GPT models) and text-to-speech conversion
2. **Promptic**: A library for working with LLM prompts
3. **PyPDF**: Used for extracting text from PDF files
4. **Gradio**: The web UI framework
5. **Pydantic**: Used for data validation and modeling
6. **Tenacity**: Used for implementing retry logic

## Customization Options

The application offers several customization options:

1. **Instruction Templates**: Multiple pre-defined templates for different output formats
2. **Model Selection**: Support for various OpenAI models for both text and audio generation
3. **Voice Selection**: Multiple voice options for the speakers
4. **Voice Instructions**: Custom instructions for each speaker's voice
5. **API Base**: Option to use a custom API endpoint for text generation
6. **Web Search**: Option to enable web search during text generation
7. **Reasoning Effort**: Control over the reasoning effort for compatible models

## Conclusion

PDF2Audio is a well-structured application that demonstrates effective use of modern AI APIs for content transformation. Its modular design and comprehensive feature set make it an excellent foundation for similar applications.

Key strengths of the codebase include:

1. **Modularity**: Clear separation of concerns between text extraction, dialogue generation, and audio synthesis
2. **Extensibility**: Easy to add new instruction templates or customize existing ones
3. **Error Handling**: Robust error handling with informative user feedback
4. **Performance Optimization**: Parallel processing for audio generation
5. **User Experience**: Rich UI with multiple editing and customization options

Developers looking to build similar applications can leverage this codebase as a starting point, focusing on extending functionality or improving specific aspects rather than building from scratch.
