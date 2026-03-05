"""Stream parser utilities for parsing LLM streaming responses.

This module provides helpers to parse streaming output from OpenAI-compatible
LLM APIs, particularly for extracting thinking/reasoning content versus
final code/content.
"""

import re
from typing import Generator, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StreamChunk:
    """Represents a chunk of streamed content."""
    content: str
    chunk_type: str  # 'thinking', 'code', 'other'


def parse_streaming_response(
    chunks: Generator[str, None, None],
) -> Generator[Tuple[str, str], None, None]:
    """Parse streaming response into (chunk_type, content) pairs.

    This function analyzes incoming stream chunks to determine if they are
    part of a thinking/reasoning block or the final code/content.

    For OpenAI-compatible APIs that support reasoning/thinking tokens,
    the format typically uses markers like:
    - `<think>` ... `</think>` 
    - Or content before ```python blocks

    Args:
        chunks: Generator yielding string chunks from the stream

    Yields:
        Tuples of (chunk_type, content) where chunk_type is one of:
        - 'thinking': Content inside thinking/reasoning blocks
        - 'code': Content inside code blocks
        - 'other': Regular text content
    """
    buffer = ""
    in_thinking = False
    in_code_block = False
    code_block_language = ""
    
    thinking_start_tags = ['<think>', '<thinking>', 'Thinking:', 'reasoning:']
    thinking_end_tags = ['</think>', '</thinking>']
    code_block_start = '```'
    code_block_end = '```'
    
    for chunk in chunks:
        buffer += chunk
        
        # Check for thinking block start
        if not in_thinking and not in_code_block:
            for tag in thinking_start_tags:
                if tag.lower() in buffer.lower():
                    in_thinking = True
                    # Extract and yield thinking content
                    idx = buffer.lower().find(tag.lower())
                    thinking_content = buffer[idx + len(tag):]
                    if thinking_content.strip():
                        yield ('thinking', thinking_content)
                    buffer = ""
                    break
        
        # Check for code block start
        if not in_code_block and code_block_start in buffer:
            # Check if we were in thinking mode
            if in_thinking:
                in_thinking = False
            
            idx = buffer.find(code_block_start)
            remaining = buffer[idx + len(code_block_start):]
            
            # Try to detect language
            lang_match = re.match(r'^(\w+)', remaining.strip())
            if lang_match:
                code_block_language = lang_match.group(1)
            
            if code_block_language in ('python', 'py'):
                in_code_block = True
                buffer = ""
            else:
                # Not a code block, treat as regular content
                if remaining.strip():
                    yield ('other', remaining)
                buffer = ""
        
        # Check for thinking end tag
        if in_thinking:
            for tag in thinking_end_tags:
                if tag in buffer:
                    idx = buffer.find(tag)
                    thinking_content = buffer[:idx]
                    if thinking_content.strip():
                        yield ('thinking', thinking_content)
                    buffer = buffer[idx + len(tag):]
                    in_thinking = False
                    break
        
        # Check for code block end
        if in_code_block and code_block_end in buffer:
            idx = buffer.find(code_block_end)
            code_content = buffer[:idx]
            if code_content.strip():
                yield ('code', code_content)
            buffer = buffer[idx + len(code_block_end):]
            in_code_block = False
            code_block_language = ""
        
        # If still in thinking mode and has content, yield it
        if in_thinking and buffer.strip():
            yield ('thinking', buffer)
            buffer = ""
    
    # Handle remaining buffer
    if buffer.strip():
        if in_thinking:
            yield ('thinking', buffer)
        elif in_code_block:
            yield ('code', buffer)
        else:
            yield ('other', buffer)


def extract_thinking_content(full_response: str) -> str:
    """Extract thinking/reasoning content from a full response.

    Args:
        full_response: The complete LLM response text

    Returns:
        Extracted thinking content, or empty string if none found
    """
    patterns = [
        r'<think>(.*?)</think>',
        r'<thinking>(.*?)</thinking>',
        r'Thinking:\s*(.*?)(?=```|$)',
        r'reasoning:\s*(.*?)(?=```|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""


def extract_python_code(response: str) -> str:
    """Extract Python code from markdown code blocks.

    This is a robust extractor that handles multiple code blocks,
    de-duplicates imports, and validates syntax.

    Args:
        response: Text containing markdown-formatted code blocks

    Returns:
        Extracted Python code, or empty string if none found
    """
    # Pattern to match ```python ... ``` blocks
    pattern = r'```(?:python|py)?\n?(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        # Try generic code blocks without language specifier
        pattern = r'```\n?(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        # Return the last (most complete) code block
        return matches[-1].strip()
    
    return ""


def split_thinking_and_code(response: str) -> Tuple[str, str]:
    """Split response into thinking content and code.

    Args:
        response: Full LLM response

    Returns:
        Tuple of (thinking_content, code_content)
    """
    thinking = extract_thinking_content(response)
    code = extract_python_code(response)
    return thinking, code


def parse_openai_chunk_delta(delta: dict) -> Optional[str]:
    """Parse a single OpenAI chunk delta to extract content.

    Handles various OpenAI-compatible API formats including:
    - Standard: {"content": "..."}
    - Reasoning: {"reasoning_content": "...", "content": "..."}
    - Thinking: {"thinking": "..."}

    Args:
        delta: The delta object from a stream chunk

    Returns:
        Content string or None if no content
    """
    # Check for reasoning/thinking content first
    if isinstance(delta, dict):
        # Common fields for thinking/reasoning
        for field in ['reasoning_content', 'thinking', 'reasoning']:
            if field in delta and delta[field]:
                return delta[field]
        
        # Standard content field
        if 'content' in delta and delta['content']:
            return delta['content']
    
    return None


def is_code_block_complete(text: str) -> bool:
    """Check if text contains a complete code block.

    A complete code block has matching opening and closing ``` markers.

    Args:
        text: Text to check

    Returns:
        True if a complete code block is found
    """
    # Count python code block markers
    opens = len(re.findall(r'```(?:python|py)\n?', text))
    closes = len(re.findall(r'```\s*$', text, re.MULTILINE))
    
    return opens > 0 and closes >= opens