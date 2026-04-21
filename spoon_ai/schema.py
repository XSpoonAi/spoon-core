import json
from enum import Enum
from typing import Any, List, Literal, Optional, Union
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class Function(BaseModel):
    name: str
    arguments: str
    
    def get_arguments_dict(self) -> dict:
        """Parse arguments string to dictionary.
        
        Returns:
            dict: Parsed arguments as dictionary
        """
        if isinstance(self.arguments, str):
            arguments = self.arguments.strip()
            if not arguments:
                return {}
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {}
        elif isinstance(self.arguments, dict):
            return self.arguments
        else:
            return {}
    
    @classmethod
    def create(cls, name: str, arguments: Union[str, dict]) -> "Function":
        """Create Function with arguments as string or dict.
        
        Args:
            name: Function name
            arguments: Function arguments as string or dict
            
        Returns:
            Function: Function instance with arguments as JSON string
        """
        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments)
        else:
            arguments_str = str(arguments)
        
        return cls(name=name, arguments=arguments_str)

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Function
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional provider-specific metadata for preserving tool-call context",
    )

class AgentState(str, Enum):
    """
    The state of the agent.
    """
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

class ToolChoice(str, Enum):
    """Tool choice options"""
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class Role(str, Enum):
    """Message role options"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


# =============================================================================
# Multimodal Content Types - Native Vision Support
# =============================================================================

class ContentType(str, Enum):
    """Types of content blocks for multimodal messages"""
    TEXT = "text"
    IMAGE = "image"          # Base64 encoded image data
    IMAGE_URL = "image_url"  # URL reference to image
    DOCUMENT = "document"    # PDF and other documents (base64)
    FILE = "file"            # File attachment (path-based)
    AUDIO = "audio"          # Audio content (future)


class ImageMediaType(str, Enum):
    """Supported image media types"""
    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    WEBP = "image/webp"


class ImageSource(BaseModel):
    """Source for base64-encoded image content (Anthropic style)"""
    type: Literal["base64"] = "base64"
    media_type: str = Field(..., description="MIME type: image/jpeg, image/png, image/gif, image/webp")
    data: str = Field(..., description="Base64-encoded image data")


class ImageUrlSource(BaseModel):
    """Source for URL-based image content (OpenAI style)"""
    url: str = Field(..., description="URL of the image or base64 data URL")
    detail: Optional[Literal["auto", "low", "high"]] = Field(
        default="auto",
        description="Image detail level for processing"
    )


class TextContent(BaseModel):
    """Text content block"""
    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content")


class ImageContent(BaseModel):
    """Image content block with base64 data (Anthropic-compatible)"""
    type: Literal["image"] = "image"
    source: ImageSource = Field(..., description="Base64 image source")


class ImageUrlContent(BaseModel):
    """Image content block with URL reference (OpenAI-compatible)"""
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlSource = Field(..., description="Image URL source")


class FileContent(BaseModel):
    """File content block (path-based, for local file references)"""
    type: Literal["file"] = "file"
    file_path: str = Field(..., description="Path to the file")
    media_type: Optional[str] = Field(default=None, description="MIME type of the file")


class DocumentSource(BaseModel):
    """Source for base64-encoded document content (PDF, etc.)"""
    type: Literal["base64"] = "base64"
    media_type: str = Field(..., description="MIME type: application/pdf, text/plain, etc.")
    data: str = Field(..., description="Base64-encoded document data")


class DocumentContent(BaseModel):
    """Document content block for PDFs and other documents (Anthropic/Gemini compatible)

    Supported by:
    - Anthropic Claude: Native PDF support via base64
    - Gemini: Native PDF support via inline_data
    - OpenAI: NOT supported (will be converted to text placeholder)
    """
    type: Literal["document"] = "document"
    source: DocumentSource = Field(..., description="Base64 document source")
    filename: Optional[str] = Field(default=None, description="Optional filename for display")


# Union of all content block types
ContentBlock = Union[TextContent, ImageContent, ImageUrlContent, DocumentContent, FileContent]

# Message content can be either a simple string or a list of content blocks
MessageContent = Union[str, List[ContentBlock]]


class Message(BaseModel):
    """Represents a chat message in the conversation.

    Supports both text-only and multimodal content:
    - Simple text: content="Hello world"
    - Multimodal: content=[TextContent(...), ImageUrlContent(...)]
    """

    id: Optional[str] = Field(default=None)
    role: ROLE_TYPE = Field(...) # type: ignore
    content: Optional[MessageContent] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)

    @property
    def is_multimodal(self) -> bool:
        """Check if this message contains multimodal content."""
        return isinstance(self.content, list)

    @property
    def text_content(self) -> str:
        """Extract text content from message (for backward compatibility).

        Returns:
            str: Combined text content from all text blocks, or empty string
        """
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # Extract text from content blocks
        texts = []
        for block in self.content:
            if isinstance(block, TextContent):
                texts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)

    @property
    def has_images(self) -> bool:
        """Check if message contains any image content."""
        if not isinstance(self.content, list):
            return False
        return any(
            isinstance(block, (ImageContent, ImageUrlContent)) or
            (isinstance(block, dict) and block.get("type") in ("image", "image_url"))
            for block in self.content
        )

    @property
    def has_documents(self) -> bool:
        """Check if message contains any document content (PDF, etc.)."""
        if not isinstance(self.content, list):
            return False
        return any(
            isinstance(block, DocumentContent) or
            (isinstance(block, dict) and block.get("type") == "document")
            for block in self.content
        )

    @classmethod
    def create_text(cls, role: str, text: str, **kwargs) -> "Message":
        """Create a simple text message.

        Args:
            role: Message role (user, assistant, system, tool)
            text: Text content
            **kwargs: Additional message fields

        Returns:
            Message: Text message instance
        """
        return cls(role=role, content=text, **kwargs)

    @classmethod
    def create_multimodal(
        cls,
        role: str,
        content_blocks: List[ContentBlock],
        **kwargs
    ) -> "Message":
        """Create a multimodal message with mixed content types.

        Args:
            role: Message role (user, assistant, system, tool)
            content_blocks: List of content blocks (TextContent, ImageContent, etc.)
            **kwargs: Additional message fields

        Returns:
            Message: Multimodal message instance
        """
        return cls(role=role, content=content_blocks, **kwargs)

    @classmethod
    def create_with_image_url(
        cls,
        role: str,
        text: str,
        image_url: str,
        detail: Literal["auto", "low", "high"] = "auto",
        **kwargs
    ) -> "Message":
        """Create a message with text and an image URL.

        Args:
            role: Message role
            text: Text content
            image_url: URL of the image
            detail: Image detail level (auto, low, high)
            **kwargs: Additional message fields

        Returns:
            Message: Multimodal message with text and image URL
        """
        content_blocks: List[ContentBlock] = [
            TextContent(text=text),
            ImageUrlContent(image_url=ImageUrlSource(url=image_url, detail=detail))
        ]
        return cls(role=role, content=content_blocks, **kwargs)

    @classmethod
    def create_with_base64_image(
        cls,
        role: str,
        text: str,
        image_data: str,
        media_type: str = "image/png",
        **kwargs
    ) -> "Message":
        """Create a message with text and a base64-encoded image.

        Args:
            role: Message role
            text: Text content
            image_data: Base64-encoded image data
            media_type: Image MIME type (image/jpeg, image/png, etc.)
            **kwargs: Additional message fields

        Returns:
            Message: Multimodal message with text and base64 image
        """
        content_blocks: List[ContentBlock] = [
            TextContent(text=text),
            ImageContent(source=ImageSource(
                type="base64",
                media_type=media_type,
                data=image_data
            ))
        ]
        return cls(role=role, content=content_blocks, **kwargs)

    @classmethod
    def create_with_pdf(
        cls,
        role: str,
        text: str,
        pdf_data: str,
        filename: Optional[str] = None,
        **kwargs
    ) -> "Message":
        """Create a message with text and a base64-encoded PDF document.

        Supported by Anthropic Claude and Gemini. OpenAI does not support PDFs.

        Args:
            role: Message role
            text: Text content / question about the PDF
            pdf_data: Base64-encoded PDF data
            filename: Optional filename for display
            **kwargs: Additional message fields

        Returns:
            Message: Multimodal message with text and PDF document
        """
        content_blocks: List[ContentBlock] = [
            TextContent(text=text),
            DocumentContent(
                source=DocumentSource(
                    type="base64",
                    media_type="application/pdf",
                    data=pdf_data
                ),
                filename=filename
            )
        ]
        return cls(role=role, content=content_blocks, **kwargs)

    @classmethod
    def create_with_document(
        cls,
        role: str,
        text: str,
        document_data: str,
        media_type: str = "application/pdf",
        filename: Optional[str] = None,
        **kwargs
    ) -> "Message":
        """Create a message with text and a base64-encoded document.

        Supported document types vary by provider:
        - Anthropic: PDF
        - Gemini: PDF, and many other formats
        - OpenAI: NOT supported (will show placeholder)

        Args:
            role: Message role
            text: Text content / question about the document
            document_data: Base64-encoded document data
            media_type: Document MIME type (default: application/pdf)
            filename: Optional filename for display
            **kwargs: Additional message fields

        Returns:
            Message: Multimodal message with text and document
        """
        content_blocks: List[ContentBlock] = [
            TextContent(text=text),
            DocumentContent(
                source=DocumentSource(
                    type="base64",
                    media_type=media_type,
                    data=document_data
                ),
                filename=filename
            )
        ]
        return cls(role=role, content=content_blocks, **kwargs)


class SystemMessage(Message):
    role: ROLE_TYPE = Field(default=Role.SYSTEM.value)  # type: ignore

TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES] # type: ignore

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    model: str = ""
    api_key: str = ""
    base_url: Optional[str] = None
    api_type: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.3

class LLMResponse(BaseModel):
    """Unified LLM response model"""
    content: str
    text: str = ""  # Original text response
    image_paths: List[dict] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    finish_reason: Optional[str] = Field(default=None)
    native_finish_reason: Optional[str] = Field(default=None)

class LLMResponseChunk(BaseModel):
    """Enhanced LLM streaming response chunk."""
    
    # Core content
    content: str = Field(..., description="Accumulated content so far")
    delta: str = Field(..., description="Incremental content in this chunk")
    
    # Provider information
    provider: str = Field(..., description="Provider name")
    model: str = Field(..., description="Model name")
    
    # Completion information
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason for completion: 'stop', 'length', 'tool_calls', or None if ongoing"
    )
    
    # Tool calls
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Accumulated tool calls"
    )
    tool_call_chunks: Optional[List[dict]] = Field(
        default=None,
        description="Incremental tool call data (provider-specific)"
    )
    
    # Usage statistics (usually in final chunk)
    usage: Optional[dict] = Field(
        default=None,
        description="Token usage: {prompt_tokens, completion_tokens, total_tokens}"
    )
    
    # Additional metadata
    metadata: dict = Field(
        default_factory=dict,
        description="Provider-specific metadata"
    )
    chunk_index: int = Field(
        default=0,
        description="Index of this chunk (0-based)"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO format timestamp"
    )
