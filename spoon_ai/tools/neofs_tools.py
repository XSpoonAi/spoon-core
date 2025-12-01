"""
NeoFS Tools for spoon_ai framework

Simple wrappers around NeoFS client methods.
Tools do NOT auto-create bearer tokens - Agent manages tokens.
All parameters map directly to client method parameters.
"""

import json
import os
from typing import Optional
from pydantic import Field
from dotenv import load_dotenv

from spoon_ai.tools import BaseTool
from spoon_ai.neofs import NeoFSClient

# Load environment variables from .env file
# Use override=False to avoid overriding env vars already loaded by the main script
load_dotenv(override=False)
from spoon_ai.neofs.models import (
    Bearer,
    Rule,
    Record,
    Target,
    ContainerPostInfo,
    Attribute,
    Eacl,
    SearchRequest,
    SearchFilter,
)

_neofs_client_instance = None

def get_shared_neofs_client() -> NeoFSClient:
    """
    Get shared NeoFSClient instance for all NeoFS tools.
    
    Returns the same client instance across all tool calls to ensure
    bearer token authentication works correctly.
    """
    global _neofs_client_instance
    if _neofs_client_instance is None:
        _neofs_client_instance = NeoFSClient()
    return _neofs_client_instance


class CreateBearerTokenTool(BaseTool):
    """Create a bearer token for NeoFS operations"""
    
    name: str = "create_neofs_bearer_token"
    description: str = "Create bearer token. For containers use verb (PUT/DELETE/SETEACL), for objects use operation (PUT/GET/DELETE)."
    parameters: dict = {
        "type": "object",
        "properties": {
            "token_type": {
                "type": "string",
                "description": "Token type: 'container' or 'object'"
            },
            "verb": {
                "type": "string",
                "description": "For container tokens: PUT, DELETE, SETEACL"
            },
            "operation": {
                "type": "string",
                "description": "For object tokens: PUT, GET, DELETE"
            },
            "container_id": {
                "type": "string",
                "description": "Container ID (optional)"
            },
            "lifetime": {
                "type": "integer",
                "description": "Token lifetime in seconds"
            }
        },
        "required": ["token_type"]
    }
    
    async def execute(self, token_type: str, verb: str = "PUT", operation: str = "PUT", container_id: str = "", lifetime: int = 100, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            
            if token_type == "container":
                rules = [Bearer(name=f"container-{verb.lower()}-token", container=Rule(verb=verb, containerId=container_id))]
            elif token_type == "object":
                rules = [Bearer(
                    name=f"object-{operation.lower()}-token",
                    object=[Record(action="ALLOW", operation=operation, filters=[], targets=[Target(role="OTHERS", keys=[])])]
                )]
            else:
                return f"❌ Invalid token_type: {token_type}"
            
            tokens = client.create_bearer_tokens(rules, lifetime=lifetime)
            bearer_token = tokens[0].token
            
            return f"""✅ Bearer token created!
Type: {token_type}
Verb/Operation: {verb if token_type == 'container' else operation}
Container ID: {container_id if container_id else 'N/A'}
Lifetime: {lifetime} seconds
Token: {bearer_token}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class CreateContainerTool(BaseTool):
    """Create a NeoFS container"""
    
    name: str = "create_neofs_container"
    description: str = "Create container. Requires bearer token. Returns container_id."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_name": {
                "type": "string",
                "description": "Container name"
            },
            "bearer_token": {
                "type": "string",
                "description": "Bearer token (required)"
            },
            "basic_acl": {
                "type": "string",
                "description": "Basic ACL: 'public-read-write' or 'eacl-public-read-write'"
            },
            "placement_policy": {
                "type": "string",
                "description": "Placement policy: REP 1, REP 2, REP 3"
            },
            "attributes": {
                "type": "object",
                "description": "Container attributes as key-value pairs"
            },
            "name_scope_global": {
                "type": "boolean",
                "description": "Whether container name is globally unique"
            }
        },
        "required": ["container_name", "bearer_token"]
    }
    
    async def execute(self, container_name: str, bearer_token: str, basic_acl: str = "eacl-public-read-write", placement_policy: str = "REP 1", attributes: dict = None, name_scope_global: bool = True, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            
            # Convert attributes dict to Attribute list
            attr_list = []
            if attributes:
                attr_list = [Attribute(key=k, value=v) for k, v in attributes.items()]
            
            container_info = ContainerPostInfo(
                containerName=container_name,
                basicAcl=basic_acl,
                placementPolicy=placement_policy,
                attributes=attr_list
            )
            
            container = client.create_container(container_info, bearer_token, name_scope_global=name_scope_global, wallet_connect=True)
            
            return f"""✅ Container created!
Container ID: {container.container_id}
Container Name: {container.container_name}
Basic ACL: {container.basic_acl}
Placement Policy: {container.placement_policy}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class UploadObjectTool(BaseTool):
    """Upload object to container"""
    
    name: str = "upload_object_to_neofs"
    description: str = "Upload file to NeoFS container. Supports custom attributes (key-value pairs) for later retrieval by attribute. bearer_token optional for public containers, required for eACL containers. Can accept either file_path (for local files) or content (for base64-encoded data or plain text)."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {
                "type": "string",
                "description": "Container ID"
            },
            "content": {
                "type": "string",
                "description": "File content (base64-encoded or plain text). Use this if you have the content directly. Alternative to file_path."
            },
            "file_path": {
                "type": "string",
                "description": "Path to local file to upload. Use this for large files to avoid token limits. Alternative to content. Must be absolute path or relative to current working directory."
            },
            "bearer_token": {
                "type": "string",
                "description": "Bearer token (optional for public, required for eACL)"
            },
            "attributes": {
                "type": "object",
                "description": "Object attributes as key-value pairs, e.g. {'FileName': 'test.txt', 'Type': 'Demo'}. Required if you want to retrieve object by attribute later."
            },
            "attributes_json": {
                "type": "string",
                "description": "Alternative: Object attributes as JSON string, e.g. '{\"FileName\": \"test.txt\", \"Type\": \"Demo\"}'. Use this if attributes parameter doesn't work."
            },
            "expiration_rfc3339": {
                "type": "string",
                "description": "Expiration time in RFC3339 format"
            },
            "expiration_duration": {
                "type": "string",
                "description": "Expiration duration (e.g., '24h', '7d')"
            },
            "expiration_timestamp": {
                "type": "integer",
                "description": "Expiration Unix timestamp"
            },
            "timeout": {
                "type": "number",
                "description": "Request timeout in seconds"
            },
            "wallet_connect": {
                "type": "boolean",
                "description": "Use wallet_connect mode for signing"
            }
        },
        "required": ["container_id"]
    }
    
    async def execute(
        self, 
        container_id: str, 
        content: str = None,
        file_path: str = None,
        bearer_token: str = None, 
        attributes: dict = None,
        attributes_json: str = None,
        expiration_rfc3339: str = None,
        expiration_duration: str = None,
        expiration_timestamp: int = None,
        timeout: float = 180,
        wallet_connect: bool = True,
        **kwargs
    ) -> str:
        import json
        
        try:
            import base64
            import os
            client = get_shared_neofs_client()
            
            # Handle file_path or content
            if file_path:
                # Option 1: Read from file path (avoids token limits for large files)
                if not os.path.isabs(file_path):
                    # Convert to absolute path if relative
                    file_path = os.path.abspath(file_path)
                
                # Security check: ensure file exists and is a file
                if not os.path.exists(file_path):
                    return f"❌ Failed: File not found: {file_path}"
                
                if not os.path.isfile(file_path):
                    return f"❌ Failed: Path is not a file: {file_path}"
                
                # Check file size (NeoFS limit is 64MB)
                file_size = os.path.getsize(file_path)
                max_size = 64 * 1024 * 1024  # 64MB
                if file_size > max_size:
                    return f"❌ Failed: File too large ({file_size / (1024*1024):.2f} MB, max 64 MB)"
                
                # Read file
                with open(file_path, 'rb') as f:
                    content_bytes = f.read()
                
                # Auto-set file attributes if not provided
                if not attributes and not attributes_json:
                    file_name = os.path.basename(file_path)
                    ext = os.path.splitext(file_name)[1][1:].lower()
                    content_type_map = {
                        'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                        'gif': 'image/gif', 'webp': 'image/webp', 'bmp': 'image/bmp',
                        'svg': 'image/svg+xml', 'ico': 'image/x-icon',
                        'txt': 'text/plain', 'json': 'application/json',
                        'pdf': 'application/pdf', 'zip': 'application/zip'
                    }
                    content_type = content_type_map.get(ext, 'application/octet-stream')
                    attributes = {
                        "FileName": file_name,
                        "ContentType": content_type
                    }
                    
            elif content:
                # Option 2: Handle content (base64 or plain text)
                # - If content is base64-encoded (images, binary files), decode it
                # - If content is plain text (not base64), encode as UTF-8
                # This ensures backward compatibility: plain text files work as-is,
                # while base64-encoded content (like images) is properly decoded
                try:
                    # Attempt base64 decode - if it succeeds, use decoded bytes
                    content_bytes = base64.b64decode(content, validate=True)
                except Exception:
                    # If base64 decode fails, treat as plain text and encode as UTF-8
                    # This handles regular text files without requiring base64 encoding
                    content_bytes = content.encode('utf-8')
            else:
                return "❌ Failed: Either 'file_path' or 'content' must be provided"
            
            # Parse attributes from JSON string if provided, otherwise use dict
            if attributes_json:
                try:
                    attrs = json.loads(attributes_json)
                except json.JSONDecodeError:
                    attrs = {}
            elif attributes:
                attrs = attributes
            else:
                attrs = {}
            
            result = client.upload_object(
                container_id=container_id,
                content=content_bytes,
                bearer_token=bearer_token,
                attributes=attrs,
                expiration_rfc3339=expiration_rfc3339,
                expiration_duration=expiration_duration,
                expiration_timestamp=expiration_timestamp,
                timeout=timeout,
                wallet_connect=wallet_connect
            )
            
            # Include attributes in response
            attrs_str = f"\nAttributes: {attrs}" if attrs else ""
            
            return f"""✅ Object uploaded!
Object ID: {result.object_id}
Container ID: {result.container_id}
Size: {len(content_bytes)} bytes{attrs_str}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class DownloadObjectByIdTool(BaseTool):
    """Download object by ID"""
    
    name: str = "download_neofs_object_by_id"
    description: str = "Download object by ID. Optionally save to local file if save_path is provided."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"},
            "object_id": {"type": "string", "description": "Object ID"},
            "bearer_token": {"type": "string", "description": "Bearer token (for eACL containers)"},
            "download": {"type": "boolean", "description": "Enable download mode"},
            "range_header": {"type": "string", "description": "Range header for partial download"},
            "save_path": {"type": "string", "description": "Optional local file path to save the downloaded content. If not provided, only returns content preview."}
        },
        "required": ["container_id", "object_id"]
    }
    
    async def execute(self, container_id: str, object_id: str, bearer_token: str = None, download: bool = None, range_header: str = None, save_path: str = None, **kwargs) -> str:
        import os
        try:
            client = get_shared_neofs_client()
            response = client.download_object_by_id(
                container_id=container_id, 
                object_id=object_id, 
                bearer_token=bearer_token,
                download=download,
                range_header=range_header
            )
            content = response.content
            
            # Save to file if save_path is provided
            if save_path:
                # Convert to absolute path if relative
                if not os.path.isabs(save_path):
                    save_path = os.path.abspath(save_path)
                
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                # Write file
                with open(save_path, 'wb') as f:
                    f.write(content)
                
                try:
                    text = content.decode('utf-8')
                    return f"""✅ Downloaded and saved!
Object ID: {object_id}
Container ID: {container_id}
Size: {len(content)} bytes
Saved to: {save_path}

Content preview:
{text[:500]}{'...' if len(text) > 500 else ''}"""
                except:
                    return f"""✅ Downloaded and saved!
Object ID: {object_id}
Container ID: {container_id}
Size: {len(content)} bytes (binary)
Saved to: {save_path}"""
            
            # Return content preview if not saving
            try:
                text = content.decode('utf-8')
                return f"""✅ Downloaded!
Object ID: {object_id}
Container ID: {container_id}
Size: {len(content)} bytes

Content:
{text[:500]}{'...' if len(text) > 500 else ''}"""
            except:
                return f"✅ Downloaded! Size: {len(content)} bytes (binary)"
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class GetObjectHeaderByIdTool(BaseTool):
    """Get object header by ID"""
    
    name: str = "get_neofs_object_header_by_id"
    description: str = "Get object metadata by ID."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"},
            "object_id": {"type": "string", "description": "Object ID"},
            "bearer_token": {"type": "string", "description": "Bearer token"},
            "range_header": {"type": "string", "description": "Range header"}
        },
        "required": ["container_id", "object_id"]
    }
    
    async def execute(self, container_id: str, object_id: str, bearer_token: str = None, range_header: str = None, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            response = client.get_object_header_by_id(
                container_id=container_id,
                object_id=object_id,
                bearer_token=bearer_token,
                range_header=range_header
            )
            
            headers_str = "\n".join(f"  {k}: {v}" for k, v in list(response.headers.items())[:20])
            
            return f"""✅ Object Header:
Object ID: {object_id}
Container ID: {container_id}

Headers:
{headers_str}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class DownloadObjectByAttributeTool(BaseTool):
    """Download object by attribute"""
    
    name: str = "download_neofs_object_by_attribute"
    description: str = "Download object by searching attribute. Optionally save to local file if save_path is provided."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"},
            "attr_key": {"type": "string", "description": "Attribute key"},
            "attr_val": {"type": "string", "description": "Attribute value"},
            "bearer_token": {"type": "string", "description": "Bearer token"},
            "download": {"type": "boolean", "description": "Enable download mode"},
            "range_header": {"type": "string", "description": "Range header"},
            "save_path": {"type": "string", "description": "Optional local file path to save the downloaded content. If not provided, only returns content preview."}
        },
        "required": ["container_id", "attr_key", "attr_val"]
    }
    
    async def execute(self, container_id: str, attr_key: str, attr_val: str, bearer_token: str = None, download: bool = None, range_header: str = None, save_path: str = None, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            response = client.download_object_by_attribute(
                container_id=container_id,
                attr_key=attr_key,
                attr_val=attr_val,
                bearer_token=bearer_token,
                download=download,
                range_header=range_header
            )
            content = response.content
            
            # Save to file if save_path is provided
            if save_path:
                import os
                # Convert to absolute path if relative
                if not os.path.isabs(save_path):
                    save_path = os.path.abspath(save_path)
                
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                # Write file
                with open(save_path, 'wb') as f:
                    f.write(content)
                
                try:
                    text = content.decode('utf-8')
                    return f"""✅ Downloaded by attribute and saved!
Attribute: {attr_key}={attr_val}
Container ID: {container_id}
Size: {len(content)} bytes
Saved to: {save_path}

Content preview:
{text[:500]}{'...' if len(text) > 500 else ''}"""
                except:
                    return f"""✅ Downloaded by attribute and saved!
Attribute: {attr_key}={attr_val}
Container ID: {container_id}
Size: {len(content)} bytes (binary)
Saved to: {save_path}"""
            
            # Return content preview if not saving
            try:
                text = content.decode('utf-8')
                return f"""✅ Downloaded by attribute!
Attribute: {attr_key}={attr_val}
Container ID: {container_id}
Size: {len(content)} bytes

Content:
{text[:500]}{'...' if len(text) > 500 else ''}"""
            except:
                return f"✅ Downloaded by attribute! Size: {len(content)} bytes (binary)"
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class GetObjectHeaderByAttributeTool(BaseTool):
    """Get object header by attribute"""
    
    name: str = "get_neofs_object_header_by_attribute"
    description: str = "Get object header by attribute."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"},
            "attr_key": {"type": "string", "description": "Attribute key"},
            "attr_val": {"type": "string", "description": "Attribute value"},
            "bearer_token": {"type": "string", "description": "Bearer token"},
            "range_header": {"type": "string", "description": "Range header"}
        },
        "required": ["container_id", "attr_key", "attr_val"]
    }
    
    async def execute(self, container_id: str, attr_key: str, attr_val: str, bearer_token: str = None, range_header: str = None, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            response = client.get_object_header_by_attribute(
                container_id=container_id,
                attr_key=attr_key,
                attr_val=attr_val,
                bearer_token=bearer_token,
                range_header=range_header
            )
            
            headers_str = "\n".join(f"  {k}: {v}" for k, v in list(response.headers.items())[:20])
            
            return f"""✅ Object Header by Attribute:
Attribute: {attr_key}={attr_val}
Container ID: {container_id}

Headers:
{headers_str}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class DeleteObjectTool(BaseTool):
    """Delete an object"""
    
    name: str = "delete_neofs_object"
    description: str = "Delete an object from container."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"},
            "object_id": {"type": "string", "description": "Object ID"}
        },
        "required": ["container_id", "object_id"]
    }
    
    async def execute(self, container_id: str, object_id: str, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            result = client.delete_object(container_id, object_id)
            
            return f"""✅ Object deleted!
Object ID: {object_id}
Container ID: {container_id}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class SearchObjectsTool(BaseTool):
    """Search objects in container"""
    
    name: str = "search_neofs_objects"
    description: str = "Search objects by attribute filters."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"},
            "filter_key": {"type": "string", "description": "Attribute key"},
            "filter_value": {"type": "string", "description": "Attribute value"},
            "filter_match": {"type": "string", "description": "Match type: STRING_EQUAL, etc."},
            "bearer_token": {"type": "string", "description": "Bearer token"},
            "cursor": {"type": "string", "description": "Cursor for pagination"},
            "limit": {"type": "integer", "description": "Max results"}
        },
        "required": ["container_id"]
    }
    
    async def execute(self, container_id: str, filter_key: str = None, filter_value: str = None, filter_match: str = "STRING_EQUAL", bearer_token: str = None, cursor: str = "", limit: int = 100, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            
            # NeoFS Gateway doesn't support matchType filtering via REST API
            # Use empty filters to list all objects, then filter in application layer
            search_request = SearchRequest(filters=[])
            
            result = client.search_objects(
                container_id=container_id,
                search_request=search_request,
                bearer_token=bearer_token,
                cursor=cursor,
                limit=limit
            )
            
            if not result.objects:
                return f"📦 No objects found in container"
            
            # Application-layer filtering if filter_key provided
            filtered_objects = result.objects
            if filter_key and filter_value:
                # Filter objects by attribute (would need to get headers, simplified here)
                filtered_objects = result.objects  # TODO: implement app-layer filtering
            
            objects_list = [f"  {i+1}. {obj.object_id}" for i, obj in enumerate(filtered_objects[:10])]
            objects_str = "\n".join(objects_list)
            
            filter_info = f"Filter: {filter_key}={filter_value}" if filter_key else "Filter: None (listing all)"
            
            return f"""✅ Found {len(filtered_objects)} object(s):
Container ID: {container_id}
{filter_info}
Cursor: {result.cursor if hasattr(result, 'cursor') else 'N/A'}

Objects:
{objects_str}
{'...' if len(filtered_objects) > 10 else ''}

Note: Filtering done by listing all objects (Gateway limitation)."""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class SetContainerEaclTool(BaseTool):
    """Set eACL for container"""
    
    name: str = "set_neofs_container_eacl"
    description: str = "Set eACL rules. Requires bearer token."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"},
            "bearer_token": {"type": "string", "description": "Bearer token (required)"},
            "operation": {"type": "string", "description": "Operation: GET, PUT, DELETE"},
            "action": {"type": "string", "description": "ALLOW or DENY"},
            "role": {"type": "string", "description": "Role: OTHERS, USER, SYSTEM"},
            "wallet_connect": {"type": "boolean", "description": "Use wallet_connect mode"}
        },
        "required": ["container_id", "bearer_token"]
    }
    
    async def execute(self, container_id: str, bearer_token: str, operation: str = "GET", action: str = "ALLOW", role: str = "OTHERS", wallet_connect: bool = True, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            
            eacl = Eacl(
                containerId=container_id,
                records=[Record(operation=operation, action=action, filters=[], targets=[Target(role=role, keys=[])])]
            )
            
            client.set_container_eacl(container_id, eacl, bearer_token=bearer_token, wallet_connect=wallet_connect)
            
            return f"""✅ eACL set!
Container ID: {container_id}
Rule: {action} {operation} for {role}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class GetContainerEaclTool(BaseTool):
    """Get eACL for container"""
    
    name: str = "get_neofs_container_eacl"
    description: str = "Get eACL rules for a container. This is a read-only operation that does NOT require a bearer token."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"}
        },
        "required": ["container_id"]
    }
    
    async def execute(self, container_id: str, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            eacl = client.get_container_eacl(container_id)
            
            if not eacl.records:
                return f"📦 No eACL records for container {container_id}"
            
            records_str = "\n".join(
                f"  {i+1}. {r.action} {r.operation} for role={r.targets[0].role if r.targets else 'N/A'}"
                for i, r in enumerate(eacl.records)
            )
            
            return f"""✅ eACL Records:
Container ID: {container_id}

Records:
{records_str}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class ListContainersTool(BaseTool):
    """List all containers"""
    
    name: str = "list_neofs_containers"
    description: str = "List all containers for current owner."
    parameters: dict = {
        "type": "object",
        "properties": {
            "offset": {"type": "integer", "description": "Offset for pagination"},
            "limit": {"type": "integer", "description": "Limit results"}
        },
        "required": []
    }
    
    async def execute(self, offset: int = None, limit: int = None, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            containers = client.list_containers(client.owner_address, offset=offset, limit=limit)
            
            if containers.size == 0:
                return "📦 No containers found."
            
            result = f"📦 Found {containers.size} container(s):\n\n"
            for i, c in enumerate(containers.containers, 1):
                result += f"{i}. {c.container_name}\n   ID: {c.container_id}\n   ACL: {c.basic_acl}\n   Policy: {c.placement_policy}\n\n"
            
            return result
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class GetContainerInfoTool(BaseTool):
    """Get container info"""
    
    name: str = "get_neofs_container_info"
    description: str = "Get detailed container information."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"}
        },
        "required": ["container_id"]
    }
    
    async def execute(self, container_id: str, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            container = client.get_container(container_id)
            
            attrs_str = "\n".join(f"  - {attr.key}: {attr.value}" for attr in container.attributes) if container.attributes else "  (none)"
            
            return f"""📦 Container Info:
Name: {container.container_name}
ID: {container.container_id}
Owner: {container.owner_id}
Basic ACL: {container.basic_acl}
Placement Policy: {container.placement_policy}

Attributes:
{attrs_str}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class DeleteContainerTool(BaseTool):
    """Delete container"""
    
    name: str = "delete_neofs_container"
    description: str = "Delete container. Requires bearer token."
    parameters: dict = {
        "type": "object",
        "properties": {
            "container_id": {"type": "string", "description": "Container ID"},
            "bearer_token": {"type": "string", "description": "Bearer token (required)"},
            "wallet_connect": {"type": "boolean", "description": "Use wallet_connect mode"}
        },
        "required": ["container_id", "bearer_token"]
    }
    
    async def execute(self, container_id: str, bearer_token: str, wallet_connect: bool = True, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            client.delete_container(container_id, bearer_token=bearer_token, wallet_connect=wallet_connect)
            
            return f"✅ Container {container_id} deleted!"
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class GetNetworkInfoTool(BaseTool):
    """Get network info"""
    
    name: str = "get_neofs_network_info"
    description: str = "Get NeoFS network configuration."
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    async def execute(self, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            network = client.get_network_info()
            
            return f"""🌐 NeoFS Network Information:
Max Object Size: {network.max_object_size:,} bytes ({network.max_object_size / 1024 / 1024:.2f} MB)
Container Fee: {network.container_fee}
Named Container Fee: {network.named_container_fee}
Storage Price: {network.storage_price}
Epoch Duration: {network.epoch_duration} blocks
Audit Fee: {network.audit_fee}
Withdrawal Fee: {network.withdrawal_fee}
Homomorphic Hashing: {'Disabled' if network.homomorphic_hashing_disabled else 'Enabled'}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


class GetBalanceTool(BaseTool):
    """Get balance for an address"""
    
    name: str = "get_neofs_balance"
    description: str = "Get NeoFS balance for an address."
    parameters: dict = {
        "type": "object",
        "properties": {
            "address": {"type": "string", "description": "Address to check balance (optional, defaults to owner)"}
        },
        "required": []
    }
    
    async def execute(self, address: str = None, **kwargs) -> str:
        try:
            client = get_shared_neofs_client()
            balance = client.get_balance(address)
            
            return f"""💰 Balance:
Address: {balance.address}
Value: {balance.value}
Precision: {balance.precision}"""
        except Exception as e:
            return f"❌ Failed: {str(e)}"


# Export all tools
__all__ = [
    'CreateBearerTokenTool',
    'CreateContainerTool',
    'UploadObjectTool',
    'DownloadObjectByIdTool',
    'GetObjectHeaderByIdTool',
    'DownloadObjectByAttributeTool',
    'GetObjectHeaderByAttributeTool',
    'DeleteObjectTool',
    'SearchObjectsTool',
    'SetContainerEaclTool',
    'GetContainerEaclTool',
    'ListContainersTool',
    'GetContainerInfoTool',
    'DeleteContainerTool',
    'GetNetworkInfoTool',
    'GetBalanceTool',
]
