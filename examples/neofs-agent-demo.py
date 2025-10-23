"""
NeoFS Agent Demo - AI-powered NeoFS storage operations

This demo shows how to use an AI agent to perform NeoFS operations:
- Create containers (with automatic bearer token management)
- Upload objects (with smart token handling)
- Set eACL (access control)
- Download objects
- Manage bearer tokens

The agent automatically handles:
- Bearer token creation and signing
- Wallet connect mode (wallet_connect=True)
- Public vs private container detection
- Error handling and troubleshooting

All test data is embedded for easy demonstration.

Usage:
    python examples/neofs-agent-demo.py
"""

import asyncio
from typing import List
from dotenv import load_dotenv

from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.chat import ChatBot
from pydantic import Field

# Import NeoFS tools
from spoon_ai.tools.neofs_tools import (
    CreateBearerTokenTool,
    CreateContainerTool,
    UploadObjectTool,
    SetContainerEaclTool,
    GetContainerEaclTool,
    ListContainersTool,
    GetContainerInfoTool,
    DeleteContainerTool,
    GetNetworkInfoTool,
    DownloadObjectByIdTool,
    GetObjectHeaderByIdTool,
    DownloadObjectByAttributeTool,
    GetObjectHeaderByAttributeTool,
    DeleteObjectTool,
    SearchObjectsTool,
    GetBalanceTool,
)

load_dotenv()


class NeoFSAgentDemo:
    """NeoFS Agent-based comprehensive demonstration"""

    # Embedded test data with all bearer token rules and configurations
    TEST_DATA = {
        "network": "testnet",
        "gateway_url": "https://rest.fs.neo.org",
        
        # Bearer token rules (for reference and agent guidance)
        "bearer_token_rules": {
            "container_operations": {
                "create_container": {
                    "token_type": "container",
                    "verb": "PUT",
                    "container_id": "",
                    "description": "For creating new containers",
                    "example": "Bearer(name='container-put', container=Rule(verb='PUT', containerId=''))"
                },
                "delete_container": {
                    "token_type": "container",
                    "verb": "DELETE",
                    "container_id": "<specific_container_id>",
                    "description": "For deleting specific container",
                    "example": "Bearer(name='container-delete', container=Rule(verb='DELETE', containerId='CID'))"
                },
                "set_eacl": {
                    "token_type": "container",
                    "verb": "SETEACL",
                    "container_id": "<specific_container_id>",
                    "description": "For setting container eACL rules",
                    "example": "Bearer(name='container-seteacl', container=Rule(verb='SETEACL', containerId='CID'))"
                }
            },
            "object_operations": {
                "upload_object": {
                    "token_type": "object",
                    "operation": "PUT",
                    "lifetime": 3600,
                    "description": "For uploading objects to eACL containers",
                    "example": "Bearer(name='object-upload', object=[Record(action='ALLOW', operation='PUT', ...)])"
                },
                "download_object": {
                    "token_type": "object",
                    "operation": "GET",
                    "lifetime": 3600,
                    "description": "For downloading objects from eACL containers",
                    "example": "Bearer(name='object-download', object=[Record(action='ALLOW', operation='GET', ...)])"
                },
                "delete_object": {
                    "token_type": "object",
                    "operation": "DELETE",
                    "lifetime": 3600,
                    "description": "For deleting objects from eACL containers",
                    "example": "Bearer(name='object-delete', object=[Record(action='ALLOW', operation='DELETE', ...)])"
                }
            }
        },
        
        # Container configurations
        "container_configs": {
            "public_container": {
                "basic_acl": "public-read-write",
                "placement_policy": "REP 1",
                "description": "Public container, anyone can read/write, no bearer token needed",
                "bearer_token_required": {
                    "create": True,
                    "upload": False,
                    "download": False,
                    "delete": False
                }
            },
            "eacl_container": {
                "basic_acl": "eacl-public-read-write",
                "placement_policy": "REP 1",
                "description": "eACL-controlled container, requires bearer token for all object operations",
                "bearer_token_required": {
                    "create": True,
                    "upload": True,
                    "download": True,
                    "delete": True,
                    "set_eacl": True
                }
            }
        },
        
        # eACL rules examples
        "eacl_rules": {
            "deny_all_others": {
                "description": "Deny all operations for OTHERS role",
                "rules": [
                    {"operation": "GET", "action": "DENY", "role": "OTHERS"},
                    {"operation": "PUT", "action": "DENY", "role": "OTHERS"},
                    {"operation": "DELETE", "action": "DENY", "role": "OTHERS"}
                ]
            },
            "allow_read_only": {
                "description": "Allow read-only for OTHERS role",
                "rules": [
                    {"operation": "GET", "action": "ALLOW", "role": "OTHERS"},
                    {"operation": "PUT", "action": "DENY", "role": "OTHERS"}
                ]
            }
        },
        
        # Test data for scenarios
        "test_scenarios": {
            "existing_containers": [
                {
                    "id": "xxxxxx",
                    "name": "demo-public-storage",
                    "type": "public",
                    "acl": "public-read-write"
                }
            ],
            "sample_files": [
                {
                    "name": "welcome.txt",
                    "content": "Hello from NeoFS! This is a public file.",
                    "attributes": {
                        "FileName": "welcome.txt",
                        "Author": "Demo Agent",
                        "Type": "Text",
                        "Environment": "Demo"
                    }
                },
                {
                    "name": "confidential.txt",
                    "content": "Q4 2025 Financial Report - Confidential Data",
                    "attributes": {
                        "FileName": "confidential.txt",
                        "Classification": "Confidential",
                        "Department": "Finance",
                        "Quarter": "Q4-2025"
                    }
                },
                {
                    "name": "report.json",
                    "content": '{"report": "Monthly Summary", "period": "October 2025"}',
                    "attributes": {
                        "FileName": "report.json",
                        "ContentType": "application/json",
                        "Type": "Report"
                    }
                }
            ]
        }
    }

    def __init__(self):
        """Initialize the demo with embedded test data"""
        self.load_test_data()
        self.agents = {}

    def load_test_data(self):
        """Load test data from embedded TEST_DATA configuration"""
        try:
            data = self.TEST_DATA
            
            # Load basic configuration
            self.network = data.get("network", "testnet")
            self.gateway_url = data.get("gateway_url", "")
            
            # Load bearer token rules
            self.bearer_token_rules = data.get("bearer_token_rules", {})
            
            # Load container configurations
            self.container_configs = data.get("container_configs", {})
            
            # Load eACL rules
            self.eacl_rules = data.get("eacl_rules", {})
            
            # Load test scenarios
            test_scenarios = data.get("test_scenarios", {})
            self.existing_containers = test_scenarios.get("existing_containers", [])
            self.sample_files = test_scenarios.get("sample_files", [])
            
            print(f"✅ Loaded test data from embedded configuration")
            print(f"   Network: {self.network}")
            print(f"   Gateway: {self.gateway_url}")
            print(f"   Bearer Token Rules: {len(self.bearer_token_rules.get('container_operations', {}))} container + {len(self.bearer_token_rules.get('object_operations', {}))} object")
            print(f"   Container Configs: {len(self.container_configs)}")
            print(f"   eACL Rules: {len(self.eacl_rules)}")
            print(f"   Sample Files: {len(self.sample_files)}")
            
        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            # Set minimal defaults
            self.network = "testnet"
            self.gateway_url = ""
            self.bearer_token_rules = {}
            self.container_configs = {}
            self.eacl_rules = {}
            self.existing_containers = []
            self.sample_files = []

    def create_agent(self, name: str, tools: List, description: str) -> ToolCallAgent:
        """Create a specialized agent with specific tools"""
        
        class NeoFSSpecializedAgent(ToolCallAgent):
            agent_name: str = name
            agent_description: str = description
            system_prompt: str = f"""
        You are a NeoFS storage specialist focused on {description}.

        CRITICAL - Bearer Token Management Rules:

        1. Container Operations (require container bearer tokens):
        - CREATE container: create_neofs_bearer_token(token_type="container", verb="PUT", container_id="")
        - DELETE container: create_neofs_bearer_token(token_type="container", verb="DELETE", container_id="<CID>")
        - SET eACL: create_neofs_bearer_token(token_type="container", verb="SETEACL", container_id="<CID>")

        2. Object Operations (depends on container type):
        - PUBLIC container (basic_acl="public-read-write"):
            * Upload: NO bearer token needed
            * Download: NO bearer token needed
            * Delete: NO bearer token needed
        
        - eACL container (basic_acl="eacl-public-read-write"):
            * Upload: create_neofs_bearer_token(token_type="object", operation="PUT", lifetime=3600)
            * Download: create_neofs_bearer_token(token_type="object", operation="GET", lifetime=3600)
            * Delete: create_neofs_bearer_token(token_type="object", operation="DELETE", lifetime=3600)

        3. Complete Workflow Examples:

        A. Create PUBLIC Container and Upload:
            Step 1: create_neofs_bearer_token(token_type="container", verb="PUT")
            Step 2: create_neofs_container(name, bearer_token, basic_acl="public-read-write")
            Step 3: upload_object_to_neofs(container_id, content) ← NO token!
        
        B. Create eACL Container and Upload:
            Step 1: create_neofs_bearer_token(token_type="container", verb="PUT")
            Step 2: create_neofs_container(name, bearer_token, basic_acl="eacl-public-read-write")
            Step 3: create_neofs_bearer_token(token_type="object", operation="PUT")
            Step 4: upload_object_to_neofs(container_id, content, bearer_token)
        
        C. Set eACL to Deny All:
            Step 1: create_neofs_bearer_token(token_type="container", verb="SETEACL", container_id="<CID>")
            Step 2: set_neofs_container_eacl(container_id, bearer_token, operation="GET", action="DENY", role="OTHERS")

        Always execute steps in correct order. Track all container_ids, object_ids, and bearer_tokens.
        Explain each step clearly.
        """
            max_steps: int = 20
            avaliable_tools: ToolManager = Field(default_factory=lambda: ToolManager(tools))
        
        agent = NeoFSSpecializedAgent(
            llm=ChatBot(
                llm_provider="openrouter",
                model_name="openai/gpt-4o"
            )
        )
        return agent

    def setup_agents(self):
        """Setup specialized agents for different NeoFS operations"""
        
        # Container Manager Agent (5 tools)
        container_tools = [
            CreateBearerTokenTool(),
            CreateContainerTool(),
            ListContainersTool(),
            GetContainerInfoTool(),
            DeleteContainerTool(),
        ]
        self.agents['container'] = self.create_agent(
            "Container Manager",
            container_tools,
            "Expert in NeoFS container management, creation, and configuration"
        )
        
        # Object Storage Agent (8 tools)
        object_tools = [
            CreateBearerTokenTool(),
            UploadObjectTool(),
            DownloadObjectByIdTool(),
            DownloadObjectByAttributeTool(),
            GetObjectHeaderByIdTool(),
            GetObjectHeaderByAttributeTool(),
            DeleteObjectTool(),
            SearchObjectsTool(),
        ]
        self.agents['object'] = self.create_agent(
            "Object Storage Manager",
            object_tools,
            "Specialist in NeoFS object operations, upload, download, and search. Supports both by_id and by_attribute operations."
        )
        
        # Access Control Agent (4 tools)
        access_tools = [
            CreateBearerTokenTool(),
            SetContainerEaclTool(),
            GetContainerEaclTool(),
            GetContainerInfoTool(),
        ]
        self.agents['access'] = self.create_agent(
            "Access Control Manager",
            access_tools,
            "Expert in NeoFS eACL configuration and bearer token management"
        )
        
        # Network Monitor Agent (3 tools)
        network_tools = [
            GetNetworkInfoTool(),
            GetBalanceTool(),
            ListContainersTool(),
        ]
        self.agents['network'] = self.create_agent(
            "Network Monitor",
            network_tools,
            "Specialist in NeoFS network status, balance, and monitoring"
        )

    def print_section_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")

    async def run_agent_scenario(self, agent_name: str, scenario_title: str, user_message: str):
        """Run a specific scenario with an agent"""
        print(f"\n{'-'*60}")
        print(f" Agent: {self.agents[agent_name].agent_name}")
        print(f" Scenario: {scenario_title}")
        print(f" Query: {user_message[:100]}...")
        print(f"{'-'*60}")

        try:
            # Clear agent state before running
            self.agents[agent_name].clear()

            # Run the agent with the user message
            response = await self.agents[agent_name].run(user_message)

            print(f"✅ Response: {response}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    async def demo_network_status(self):
        """Demonstrate network monitoring capabilities"""
        self.print_section_header("1. Network Status and Monitoring")

        await self.run_agent_scenario(
            'network',
            "Network Configuration",
            "Show me the current NeoFS network configuration. What's the max object size, fees, and epoch duration?"
        )

        await self.run_agent_scenario(
            'network',
            "Account Balance",
            "Check my NeoFS balance. What's my current balance and precision?"
        )

    async def demo_container_operations(self):
        """Demonstrate container management capabilities"""
        self.print_section_header("2. Container Management Operations")
        
        # Use persistent agent (don't clear between scenarios)
        agent = self.agents['container']
        
        # Scenario 1: List containers
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: List Existing Containers")
        print(f"{'-'*60}")
        
        response1 = await agent.run(
            "List all my existing NeoFS containers. Show me their names, IDs, ACL settings, and placement policies."
        )
        print(f"✅ Response: {response1}")

        # Scenario 2: Create PUBLIC container (agent remembers context)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Create PUBLIC Container")
        print(f"{'-'*60}")
        
        import time
        timestamp = int(time.time())
        
        response2 = await agent.run(f"""Create a PUBLIC container with the following configuration:
        - Container name: 'agent-demo-public-{timestamp}'
        - Basic ACL: 'public-read-write'
        - Placement policy: 'REP 1'
        - Attributes: {{"environment": "demo", "type": "public", "created_by": "agent"}}

        Remember to:
        1. First create a container bearer token (verb=PUT)
        2. Then create the container using that token
        3. Show me the container ID and remember it for later use

        Explain each step.""")
        print(f"✅ Response: {response2}")

        # Scenario 3: Create eACL container (agent still remembers)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Create eACL Container")
        print(f"{'-'*60}")
        
        response3 = await agent.run(f"""Create an eACL-controlled container:
            - Container name: 'agent-demo-eacl-{timestamp}'
            - Basic ACL: 'eacl-public-read-write'
            - Placement policy: 'REP 1'
            - Attributes: {{"environment": "production", "security": "high", "created_by": "agent"}}

            Steps:
            1. Create container bearer token (verb=PUT)
            2. Create the container
            3. Explain the difference between this and PUBLIC containers

            Show me the container ID and remember it as 'eacl_container_id' for later.""")
        print(f"✅ Response: {response3}")

        # Scenario 4: Get details of PUBLIC container we just created
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Get Container Details")
        print(f"{'-'*60}")
        
        response4 = await agent.run(
            f"Get detailed information about the PUBLIC container we just created (agent-demo-public-{timestamp}). Show me all its attributes and configuration."
        )
        print(f"✅ Response: {response4}")

    async def demo_public_container_workflow(self):
        """Demonstrate PUBLIC container complete workflow"""
        self.print_section_header("3. PUBLIC Container Complete Workflow")
        
        # Use PUBLIC container ID
        public_container_id = "xxxxxxxxx"
        
        # Use persistent agent to remember object_id
        agent = self.agents['object']

        sample_file = self.sample_files[0] if self.sample_files else {
            "name": "test.txt",
            "content": "Test content",
            "attributes": {"FileName": "test.txt"}
        }

        # Step 1: Upload
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Upload to PUBLIC Container")
        print(f"{'-'*60}")
        
        response1 = await agent.run(f"""Upload a file to PUBLIC container {public_container_id}:

            File details:
            - Content: '{sample_file['content']}'
            - Attributes: {sample_file['attributes']}

            Since this is a PUBLIC container, you don't need a bearer token for upload.
            Show me the object ID after upload and remember it for later.""")
        print(f"✅ Response: {response1}")

        # Step 2: Search (agent remembers container_id)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Search Objects")
        print(f"{'-'*60}")
        
        response2 = await agent.run(f"""Search for objects in the PUBLIC container {public_container_id}:
            - Filter by: FileName = '{sample_file['attributes']['FileName']}'
            - Show me all matching objects""")
        print(f"✅ Response: {response2}")

        # Step 3: Download (agent remembers object_id from step 1)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Download Object")
        print(f"{'-'*60}")
        
        response3 = await agent.run("""Download the object we just uploaded.
            Since it's PUBLIC, no bearer token is needed.
            Show me the file content.""")
        print(f"✅ Response: {response3}")

    async def demo_eacl_container_workflow(self):
        """Demonstrate eACL container complete workflow with full token management"""
        self.print_section_header("4. eACL Container Complete Workflow")
        
        # Use existing eACL container
        eacl_container_id = "xxxxxxxxx"
        
        # Use persistent agent
        agent = self.agents['access']
        object_agent = self.agents['object']

        sample_file = self.sample_files[1] if len(self.sample_files) > 1 else {
            "name": "secret.txt",
            "content": "Confidential data",
            "attributes": {"FileName": "secret.txt", "Classification": "Secret"}
        }

        # Step 1: Set eACL (CRITICAL step to activate bearer token validation!)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Set eACL to DENY GET")
        print(f"{'-'*60}")
        
        response1 = await agent.run(f"""Set eACL on container {eacl_container_id}:

            CRITICAL: This step activates bearer token authentication!

            Steps:
            1. Create eACL bearer token (token_type="container", verb="SETEACL", container_id="{eacl_container_id}")
            2. Set eACL rule: operation="GET", action="DENY", role="OTHERS"
            3. Explain why this step is necessary for bearer tokens to work

            Show me the result.""")
        print(f"✅ Response: {response1}")

        # Step 2: Upload object with bearer token (now it works!)
        print(f"\n{'-'*60}")
        print(f" Agent: {object_agent.agent_name}")
        print(f" Scenario: Upload to eACL Container")
        print(f"{'-'*60}")
        
        response2 = await object_agent.run(f"""Upload a file to eACL container {eacl_container_id}:

            Steps:
            1. Create object bearer token (token_type="object", operation="PUT", lifetime=3600)
            2. Upload file:
            - Content: '{sample_file['content']}'
            - Attributes: {sample_file['attributes']}

            Execute both steps and show me:
            - The bearer token created
            - The object ID after upload
            - Explain why bearer token works (because we set eACL)

            Remember the object ID for later.""")
        print(f"✅ Response: {response2}")

        # Step 3: Download object with bearer token
        print(f"\n{'-'*60}")
        print(f" Agent: {object_agent.agent_name}")
        print(f" Scenario: Download from eACL Container")
        print(f"{'-'*60}")
        
        response3 = await object_agent.run("""Download the file we just uploaded:

                Steps:
                1. Create object bearer token (token_type="object", operation="GET", lifetime=3600)
                2. Download using the token and object ID from previous step
                3. Show me the file content

                Explain the security model.""")
        print(f"✅ Response: {response3}")
        
    async def demo_access_control(self):
        """Demonstrate container creation and deletion"""
        self.print_section_header("5. Container Lifecycle: Create and Delete")
        
        import time
        timestamp = int(time.time())
        
        # Use persistent agent to remember container IDs
        agent = self.agents['container']

        # Scenario 1: Create and Delete PUBLIC Container
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Create and Delete PUBLIC Container")
        print(f"{'-'*60}")
        
        response1 = await agent.run(f"""Complete workflow to create and delete a PUBLIC container:

        Steps:
        1. Create bearer token (token_type="container", verb="PUT")
        2. Create PUBLIC container:
        - Name: 'test-public-{timestamp}'
        - Basic ACL: 'public-read-write'
        - Placement policy: 'REP 1'
        3. Show the container ID and remember it
        4. Create DELETE bearer token (token_type="container", verb="DELETE", container_id=<the_container_id>)
        5. Delete the container we just created

        Explain each step and what bearer tokens are needed.""")
        print(f"✅ Response: {response1}")

        # Scenario 2: Create and Delete eACL Container
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Create and Delete eACL Container")
        print(f"{'-'*60}")
        
        response2 = await agent.run(f"""Complete workflow to create and delete an eACL container:

        Steps:
        1. Create bearer token (token_type="container", verb="PUT")
        2. Create eACL container:
        - Name: 'test-eacl-{timestamp}'
        - Basic ACL: 'eacl-public-read-write'
        - Placement policy: 'REP 1'
        3. Show the container ID and remember it
        4. Create DELETE bearer token (token_type="container", verb="DELETE", container_id=<the_container_id>)
        5. Delete the container we just created

        Explain the complete lifecycle and bearer token requirements.""")
        print(f"✅ Response: {response2}")

    async def demo_advanced_scenarios(self):
        """Demonstrate advanced object operations for both PUBLIC and eACL containers"""
        self.print_section_header("6. Advanced Object Operations")
        
        # User-provided container IDs
        public_container_id = "xxxxxxxxx"
        eacl_container_id = "xxxxxxxxx"
        
        # Use persistent agent
        agent = self.agents['object']
        
        # ===== PUBLIC Container Operations =====
        print("\n" + "="*60)
        print(" PUBLIC Container Operations")
        print("="*60)
        
        # Scenario 1: Upload to PUBLIC container
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Upload to PUBLIC Container")
        print(f"{'-'*60}")
        
        response1 = await agent.run(f"""Upload a file to PUBLIC container.

Call the upload_object_to_neofs tool with exactly these parameters:
- container_id: "{public_container_id}"
- content: "Test file for advanced demo"
- attributes_json: "{{\\"FileName\\": \\"advanced-test.txt\\", \\"Type\\": \\"Demo\\"}}"

Note: Use attributes_json (string type) instead of attributes (object type).
No bearer_token parameter needed (PUBLIC container).
Remember the object_id from the response for next steps.""")
        print(f"✅ Response: {response1}")
        
        # Scenario 2: Get object header by attribute
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Get Object Header by Attribute (PUBLIC)")
        print(f"{'-'*60}")
        
        response2 = await agent.run(f"""Get object header by attribute from PUBLIC container.

Call the get_neofs_object_header_by_attribute tool with exactly these parameters:
- container_id: "{public_container_id}"
- attr_key: "FileName"
- attr_val: "advanced-test.txt"

No bearer_token parameter needed (PUBLIC container).
Show me the object metadata.""")
        print(f"✅ Response: {response2}")
        
        # Scenario 3: Download object by attribute
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Download Object by Attribute (PUBLIC)")
        print(f"{'-'*60}")
        
        response3 = await agent.run(f"""Download object by attribute from PUBLIC container.

Call the download_neofs_object_by_attribute tool with exactly these parameters:
- container_id: "{public_container_id}"
- attr_key: "FileName"
- attr_val: "advanced-test.txt"

No bearer_token parameter needed (PUBLIC container).
Show me the file content.""")
        print(f"✅ Response: {response3}")
        
        # Scenario 4: Get object header by ID
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Get Object Header by ID (PUBLIC)")
        print(f"{'-'*60}")
        
        response4 = await agent.run("""Get object header by ID for the object we just uploaded.
                    Use the object_id from the upload step. No bearer token needed.""")
        print(f"✅ Response: {response4}")
        
        # Scenario 5: Delete object
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Delete Object (PUBLIC)")
        print(f"{'-'*60}")
        
        response5 = await agent.run("""Delete the object we uploaded.
        Use the container_id and object_id. No bearer token needed for delete.""")
        print(f"✅ Response: {response5}")
        
        # ===== eACL Container Operations =====
        print("\n" + "="*60)
        print(" eACL Container Operations")
        print("="*60)
        
        # Scenario 6: Get container eACL (use access agent)
        print(f"\n{'-'*60}")
        access_agent = self.agents['access']
        print(f" Agent: {access_agent.agent_name}")
        print(f" Scenario: Get Container eACL")
        print(f"{'-'*60}")
        
        response6 = await access_agent.run(f"""Get eACL rules for container {eacl_container_id}.

Use the get_neofs_container_eacl tool directly with just the container_id.
This is a read-only operation that does NOT require a bearer token.

Show me the current access control configuration.""")
        print(f"✅ Response: {response6}")
        
        # Scenario 7: Upload to eACL container (needs PUT token)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Upload to eACL Container")
        print(f"{'-'*60}")
        
        response7 = await agent.run(f"""Upload a file to eACL container.

Steps:
1. First, call create_neofs_bearer_token with:
   - token_type: "object"
   - operation: "PUT"
   - lifetime: 3600

2. Then, call upload_object_to_neofs with exactly these parameters:
   - container_id: "{eacl_container_id}"
   - content: "Advanced eACL demo file"
   - attributes_json: "{{\\"FileName\\": \\"eacl-advanced-test.txt\\", \\"Type\\": \\"Demo\\"}}"
   - bearer_token: <the token from step 1>

Note: Use attributes_json (string) instead of attributes (object).
Remember the object_id from the response. Explain why PUT bearer token is needed.""")
        print(f"✅ Response: {response7}")
        
        # Scenario 8: Get object header by attribute (eACL - needs GET token)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Get Object Header by Attribute (eACL)")
        print(f"{'-'*60}")
        
        response8 = await agent.run(f"""Get object header by attribute from eACL container.

Steps:
1. First, call create_neofs_bearer_token with:
   - token_type: "object"
   - operation: "GET"
   - lifetime: 3600

2. Then, call get_neofs_object_header_by_attribute with exactly these parameters:
   - container_id: "{eacl_container_id}"
   - attr_key: "FileName"
   - attr_val: "eacl-advanced-test.txt"
   - bearer_token: <the token from step 1>

Show the object metadata. Explain why GET bearer token is needed.""")
        print(f"✅ Response: {response8}")
        
        # Scenario 9: Download object by attribute (eACL - needs GET token)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Download Object by Attribute (eACL)")
        print(f"{'-'*60}")
        
        response9 = await agent.run(f"""Download object by attribute from eACL container.

Steps:
1. First, call create_neofs_bearer_token with:
   - token_type: "object"
   - operation: "GET"
   - lifetime: 3600

2. Then, call download_neofs_object_by_attribute with exactly these parameters:
   - container_id: "{eacl_container_id}"
   - attr_key: "FileName"
   - attr_val: "eacl-advanced-test.txt"
   - bearer_token: <the token from step 1>

Show me the file content. Explain why GET bearer token is needed.""")
        print(f"✅ Response: {response9}")
        
        # Scenario 10: Get object header by ID (eACL - needs GET token)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Get Object Header by ID (eACL)")
        print(f"{'-'*60}")
        
        response10 = await agent.run("""Get detailed object header by ID from eACL container:

Steps:
1. Create bearer token (token_type="object", operation="GET", lifetime=3600)
2. Get header by ID using the object_id from the upload step
3. Use the bearer token

Show all the object metadata including headers.""")
        print(f"✅ Response: {response10}")
        
        # Scenario 11: Delete object (eACL - no token needed)
        print(f"\n{'-'*60}")
        print(f" Agent: {agent.agent_name}")
        print(f" Scenario: Delete Object (eACL)")
        print(f"{'-'*60}")
        
        response11 = await agent.run("""Delete the object we uploaded to the eACL container:

Use the container_id and object_id from previous steps.
No bearer token needed for delete operation.""")
        print(f"✅ Response: {response11}")

    async def run_comprehensive_demo(self):
        """Run the complete agent-based demonstration"""
        print("🚀 NeoFS Storage - AI Agent Comprehensive Demonstration")
        print("=" * 80)
        print("This demo showcases NeoFS storage tools through specialized AI agents")
        print("Each agent is an expert in specific aspects of NeoFS")
        print("=" * 80)
        print(f" Network: {self.network}")
        print(f" Gateway: {self.gateway_url}")
        print(f" Bearer Token Rules: Configured for all operations")
        print(f" Container Configs: PUBLIC + eACL templates ready")
        print(f" eACL Rules: Multiple patterns available")

        try:
            # Setup all specialized agents
            print("\n🔧 Setting up specialized agents...")
            self.setup_agents()
            print(f"✅ Created {len(self.agents)} specialized agents")

            # Run comprehensive demonstrations
            # await self.demo_network_status()
            # await self.demo_container_operations()
            # await self.demo_public_container_workflow()
            # await self.demo_eacl_container_workflow()
            # await self.demo_access_control()
            await self.demo_advanced_scenarios()

            # Final summary
            self.print_section_header("Demo Completed Successfully")
            for agent_name, agent in self.agents.items():
                tool_count = len(agent.avaliable_tools.tools)
                print(f"  ✅ {agent.agent_name}: {tool_count} specialized tools")

            total_tools = sum(len(agent.avaliable_tools.tools) for agent in self.agents.values())
            print(f"\n🔧 Total Tools Demonstrated: 16 NeoFS tools")
            print("   All demonstrations powered by AI agents with domain expertise")
            print("   Each agent provides intelligent analysis and workflow orchestration")
            print("\nAgent Capabilities:")
            print("   1. Container Manager: Create, list, manage containers")
            print("   2. Object Storage Manager: Upload, download, search objects")
            print("   3. Access Control Manager: Configure eACL, manage bearer tokens")
            print("   4. Network Monitor: Check status, balance, network info")
            print("\nKey Features Demonstrated:")
            print("   ✅ Automatic bearer token creation and management")
            print("   ✅ PUBLIC vs eACL container workflows")
            print("   ✅ Complete object lifecycle (upload, search, download, delete)")
            print("   ✅ eACL configuration and access control")
            print("   ✅ Multi-file operations and batch processing")

        except Exception as e:
            print(f"\n❌ Demo error: {str(e)}")
            print("Please check your environment setup and network connectivity")


async def main():
    """Main demonstration function"""
    print("\n🌟 NeoFS Storage - AI Agent Demonstration")
    print("=" * 80)
    print("Showcasing comprehensive NeoFS storage operations through specialized AI agents")
    print("Each agent is equipped with domain-specific tools and expertise")
    print("=" * 80)

    demo = NeoFSAgentDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
