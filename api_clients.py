# api_clients.py
"""
Final Fixed API client classes with completely corrected email field structures
"""
import requests
from typing import Dict, Optional
from datetime import datetime
import json


class PlainClient:
    """GraphQL client for Plain API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://core-api.uk.plain.com/graphql/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def test_connection(self) -> Dict:
        """Test the API connection and return workspace info"""
        query = """
        query {
            myWorkspace {
                id
                name
                publicName
            }
        }
        """
        
        payload = {"query": query}
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            result = response.json()
            
            print(f"API Response Status: {response.status_code}")
            print(f"API Response: {json.dumps(result, indent=2)}")
            
            if "errors" in result:
                return {"success": False, "errors": result["errors"]}
            
            workspace = result.get("data", {}).get("myWorkspace")
            return {"success": True, "workspace": workspace}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_customer_first(self, email: str, name: str = None) -> Optional[Dict]:
        """Create customer using the corrected email field structure for both input and output"""
        mutation = """
        mutation upsertCustomer($input: UpsertCustomerInput!) {
            upsertCustomer(input: $input) {
                customer {
                    id
                    email {
                        email
                        isVerified
                    }
                    fullName
                }
                error {
                    message
                    type
                    code
                }
            }
        }
        """
        
        variables = {
            "input": {
                "identifier": { "emailAddress": email },
                "onCreate": {
                    "email": { "email": email, "isVerified": False },  # <-- add this
                    "fullName": name or "Support Customer"
                },
                "onUpdate": {
                    "email": { "email": email, "isVerified": False }   # optional but consistent
                }
            }
        }
        
        payload = {
            "query": mutation,
            "variables": variables
        }
        
        try:
            print(f"Creating customer with email: {email}")
            print(f"Customer creation payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            result = response.json()
            
            print(f"Customer creation response: {json.dumps(result, indent=2)}")
            
            if "errors" in result:
                raise Exception(f"Customer creation GraphQL errors: {result['errors']}")
            
            customer_data = result.get("data", {}).get("upsertCustomer", {})
            if customer_data.get("error"):
                raise Exception(f"Customer creation error: {customer_data['error']}")
            
            return customer_data.get("customer")
            
        except Exception as e:
            raise Exception(f"Failed to create customer: {str(e)}")

    def debug_create_thread(self, title: str, description: str, customer_email: str, priority: str = "MEDIUM") -> dict:
        """Debug version of create_thread that shows detailed info"""
        print(f"\n=== DEBUG: Creating Plain Thread ===")
        print(f"Title: {title}")
        print(f"Description: {description[:100]}...")
        print(f"Customer Email: {customer_email}")
        print(f"Priority: {priority}")
        
        try:
            # First test connection
            print("Testing API connection...")
            connection_test = self.test_connection()
            if not connection_test.get("success"):
                return {"success": False, "error": "API connection failed", "details": connection_test}
            
            print("✅ API connection successful")
            workspace = connection_test.get("workspace")
            if workspace:
                print(f"Connected to workspace: {workspace.get('name', 'Unknown')} (ID: {workspace.get('id')})")
            
            # First ensure customer exists
            print("Creating/finding customer...")
            customer = self.create_customer_first(customer_email)
            customer_id = customer.get("id") if customer else None
            print(f"✅ Customer ID: {customer_id}")
            if customer and customer.get("email"):
                customer_email_data = customer.get("email")
                if isinstance(customer_email_data, dict):
                    print(f"✅ Customer Email: {customer_email_data.get('email')} (Verified: {customer_email_data.get('isVerified')})")
                else:
                    print(f"✅ Customer Email: {customer_email_data}")
            
            print("Creating thread...")
            mutation = """
            mutation createThread($input: CreateThreadInput!) {
                createThread(input: $input) {
                    thread {
                        id
                        title
                        description
                        status
                        createdAt {
                            iso8601
                        }
                        customer {
                            id
                            email {
                                email
                                isVerified
                            }
                        }
                    }
                    error {
                        message
                        type
                        code
                    }
                }
            }
            """
            
            priority_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "URGENT": 4}
            priority_int = priority_map.get(priority.upper(), 2)
            
            if customer_id:
                customer_identifier = {"customerId": customer_id}
                print(f"Using customer ID: {customer_id}")
            else:
                customer_identifier = {"emailAddress": customer_email}
                print(f"Using customer email: {customer_email}")
            
            variables = {
                "input": {
                    "title": title,
                    "description": description,
                    "customerIdentifier": customer_identifier,
                    "priority": priority_int
                }
            }
            
            payload = {"query": mutation, "variables": variables}
            
            print(f"Sending thread creation payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            result = response.json()
            
            print(f"Thread creation response status: {response.status_code}")
            print(f"Thread creation response body: {json.dumps(result, indent=2)}")
            
            return {
                "success": True, 
                "result": result, 
                "response_code": response.status_code,
                "customer_id": customer_id
            }
            
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            return {"success": False, "error": str(e)}

    def create_thread(self, title: str, description: str, customer_email: str, priority: str = "MEDIUM") -> Optional[Dict]:
        """Create a new thread (ticket) in Plain"""
        
        # First ensure customer exists
        try:
            customer = self.create_customer_first(customer_email)
            customer_id = customer.get("id") if customer else None
        except Exception as e:
            # If customer creation fails, we'll try using email directly
            customer_id = None
            print(f"Customer creation failed, trying direct approach: {e}")
        
        mutation = """
        mutation createThread($input: CreateThreadInput!) {
            createThread(input: $input) {
                thread {
                    id
                    title
                    description
                    status
                    createdAt {
                        iso8601
                    }
                }
                error {
                    message
                    type
                    code
                }
            }
        }
        """
        
        # Convert priority string to integer 
        priority_map = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "URGENT": 4
        }
        priority_int = priority_map.get(priority.upper(), 2)
        
        # Try different customer identifier approaches
        if customer_id:
            # Use customer ID if we have it
            customer_identifier = {"customerId": customer_id}
        else:
            # Fall back to email address
            customer_identifier = {"emailAddress": customer_email}
        
        variables = {
            "input": {
                "title": title,
                "description": description,
                "customerIdentifier": customer_identifier,
                "priority": priority_int
            }
        }
        
        payload = {
            "query": mutation,
            "variables": variables
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            
            if "errors" in result:
                raise Exception(f"GraphQL errors: {result['errors']}")
            
            thread_data = result.get("data", {}).get("createThread", {})
            if thread_data.get("error"):
                raise Exception(f"Plain API error: {thread_data['error']}")
            
            return thread_data.get("thread")
            
        except Exception as e:
            raise Exception(f"Failed to create Plain thread: {str(e)}")

    def upsert_customer(self, email: str, full_name: str = None) -> Optional[Dict]:
        """Create or update a customer first - with corrected email field structure"""
        
        mutation = """
        mutation upsertCustomer($input: UpsertCustomerInput!) {
            upsertCustomer(input: $input) {
                customer {
                    id
                    email {
                        email
                        isVerified
                    }
                    fullName
                }
                error {
                    message
                    type
                    code
                }
            }
        }
        """
        
        variables = {
            "input": {
                "identifier": {
                    "emailAddress": email  # This is for the identifier input
                },
                "onCreate": {
                    "email": {
                        "email": email,
                        "isVerified": False
                    },
                    "fullName": full_name or email.split('@')[0]
                },
                "onUpdate": {}
            }
        }
        
        payload = {
            "query": mutation,
            "variables": variables
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            
            if "errors" in result:
                raise Exception(f"GraphQL errors: {result['errors']}")
            
            customer_data = result.get("data", {}).get("upsertCustomer", {})
            if customer_data.get("error"):
                raise Exception(f"Plain API error: {customer_data['error']}")
            
            return customer_data.get("customer")
            
        except Exception as e:
            raise Exception(f"Failed to upsert customer: {str(e)}")


class ShortcutClient:
    """REST client for Shortcut API"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.app.shortcut.com/api/v3"
        self.headers = {
            "Content-Type": "application/json",
            "Shortcut-Token": api_token
        }
        self._default_project_id = None
        self._default_workflow_id = None
    
    def get_projects(self) -> Optional[list]:
        """Get list of projects (useful for getting project_id)"""
        try:
            response = requests.get(f"{self.base_url}/projects", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to get projects: {str(e)}")
    
    def get_workflows(self) -> Optional[list]:
        """Get list of workflows"""
        try:
            response = requests.get(f"{self.base_url}/workflows", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to get workflows: {str(e)}")
    
    def get_workflow_states(self, workflow_id: int) -> Optional[list]:
        """Get workflow states for a specific workflow"""
        try:
            response = requests.get(f"{self.base_url}/workflows/{workflow_id}", headers=self.headers)
            response.raise_for_status()
            workflow = response.json()
            return workflow.get("states", [])
        except Exception as e:
            raise Exception(f"Failed to get workflow states: {str(e)}")
    
    def get_default_project_id(self) -> Optional[int]:
        """Get the first available project ID"""
        if self._default_project_id:
            return self._default_project_id
            
        try:
            projects = self.get_projects()
            if projects and len(projects) > 0:
                self._default_project_id = projects[0]["id"]
                return self._default_project_id
        except:
            pass
        return None
    
    def get_default_workflow_state_id(self) -> Optional[int]:
        """Get the first available workflow state ID"""
        try:
            workflows = self.get_workflows()
            if workflows and len(workflows) > 0:
                # Get the first workflow
                first_workflow = workflows[0]
                workflow_id = first_workflow["id"]
                
                # Get its states
                states = self.get_workflow_states(workflow_id)
                if states and len(states) > 0:
                    # Return the first state ID (usually the initial state)
                    return states[0]["id"]
        except Exception as e:
            print(f"Could not get default workflow state: {e}")
        return None
    
    def create_story(self, name: str, description: str, story_type: str = "feature", project_id: Optional[int] = None) -> Optional[Dict]:
        """Create a new story in Shortcut"""
        
        payload = {
            "name": name,  # Required field
            "description": description,
            "story_type": story_type
        }
        
        # Try to add project_id first (preferred)
        if project_id:
            payload["project_id"] = project_id
        else:
            # Try to get a default project
            default_project = self.get_default_project_id()
            if default_project:
                payload["project_id"] = default_project
            else:
                # If no project, try workflow_state_id (not workflow_id!)
                default_workflow_state = self.get_default_workflow_state_id()
                if default_workflow_state:
                    payload["workflow_state_id"] = default_workflow_state
                else:
                    raise Exception("Workspace requires either project_id or workflow_state_id, but none found")
        
        try:
            response = requests.post(
                f"{self.base_url}/stories",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 422:
                error_data = response.json() if response.content else {}
                raise Exception(f"Validation error: {error_data}")
            elif response.status_code == 400:
                error_data = response.json() if response.content else {}
                raise Exception(f"Bad request error: {error_data}")
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise Exception(f"Failed to create Shortcut story: {str(e)}")


# Test function for debugging
def test_plain_api_standalone(api_key: str, test_email: str = "test@example.com"):
    """Standalone test function for Plain API"""
    print("🧪 Testing Plain API...")
    
    client = PlainClient(api_key)
    
    # Test the debug method
    result = client.debug_create_thread(
        title="Test Thread - Debug",
        description="This is a test thread for debugging",
        customer_email=test_email,
        priority="MEDIUM"
    )
    
    print(f"\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2))
    
    return result