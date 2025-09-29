"""
GitLab Manager Component

Handles all GitLab-related operations including:
- Authentication and token management
- File fetching from GitLab repositories
- API interactions and error handling
- GitLab URL processing and validation
"""

import re
import aiohttp
from typing import Optional, Dict, Any
from urllib.parse import urlparse, unquote


class GitLabManager:
    """
    Manages GitLab authentication and file operations.
    """
    
    def __init__(self):
        self.api_timeout = 30
    
    def validate_gitlab_token(self, token: str) -> bool:
        """
        Validate GitLab token format.
        
        Args:
            token: GitLab access token
            
        Returns:
            True if token format is valid, False otherwise
        """
        if not token:
            return False
        
        # Basic token format validation
        # GitLab tokens are typically alphanumeric with some special chars
        if len(token) < 10:  # Too short
            return False
        
        # Check for basic token pattern
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', token))
    
    def extract_gitlab_info(self, url: str) -> Optional[Dict[str, str]]:
        """
        Extract GitLab repository information from URL.
        
        Args:
            url: GitLab URL
            
        Returns:
            Dictionary with host, project_path, file_path, branch info or None
        """
        try:
            # Parse the URL
            parsed = urlparse(url)
            
            if 'gitlab' not in parsed.netloc.lower():
                return None
            
            # Extract path components
            path_parts = parsed.path.strip('/').split('/')
            
            if len(path_parts) < 2:
                return None
            
            # Standard GitLab URL patterns:
            # https://gitlab.com/user/project/-/blob/branch/path/to/file
            # https://gitlab.com/user/project/-/raw/branch/path/to/file
            
            host = f"{parsed.scheme}://{parsed.netloc}"
            
            # Find the project separator (-/blob, -/raw, etc.)
            separator_idx = -1
            for i, part in enumerate(path_parts):
                if part.startswith('-'):
                    separator_idx = i
                    break
            
            if separator_idx == -1:
                # No separator found, treat as simple project path
                project_path = '/'.join(path_parts)
                return {
                    'host': host,
                    'project_path': project_path,
                    'file_path': '',
                    'branch': 'main'
                }
            
            # Extract project path (everything before separator)
            project_path = '/'.join(path_parts[:separator_idx])
            
            # Extract file info (everything after separator)
            remaining_parts = path_parts[separator_idx:]
            
            # Determine branch and file path
            branch = 'main'
            file_path = ''
            
            if len(remaining_parts) > 2:
                # Format: -/blob/branch/path/to/file
                if remaining_parts[1] in ['blob', 'raw']:
                    branch = remaining_parts[2] if len(remaining_parts) > 2 else 'main'
                    file_path = '/'.join(remaining_parts[3:]) if len(remaining_parts) > 3 else ''
            
            return {
                'host': host,
                'project_path': project_path,
                'file_path': file_path,
                'branch': branch
            }
            
        except Exception as e:
            print(f"Error parsing GitLab URL {url}: {e}")
            return None
    
    def build_gitlab_api_url(self, gitlab_info: Dict[str, str]) -> str:
        """
        Build GitLab API URL for file access.
        
        Args:
            gitlab_info: Dictionary with GitLab information
            
        Returns:
            API URL for file access
        """
        host = gitlab_info['host']
        project_path = gitlab_info['project_path']
        file_path = gitlab_info['file_path']
        branch = gitlab_info['branch']
        
        # Encode project path for URL
        encoded_project = project_path.replace('/', '%2F')
        
        # Build API URL
        if file_path:
            # Specific file
            encoded_file_path = file_path.replace('/', '%2F')
            api_url = f"{host}/api/v4/projects/{encoded_project}/repository/files/{encoded_file_path}/raw"
            api_url += f"?ref={branch}"
        else:
            # Repository root - get README or list files
            api_url = f"{host}/api/v4/projects/{encoded_project}/repository/files/README.md/raw"
            api_url += f"?ref={branch}"
        
        return api_url
    
    async def fetch_gitlab_file(self, url: str, token: str) -> Optional[str]:
        """
        Fetch file content from GitLab using API.
        
        Args:
            url: GitLab file URL
            token: GitLab access token
            
        Returns:
            File content as string or None if failed
        """
        if not self.validate_gitlab_token(token):
            print(f"Invalid GitLab token format")
            return None
        
        gitlab_info = self.extract_gitlab_info(url)
        if not gitlab_info:
            print(f"Could not parse GitLab URL: {url}")
            return None
        
        api_url = self.build_gitlab_api_url(gitlab_info)
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers, timeout=self.api_timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        return content
                    elif response.status == 404:
                        # Try alternative file extensions for README
                        if 'README.md' in api_url:
                            for alt_readme in ['README.rst', 'README.txt', 'README']:
                                alt_url = api_url.replace('README.md', alt_readme)
                                async with session.get(alt_url, headers=headers, timeout=self.api_timeout) as alt_response:
                                    if alt_response.status == 200:
                                        return await alt_response.text()
                        
                        print(f"File not found: {url} (API: {api_url})")
                        return None
                    elif response.status == 401:
                        print(f"Authentication failed for GitLab. Check your token.")
                        return None
                    elif response.status == 403:
                        print(f"Access forbidden for GitLab repository. Check permissions.")
                        return None
                    else:
                        print(f"GitLab API error {response.status}: {await response.text()}")
                        return None
                        
        except aiohttp.ClientError as e:
            print(f"Network error accessing GitLab API: {e}")
            return None
        except Exception as e:
            print(f"Error fetching from GitLab: {e}")
            return None
    
    def convert_gitlab_to_raw_url(self, url: str) -> str:
        """
        Convert GitLab blob URL to raw URL.
        
        Args:
            url: GitLab blob URL
            
        Returns:
            Raw URL for direct file access
        """
        # Convert blob URLs to raw URLs for direct access
        if '/-/blob/' in url:
            return url.replace('/-/blob/', '/-/raw/')
        
        return url
    
    async def test_gitlab_connection(self, token: str, host: str = "https://gitlab.com") -> bool:
        """
        Test GitLab connection and token validity.
        
        Args:
            token: GitLab access token
            host: GitLab host URL
            
        Returns:
            True if connection successful, False otherwise
        """
        if not self.validate_gitlab_token(token):
            return False
        
        test_url = f"{host}/api/v4/user"
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(test_url, headers=headers, timeout=self.api_timeout) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        print(f"GitLab connection successful. User: {user_data.get('username', 'unknown')}")
                        return True
                    else:
                        print(f"GitLab connection failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"Error testing GitLab connection: {e}")
            return False
    
    def is_gitlab_url(self, url: str) -> bool:
        """
        Check if URL is from GitLab.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is from GitLab, False otherwise
        """
        try:
            parsed = urlparse(url)
            return 'gitlab' in parsed.netloc.lower()
        except:
            return False
    
    async def get_project_info(self, project_path: str, token: str, host: str = "https://gitlab.com") -> Optional[Dict[str, Any]]:
        """
        Get GitLab project information.
        
        Args:
            project_path: Project path (e.g., "user/project")
            token: GitLab access token
            host: GitLab host URL
            
        Returns:
            Project information dictionary or None
        """
        if not self.validate_gitlab_token(token):
            return None
        
        encoded_project = project_path.replace('/', '%2F')
        api_url = f"{host}/api/v4/projects/{encoded_project}"
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers, timeout=self.api_timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Failed to get project info: {response.status}")
                        return None
                        
        except Exception as e:
            print(f"Error getting project info: {e}")
            return None
