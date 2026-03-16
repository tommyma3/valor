"""
Python Interpreter Tool for IterResearch.

Provides sandboxed Python code execution capability.
"""
import re
import json
import time
import random
import requests
from typing import Dict, List, Optional, Union
from requests.exceptions import Timeout

try:
    from config import SANDBOX_ENDPOINTS
except ImportError:
    import os
    SANDBOX_ENDPOINTS = os.getenv("SANDBOX_ENDPOINTS", "http://127.0.0.1:8080").split(",")


class PythonInterpreter:
    """
    Python code interpreter with sandboxed execution.
    
    Executes Python code in a secure, isolated environment.
    Supports multiple sandbox endpoints for load balancing.
    """
    
    name = "python_interpreter"
    description = '''Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.
**Make sure to use print() for any output you want to see in the results.**'''
    
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Remember to use print() statements for any output you want to see.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, endpoints: Optional[List[str]] = None, timeout: int = 50):
        """
        Initialize the Python interpreter.
        
        Args:
            endpoints: List of sandbox endpoint URLs. Falls back to config if not provided.
            timeout: Default execution timeout in seconds.
        """
        self.endpoints = endpoints or SANDBOX_ENDPOINTS
        self.default_timeout = timeout

    def _extract_code(self, code: str) -> str:
        """
        Extract code from markdown code blocks if present.
        
        Args:
            code: Raw code string, potentially wrapped in markdown.
            
        Returns:
            Clean code string.
        """
        # Try to extract from triple backticks
        triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
        if triple_match:
            return triple_match.group(1)
        
        # Try to extract from <code> tags
        code_match = re.search(r'<code>(.*?)</code>', code, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        return code

    def _execute_on_endpoint(self, code: str, endpoint: str, timeout: int) -> tuple:
        """
        Execute code on a specific endpoint.
        
        Args:
            code: Python code to execute.
            endpoint: Sandbox endpoint URL.
            timeout: Execution timeout.
            
        Returns:
            Tuple of (success, result, execution_time).
        """
        try:
            payload = {
                "code": code,
                "language": "python",
                "run_timeout": timeout
            }
            
            response = requests.post(
                f"{endpoint}/run",
                json=payload,
                timeout=timeout + 10
            )
            response.raise_for_status()
            result = response.json()
            
            output_parts = []
            
            if result.get("stdout"):
                output_parts.append(f"stdout:\n{result['stdout']}")
            
            if result.get("stderr"):
                output_parts.append(f"stderr:\n{result['stderr']}")
            
            execution_time = result.get("execution_time", 0)
            if execution_time >= timeout - 1:
                output_parts.append("[PythonInterpreter Error] TimeoutError: Execution timed out.")
            
            output = '\n'.join(output_parts)
            return True, output if output.strip() else "Finished execution.", execution_time
            
        except Timeout:
            return False, "[Python Interpreter Error] TimeoutError: Execution timed out.", None
        except Exception as e:
            return False, f"[Python Interpreter Error]: {str(e)}", None

    def call(self, params: Union[str, dict], timeout: Optional[int] = None, **kwargs) -> str:
        """
        Execute Python code.
        
        Args:
            params: Either a JSON string or dict with 'code' field.
            timeout: Execution timeout in seconds.
            
        Returns:
            Execution output or error message.
        """
        timeout = timeout or self.default_timeout
        
        # Parse parameters
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                # Treat as raw code
                params = {"code": params}
        
        code = params.get("code", params.get("raw", ""))
        code = self._extract_code(code)
        
        if not code.strip():
            return "[Python Interpreter Error]: Empty code."
        
        if not self.endpoints:
            return "[Python Interpreter Error]: No sandbox endpoints configured. Please set SANDBOX_ENDPOINTS in environment."
        
        # Try endpoints with retry
        max_attempts = min(8, len(self.endpoints) * 2)
        last_error = None
        
        for attempt in range(max_attempts):
            endpoint = random.choice(self.endpoints)
            
            success, result, exec_time = self._execute_on_endpoint(code, endpoint, timeout)
            
            if success:
                return result
            
            last_error = result
            time.sleep(1)
        
        return last_error if last_error else "[Python Interpreter Error]: All attempts failed."


